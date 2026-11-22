# ============================================================
# STEP 29 - FULLY NESTED ESTIMATE OF THE WHOLE SELECTION PROCEDURE
# ============================================================
# The "top-3" models of step 23/16 are ranked with inner-CV scores averaged over
# ALL outer folds, so their outer-fold test numbers are mildly optimistic
# (the ranking indirectly used every subject). The unbiased estimate of the
# procedure "tune each of 16 models, keep the best by inner-CV accuracy" is
# obtained by making that choice INSIDE each outer training fold only:
#   * hyperparameters  : tuned by inner CV on the outer-training subjects (step 23)
#   * model family     : the family with the highest inner-CV accuracy in THAT fold
#   * evaluation       : that fold's winner, on that fold's unseen subjects
# Hyperparameters chosen in each fold are saved (nested_selection_folds.csv: best_parameters).
# Predictions of the 5 winners are pooled (51 subjects, each predicted once by a
# model that never saw them) and bootstrapped at subject level.

import os
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, recall_score)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUB = os.environ.get("RESULTS_SUBDIR", "model_benchmark")          # same override as step 23
BENCH = os.path.join(BASE_DIR, "results", SUB)
OUT = os.path.join(BASE_DIR, "results", SUB.replace("model_benchmark", "nested_selection"))
os.makedirs(OUT, exist_ok=True)
SEED, N_BOOT = 42, 10000

folds = pd.read_csv(os.path.join(BENCH, "model_benchmark_fold_results.csv"))
preds = pd.read_csv(os.path.join(BENCH, "model_benchmark_predictions.csv"))

# Winner per fold: highest inner-CV accuracy; deterministic tie-break = model name.
winners = (folds.sort_values(["fold", "inner_accuracy", "model"], ascending=[True, False, True], kind="mergesort")
                .groupby("fold", as_index=False).head(1).sort_values("fold"))
winners[["fold", "model", "inner_accuracy", "best_parameters", "accuracy", "balanced_accuracy",
         "f1", "roc_auc"]].to_csv(os.path.join(OUT, "nested_selection_folds.csv"), index=False)

rows = [preds[(preds.model == w.model) & (preds.fold == w.fold)].assign(winner=w.model)
        for w in winners.itertuples()]
P = pd.concat(rows, ignore_index=True)
assert P.subject.is_unique and len(P) == preds.subject.nunique(), "each subject must be predicted exactly once"
P.to_csv(os.path.join(OUT, "nested_selection_predictions.csv"), index=False)

y = P.true_target.to_numpy(); yp = P.predicted_target.to_numpy()

def metrics(y, yp):
    return {
        "Accuracy": accuracy_score(y, yp),
        "Balanced_Accuracy": balanced_accuracy_score(y, yp),
        "F1": f1_score(y, yp, zero_division=0),
        "Sensitivity": recall_score(y, yp, pos_label=1, zero_division=0),
        "Specificity": recall_score(y, yp, pos_label=0, zero_division=0),
        "MCC": matthews_corrcoef(y, yp),
        # NOTE: no pooled AUC. Fold winners are different model families whose scores (probabilities vs raw SVM
        # decision values) are on different scales, so a pooled AUC would be meaningless. Per-fold AUCs are in
        # nested_selection_folds.csv and summarised as mean +/- SD below.
    }

point = metrics(y, yp)
rng = np.random.default_rng(SEED)
n = len(y); boots = {k: [] for k in point}
for _ in range(N_BOOT):
    idx = rng.integers(0, n, n)
    if len(np.unique(y[idx])) < 2:
        continue
    for k, v in metrics(y[idx], yp[idx]).items():
        boots[k].append(v)
out = pd.DataFrame([{"metric": k, "point_estimate": point[k],
                     "ci_lower_95": np.nanpercentile(boots[k], 2.5),
                     "ci_upper_95": np.nanpercentile(boots[k], 97.5),
                     "n_subjects": n, "n_bootstrap": N_BOOT} for k in point])
out.to_csv(os.path.join(OUT, "nested_selection_ci.csv"), index=False)
auc = winners["roc_auc"]
pd.DataFrame([{"mean_fold_roc_auc": auc.mean(), "sd_fold_roc_auc": auc.std(), "n_folds": len(auc)}]).to_csv(
    os.path.join(OUT, "nested_selection_auc.csv"), index=False)

print("STEP 29 - fully nested selection estimate (family + hyperparameters chosen inside each outer fold)")
print(winners[["fold", "model", "inner_accuracy", "accuracy", "roc_auc"]].to_string(index=False))
print()
print(out.to_string(index=False))
