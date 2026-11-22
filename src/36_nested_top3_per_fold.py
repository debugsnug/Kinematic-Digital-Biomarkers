# ============================================================
# STEP 36 - PER-OUTER-FOLD TOP-3 SELECTION AND VALIDATION (selection-unbiased version of requirement 6)
# ============================================================
# Step 16 validates a GLOBAL top-3 whose ranking pools inner-CV scores over all outer folds (optimistic).
# Here the top-3 families are chosen SEPARATELY inside each outer training fold, from inner-CV accuracy on
# that fold's training subjects only (deterministic tie-break: model name), and then evaluated on that fold's
# unseen subjects. No outer-test information enters any selection. Reported:
#   rank-1/2/3 slots (the model ranked 1st/2nd/3rd in each fold; families can differ between folds),
#   a majority vote of the three selected models (label vote; no score averaging across model families).
# This is the PRIMARY top-3 validation result. Fold-specific model identities are expected to vary.
# Bootstrap intervals below resample the 51 subject-level out-of-fold predictions; they are conditional on
# the fitted fold models and do not represent external-site/cohort uncertainty.
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef, recall_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUB = os.environ.get("RESULTS_SUBDIR", "model_benchmark")
BENCH = os.path.join(BASE, "results", SUB)
OUT = os.path.join(BASE, "results", SUB.replace("model_benchmark", "nested_top3_per_fold"))
os.makedirs(OUT, exist_ok=True)
SEED, N_BOOT = 42, 10000

folds = pd.read_csv(os.path.join(BENCH, "model_benchmark_fold_results.csv"))
preds = pd.read_csv(os.path.join(BENCH, "model_benchmark_predictions.csv"))
ranked = folds.sort_values(["fold", "inner_accuracy", "model"], ascending=[True, False, True], kind="mergesort")
ranked["rank"] = ranked.groupby("fold").cumcount() + 1
top = ranked[ranked["rank"] <= 3]
top[["fold", "rank", "model", "inner_accuracy", "best_parameters", "accuracy", "balanced_accuracy", "f1", "roc_auc"]].to_csv(
    os.path.join(OUT, "per_fold_top3_selection.csv"), index=False)

def met(y, p):
    return {"Accuracy": accuracy_score(y, p), "Balanced_Accuracy": balanced_accuracy_score(y, p), "F1": f1_score(y, p, zero_division=0),
            "Sensitivity": recall_score(y, p, pos_label=1, zero_division=0), "Specificity": recall_score(y, p, pos_label=0, zero_division=0),
            "MCC": matthews_corrcoef(y, p)}

pooled = {}
for r in (1, 2, 3):
    rows = [preds[(preds.model == t.model) & (preds.fold == t.fold)] for t in top[top["rank"] == r].itertuples()]
    pooled[f"rank{r}_slot"] = pd.concat(rows).sort_values("subject").reset_index(drop=True)
subs = pooled["rank1_slot"].subject.to_numpy()
for k, v in pooled.items():
    assert (v.subject.to_numpy() == subs).all() and v.subject.is_unique and len(v) == 51
y = pooled["rank1_slot"].true_target.to_numpy()
votes = np.stack([pooled[f"rank{r}_slot"].predicted_target.to_numpy() for r in (1, 2, 3)])
vote = (votes.sum(axis=0) >= 2).astype(int)                      # 3 voters, no ties
cand = {"rank1_slot": pooled["rank1_slot"].predicted_target.to_numpy(), "rank2_slot": votes[1], "rank3_slot": votes[2], "majority_vote_of_top3": vote}

rng = np.random.default_rng(SEED); rows = []
# Vectorised subject bootstrap: same paired subject resampling, without 40,000 Python/sklearn calls.
boot_idx = rng.integers(0, len(y), size=(N_BOOT, len(y)))
yb = y[boot_idx]
valid = (yb == 1).sum(axis=1) > 0
valid &= (yb == 0).sum(axis=1) > 0

def boot_confusion_metrics(yb, pb):
    tp = ((yb == 1) & (pb == 1)).sum(axis=1).astype(float)
    tn = ((yb == 0) & (pb == 0)).sum(axis=1).astype(float)
    fp = ((yb == 0) & (pb == 1)).sum(axis=1).astype(float)
    fn = ((yb == 1) & (pb == 0)).sum(axis=1).astype(float)
    acc = (tp + tn) / (tp + tn + fp + fn)
    sens = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    spec = np.divide(tn, tn + fp, out=np.zeros_like(tn), where=(tn + fp) > 0)
    f1v = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=(2 * tp + fp + fn) > 0)
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(tp * tn - fp * fn, denom, out=np.zeros_like(tp), where=denom > 0)
    return {"Accuracy": acc, "Balanced_Accuracy": (sens + spec) / 2, "F1": f1v,
            "Sensitivity": sens, "Specificity": spec, "MCC": mcc}

for name, p in cand.items():
    pt = met(y, p)
    pb = p[boot_idx]
    boot_values = boot_confusion_metrics(yb, pb)
    for k in pt:
        values = boot_values[k][valid]
        rows.append({"procedure": name, "metric": k, "point_estimate": pt[k],
                     "ci_lower_95": np.nanpercentile(values, 2.5), "ci_upper_95": np.nanpercentile(values, 97.5)})
R = pd.DataFrame(rows); R.to_csv(os.path.join(OUT, "per_fold_top3_pooled_metrics.csv"), index=False)

# Fold-level dispersion for the three rank slots and the majority-vote procedure.
# Each fold contributes its own held-out subjects; no fold is used to choose another fold's model.
fold_metric_rows = []
for fold in sorted(top.fold.unique()):
    fold_rows = top[top.fold == fold].sort_values("rank")
    fold_pred_parts = []
    for rank in (1, 2, 3):
        chosen = fold_rows[fold_rows["rank"] == rank].iloc[0]
        part = preds[(preds.model == chosen.model) & (preds.fold == fold)].sort_values("subject")
        fold_pred_parts.append(part)
        mm = met(part.true_target.to_numpy(), part.predicted_target.to_numpy())
        fold_metric_rows.append({"fold": int(fold), "procedure": f"rank{rank}_slot", **mm})
    yy = fold_pred_parts[0].true_target.to_numpy()
    pp = np.stack([part.predicted_target.to_numpy() for part in fold_pred_parts])
    vv = (pp.sum(axis=0) >= 2).astype(int)
    fold_metric_rows.append({"fold": int(fold), "procedure": "majority_vote_of_top3", **met(yy, vv)})
fold_metrics = pd.DataFrame(fold_metric_rows)
fold_metrics.to_csv(os.path.join(OUT, "per_fold_top3_fold_metrics.csv"), index=False)
fold_summary = fold_metrics.groupby("procedure").agg({
    "Accuracy": ["mean", "std"], "Balanced_Accuracy": ["mean", "std"], "F1": ["mean", "std"], "MCC": ["mean", "std"]
})
fold_summary.columns = ["_".join(x) for x in fold_summary.columns]
fold_summary.reset_index().to_csv(os.path.join(OUT, "per_fold_top3_fold_summary.csv"), index=False)
pd.DataFrame({"subject": subs, "true_target": y, **{k: v for k, v in cand.items()}}).to_csv(os.path.join(OUT, "per_fold_top3_predictions.csv"), index=False)
print("STEP 36 - PRIMARY: per-fold top-3 selection (inner CV only) evaluated on unseen subjects")
print(top[["fold", "rank", "model", "inner_accuracy", "accuracy"]].to_string(index=False))
print(R[R.metric.isin(["Accuracy", "Balanced_Accuracy", "MCC"])].round(3).to_string(index=False))
