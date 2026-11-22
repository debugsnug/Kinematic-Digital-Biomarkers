# ============================================================
# STEP 16 - GLOBAL TOP-3 DIAGNOSTIC (NOT THE PRIMARY UNBIASED VALIDATION)
# ============================================================
# The families come from results/model_benchmark/selected_top_models.json (ranked by inner-CV accuracy only).
# They are re-run here through an INDEPENDENT code path - a real sklearn Pipeline (imputer -> scaler -> mutual-
# information SelectKBest -> model) inside GridSearchCV(StratifiedGroupKFold, groups=subject) - using exactly the
# SAME definitions, grids, class weights, scoring and folds as step 23 (shared model_catalog.py). The predictions
# are compared with the benchmark's saved out-of-fold predictions (cross-check), and all extended metrics are
# computed from the saved out-of-fold predictions. Brier score only for models with genuine probabilities.
# IMPORTANT: The global ranking pools inner-CV scores from all outer folds. A subject held out in one outer
# fold can appear in the training data of other outer folds and therefore influence this global ranking.
# Consequently these numbers are DIAGNOSTIC ONLY and MUST NOT be reported as an unbiased final test estimate.
# The primary top-3 validation is src/36_nested_top3_per_fold.py: each fold selects its own top three using
# only that fold's training subjects and evaluates them on that fold's untouched subjects.

import json
import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
import model_catalog as mc

warnings.filterwarnings("ignore")
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "experiment_dataset.csv")
BENCH = os.path.join(BASE_DIR, "results", "model_benchmark")
OUT_DIR = os.path.join(BASE_DIR, "results", "top3_validation")
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 70 + "\nSTEP 16 - GLOBAL TOP-3 DIAGNOSTIC ONLY (NOT PRIMARY VALIDATION)\n" + "=" * 70)
print("WARNING: pooled global ranking is selection-optimistic. Use step 36 for primary top-3 validation.", flush=True)
df = pd.read_csv(DATA_PATH)
y = df["target"].to_numpy(); groups = df["subject"].to_numpy()
X = df.drop(columns=["target", "subject", "side", "group"], errors="ignore").select_dtypes(include=[np.number])
print(f"Dataset {df.shape}; predictors {X.shape[1]}; subjects {len(np.unique(groups))}; classes {dict(zip(*np.unique(y, return_counts=True)))}")

sel_path = os.path.join(BENCH, "selected_top_models.json")
if not os.path.exists(sel_path):
    raise FileNotFoundError(f"{sel_path} not found. Run 23_model_benchmark.py first.")
selected = json.load(open(sel_path, encoding="utf-8"))["selected_models"]
if not 2 <= len(selected) <= 3:
    raise ValueError(f"Expected 2-3 selected models, found {selected}")
print(f"Selected by inner-CV accuracy: {selected}")
bench_pred = pd.read_csv(os.path.join(BENCH, "model_benchmark_predictions.csv"))
bench_fold = pd.read_csv(os.path.join(BENCH, "model_benchmark_fold_results.csv"))

outer_cv, inner_cv = mc.make_cv(mc.SEED)
fold_rows, pred_rows, param_rows, cm_rows, check_rows = [], [], [], [], []
ALL_FAMILIES = selected + [m for m in mc.get_candidate_models() if m not in selected]   # cross-check covers all 16
for name in ALL_FAMILIES:
    is_sel = name in selected
    print(f"\n{'VALIDATING' if is_sel else 'CROSS-CHECKING (not selected)'} {name}", flush=True)
    for fold, (tr, te) in enumerate(outer_cv.split(X, y, groups), start=1):
        assert not set(groups[tr]) & set(groups[te])
        gs = mc.fit_nested(name, X.iloc[tr], y[tr], groups[tr], inner_cv)       # tuning on training subjects only
        best = gs.best_estimator_
        y_pred = best.predict(X.iloc[te]); y_score, stype = mc.score_of(best, X.iloc[te])
        yt = y[te]
        tn, fp, fn, tp = confusion_matrix(yt, y_pred, labels=[0, 1]).ravel()
        if is_sel: fold_rows.append({
            "model": name, "fold": fold, "n_train_subjects": len(np.unique(groups[tr])),
            "n_test_subjects": len(np.unique(groups[te])),
            "accuracy": accuracy_score(yt, y_pred), "balanced_accuracy": balanced_accuracy_score(yt, y_pred),
            "f1": f1_score(yt, y_pred, zero_division=0), "precision": precision_score(yt, y_pred, zero_division=0),
            "recall_sensitivity": recall_score(yt, y_pred, zero_division=0),
            "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
            "roc_auc": roc_auc_score(yt, y_score), "pr_auc": average_precision_score(yt, y_score),
            "mcc": matthews_corrcoef(yt, y_pred),
            "brier_score": brier_score_loss(yt, y_score) if stype == "probability" else np.nan, "score_type": stype})
        if is_sel: cm_rows.append({"model": name, "fold": fold, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
        for s, a, p, sc in (zip(groups[te], yt, y_pred, y_score) if is_sel else []):
            pred_rows.append({"model": name, "fold": fold, "subject": s, "true_target": int(a),
                              "predicted_target": int(p), "predicted_score": float(sc), "score_type": stype})
        bp = {"k": gs.best_params_["selector__k"], **{k[7:]: v for k, v in gs.best_params_.items() if k.startswith("model__")}}
        if is_sel: param_rows.append({"model": name, "fold": fold, "best_params": str(bp), "best_inner_accuracy": gs.best_score_})
        # cross-check against the benchmark's pre-fitted implementation
        b = bench_pred[(bench_pred.model == name) & (bench_pred.fold == fold)].set_index("subject")
        mine = pd.DataFrame({"subject": groups[te], "pred": y_pred}).set_index("subject")
        same_pred = bool((b.loc[mine.index, "predicted_target"].to_numpy() == mine["pred"].to_numpy()).all())
        bf = bench_fold[(bench_fold.model == name) & (bench_fold.fold == fold)].iloc[0]
        check_rows.append({"model": name, "selected": is_sel, "fold": fold, "predictions_identical_to_benchmark": same_pred,
                           "best_params_identical_to_benchmark": str(bp) == bf["best_parameters"],
                           "inner_accuracy_diff": abs(gs.best_score_ - bf["inner_accuracy"])})
        print(f"  fold {fold}: identical-to-benchmark={same_pred}  params={bp}", flush=True)

results_df, pred_df = pd.DataFrame(fold_rows), pd.DataFrame(pred_rows)
metrics = ["accuracy", "balanced_accuracy", "f1", "precision", "recall_sensitivity", "specificity", "roc_auc", "pr_auc", "mcc", "brier_score"]
summary = results_df.groupby("model")[metrics].agg(["mean", "std"])
summary.columns = ["_".join(c) for c in summary.columns]
summary = summary.reset_index()
results_df.to_csv(os.path.join(OUT_DIR, "top3_fold_results.csv"), index=False)
summary.to_csv(os.path.join(OUT_DIR, "top3_summary.csv"), index=False)
pred_df.to_csv(os.path.join(OUT_DIR, "top3_predictions.csv"), index=False)
pd.DataFrame(param_rows).to_csv(os.path.join(OUT_DIR, "top3_best_parameters.csv"), index=False)
pd.DataFrame(cm_rows).to_csv(os.path.join(OUT_DIR, "top3_confusion_matrices.csv"), index=False)
chk = pd.DataFrame(check_rows); chk.to_csv(os.path.join(OUT_DIR, "top3_crosscheck_vs_benchmark.csv"), index=False)

print("\n" + "=" * 70 + "\nCROSS-CHECK: Pipeline/GridSearchCV path vs benchmark pre-fitted path\n" + "=" * 70)
print(f"predictions identical in {int(chk.predictions_identical_to_benchmark.sum())}/{len(chk)} model-folds; "
      f"best parameters identical in {int(chk.best_params_identical_to_benchmark.sum())}/{len(chk)}; "
      f"max |inner accuracy diff| = {chk.inner_accuracy_diff.max():.2e}")
print("\n" + "=" * 70 + "\nGLOBAL TOP-3 DIAGNOSTIC SUMMARY (outer folds; OPTIMISTIC selection)\n" + "=" * 70)
print(summary.round(4).to_string(index=False))
print("\nSTEP 16 COMPLETE")
