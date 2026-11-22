# ============================================================
# STEP 23 - 16-MODEL BENCHMARK WITH LEAK-FREE INNER SELECTION
# ============================================================
# Nested subject-wise CV: outer 5-fold StratifiedGroupKFold (test subjects), inner 4-fold (tuning).
# Model definitions / grids / class weights come from model_catalog.py (shared with steps 16, 20, 27, 31, 34).
# Imputer, scaler and mutual-information SelectKBest are fitted on training subjects only; for speed they are
# pre-fitted once per training fold and per inner-training fold (mathematically identical to a Pipeline -
# step 16 re-runs the selected models through a real sklearn Pipeline + GridSearchCV and asserts identical predictions).
#
# Environment overrides (used by the runner for the secondary / ablation feature sets and repeated partitions):
#   EXPERIMENT_DATASET  file in data/processed        (default experiment_dataset.csv = PRIMARY coordinate-only)
#   RESULTS_SUBDIR      sub-folder of results/        (default model_benchmark)
#   CV_SEED             seed of the outer/inner splitters (default 42)

import json
import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
import model_catalog as mc

warnings.filterwarnings("ignore")

DATASET = os.environ.get("EXPERIMENT_DATASET", "experiment_dataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", os.environ.get("RESULTS_SUBDIR", "model_benchmark"))
CV_SEED = int(os.environ.get("CV_SEED", mc.SEED))
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("STEP 23 - 16-MODEL BENCHMARK & LEAK-FREE SELECTION")
print(f"dataset={DATASET}  output={os.path.relpath(OUTPUT_DIR, BASE_DIR)}  cv_seed={CV_SEED}")
print("=" * 70)

df = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", DATASET))
y = df["target"].to_numpy()
groups = df["subject"].to_numpy()
X_df = df.drop(columns=["target", "subject", "side", "group"], errors="ignore").select_dtypes(include=[np.number])
feature_names = X_df.columns.tolist()
X = X_df.to_numpy(dtype=float)
print(f"Dataset shape  : {df.shape}\nSubjects       : {len(np.unique(groups))}\nPredictors     : {X.shape[1]}")
print(f"TD (class 0)   : {(y == 0).sum()}\nDCP (class 1)  : {(y == 1).sum()}")

outer_cv, inner_cv = mc.make_cv(CV_SEED)
models_dict = mc.get_candidate_models()
assert len(models_dict) == 16
print(f"\nBenchmarking {len(models_dict)} candidate classifiers.")


def prefit(X_tr, y_tr, X_te, k):
    imp, sc = SimpleImputer(strategy="median"), StandardScaler()
    sel = SelectKBest(score_func=mc.mi_score, k=k)
    A = sel.fit_transform(sc.fit_transform(imp.fit_transform(X_tr)), y_tr)
    B = sel.transform(sc.transform(imp.transform(X_te)))
    return A, B


print("\nPreparing leak-free fold representations (fit on training subjects only)...")
fold_rows, outer_fold_data = [], []
for fold_idx, (tr, te) in enumerate(outer_cv.split(X, y, groups), start=1):
    assert not set(groups[tr]) & set(groups[te]), "subject overlap between outer train and test"
    fold_rows += [{"subject": s, "outer_fold": fold_idx} for s in groups[te]]
    outer_k = {k: prefit(X[tr], y[tr], X[te], k) for k in mc.K_VALUES}
    inner_k = {k: [] for k in mc.K_VALUES}
    for itr, ival in inner_cv.split(X[tr], y[tr], groups[tr]):
        assert not set(groups[tr][itr]) & set(groups[tr][ival]), "subject overlap between inner train and validation"
        for k in mc.K_VALUES:
            A, B = prefit(X[tr][itr], y[tr][itr], X[tr][ival], k)
            inner_k[k].append((A, y[tr][itr], B, y[tr][ival]))
    outer_fold_data.append({"fold": fold_idx, "y_tr": y[tr], "y_te": y[te], "grp_te": groups[te],
                            "outer_k": outer_k, "inner_k": inner_k})
pd.DataFrame(fold_rows).sort_values("subject").to_csv(os.path.join(OUTPUT_DIR, "outer_fold_assignment.csv"), index=False)
print("Representations ready. Starting benchmark...")

summary_results, fold_results, prediction_results = [], [], []
for model_num, (name, (cls, fixed, grid)) in enumerate(models_dict.items(), start=1):
    print(f"\n[{model_num:02d}/16] Evaluating {name}...", flush=True)
    combos = [mc.split_params(p) for p in ParameterGrid(mc.sk_grid(name))]       # same order as GridSearchCV
    metrics_this = []
    for fd in outer_fold_data:
        best_acc, best = -1.0, None
        for k, kw in combos:
            accs = []
            for A, ya, B, yb in fd["inner_k"][k]:
                m = cls(**{**fixed, **kw}).fit(A, ya)
                accs.append(accuracy_score(yb, m.predict(B)))
            if float(np.mean(accs)) > best_acc:                                  # strict >: first best wins (as GridSearchCV)
                best_acc, best = float(np.mean(accs)), (k, kw)
        k, kw = best
        A, B = fd["outer_k"][k]
        final = cls(**{**fixed, **kw}).fit(A, fd["y_tr"])
        y_pred = final.predict(B)
        y_score, score_type = mc.score_of(final, B)
        y_te = fd["y_te"]
        tn, fp, fn, tp = confusion_matrix(y_te, y_pred, labels=[0, 1]).ravel()
        res = {"model": name, "fold": fd["fold"], "inner_accuracy": best_acc,
               "accuracy": accuracy_score(y_te, y_pred), "balanced_accuracy": balanced_accuracy_score(y_te, y_pred),
               "f1": f1_score(y_te, y_pred, zero_division=0), "precision": precision_score(y_te, y_pred, zero_division=0),
               "recall_sensitivity": recall_score(y_te, y_pred, zero_division=0),
               "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
               "roc_auc": roc_auc_score(y_te, y_score), "pr_auc": average_precision_score(y_te, y_score),
               "mcc": matthews_corrcoef(y_te, y_pred),
               # Brier only for genuine probabilities; decision-function values are never treated as probabilities
               "brier_score": brier_score_loss(y_te, y_score) if score_type == "probability" else np.nan,
               "score_type": score_type,
               "best_parameters": str({"k": k, **kw})}
        metrics_this.append(res); fold_results.append(res)
        for s, a, p, sc in zip(fd["grp_te"], y_te, y_pred, y_score):
            prediction_results.append({"model": name, "fold": fd["fold"], "subject": s, "true_target": int(a),
                                       "predicted_target": int(p), "predicted_score": float(sc), "score_type": score_type})
    m_df = pd.DataFrame(metrics_this)
    row = {"model": name, "score_type": score_type, "inner_accuracy_mean": m_df.inner_accuracy.mean(),
           "inner_accuracy_std": m_df.inner_accuracy.std()}
    for col, nm in [("accuracy", "accuracy"), ("balanced_accuracy", "balanced_accuracy"), ("f1", "f1"),
                    ("precision", "precision"), ("recall_sensitivity", "recall_sensitivity"),
                    ("specificity", "specificity"), ("roc_auc", "roc_auc"), ("pr_auc", "pr_auc"), ("mcc", "mcc"),
                    ("brier_score", "brier_score")]:
        row[f"test_{nm}_mean"] = m_df[col].mean(); row[f"test_{nm}_std"] = m_df[col].std()
    summary_results.append(row)
    print(f"   InnerAcc={m_df.inner_accuracy.mean():.4f}+/-{m_df.inner_accuracy.std():.4f} | "
          f"TestAcc={m_df.accuracy.mean():.4f}+/-{m_df.accuracy.std():.4f} | "
          f"TestAUC={m_df.roc_auc.mean():.4f}+/-{m_df.roc_auc.std():.4f}", flush=True)

summary_df = (pd.DataFrame(summary_results)
              .sort_values(["inner_accuracy_mean", "model"], ascending=[False, True], kind="mergesort")
              .reset_index(drop=True))
fold_df, prediction_df = pd.DataFrame(fold_results), pd.DataFrame(prediction_results)
summary_df.to_csv(os.path.join(OUTPUT_DIR, "model_benchmark_summary.csv"), index=False)
fold_df.to_csv(os.path.join(OUTPUT_DIR, "model_benchmark_fold_results.csv"), index=False)
prediction_df.to_csv(os.path.join(OUTPUT_DIR, "model_benchmark_predictions.csv"), index=False)

top3 = summary_df.head(3)["model"].tolist()
record = {
    "dataset": DATASET, "n_predictors": len(feature_names), "cv_seed": CV_SEED,
    "selection_criterion": "inner_accuracy_mean on training folds only (no outer test score used in ranking; ties by model name). "
                           "Ranking pools inner-CV over all outer folds, so the top-3's outer-fold results are optimistic - "
                           "see results/nested_selection for the selection-unbiased estimate.",
    "n_selected": 3, "selected_models": top3,
    "details": [{"rank": r + 1, "model": row["model"], "inner_accuracy_mean": float(row["inner_accuracy_mean"]),
                 "inner_accuracy_std": float(row["inner_accuracy_std"]),
                 "test_accuracy_mean": float(row["test_accuracy_mean"]), "test_accuracy_std": float(row["test_accuracy_std"]),
                 "test_roc_auc_mean": float(row["test_roc_auc_mean"])} for r, row in summary_df.head(3).iterrows()],
}
with open(os.path.join(OUTPUT_DIR, "selected_top_models.json"), "w", encoding="utf-8") as f:
    json.dump(record, f, indent=4)

print("\n" + "=" * 70 + "\nBENCHMARK SUMMARY (SORTED BY INNER VALIDATION ACCURACY)\n" + "=" * 70)
print(summary_df[["model", "inner_accuracy_mean", "test_accuracy_mean", "test_balanced_accuracy_mean",
                  "test_f1_mean", "test_roc_auc_mean"]].to_string(index=False))
print(f"\nTOP-3 MODELS SELECTED FOR DEDICATED VALIDATION: {top3}\n" + "=" * 70 + "\nSTEP 23 COMPLETE")
