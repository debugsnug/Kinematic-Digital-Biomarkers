# ============================================================
# STEP 15 — BOOTSTRAP CONFIDENCE INTERVALS (TOP-3 MODELS)
# ============================================================

import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

TOP3_PRED_PATH = os.path.join(
    BASE_DIR,
    "results",
    "top3_validation",
    "top3_predictions.csv"
)

NESTED_PRED_PATH = os.path.join(
    BASE_DIR,
    "results",
    "nested_cv",
    "nested_cv_predictions.csv"
)

OUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "bootstrap_ci"
)

os.makedirs(
    OUT_DIR,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

N_BOOTSTRAPS = 10000
RANDOM_STATE = 42

rng = np.random.default_rng(
    RANDOM_STATE
)


# ============================================================
# LOAD PREDICTIONS
# ============================================================

print("=" * 70)
print("STEP 15 — BOOTSTRAP CONFIDENCE INTERVALS (10,000 RESAMPLES)")
print("=" * 70)

if os.path.exists(TOP3_PRED_PATH):
    PREDICTIONS_PATH = TOP3_PRED_PATH
    print(f"Loading top-3 predictions from: {TOP3_PRED_PATH}")
elif os.path.exists(NESTED_PRED_PATH):
    PREDICTIONS_PATH = NESTED_PRED_PATH
    print(f"Loading predictions from: {NESTED_PRED_PATH}")
else:
    PREDICTIONS_PATH = os.path.join(
        BASE_DIR,
        "results",
        "model_benchmark",
        "model_benchmark_predictions.csv"
    )
    print(f"Loading predictions from: {PREDICTIONS_PATH}")

pred = pd.read_csv(
    PREDICTIONS_PATH
)

if "predicted_score" in pred.columns and "predicted_probability" not in pred.columns:
    pred["predicted_probability"] = pred["predicted_score"]

print()
print(f"Prediction rows: {len(pred)}")
print(f"Models available: {pred['model'].unique().tolist()}")
print(f"Unique subjects: {pred['subject'].nunique()}")


# ============================================================
# CONFIDENCE INTERVAL FUNCTION
# ============================================================

def percentile_ci(values):
    lower = np.percentile(values, 2.5)
    upper = np.percentile(values, 97.5)
    return float(lower), float(upper)


# ============================================================
# BOOTSTRAP OVER ALL MODELS
# ============================================================

all_ci_rows = []

for model_name in pred["model"].unique():
    m_pred = pred[pred["model"] == model_name].copy()

    y_true = m_pred["true_target"].to_numpy()
    y_pred = m_pred["predicted_target"].to_numpy()
    y_prob = m_pred["predicted_probability"].to_numpy()

    # Point estimates
    acc_pt = accuracy_score(y_true, y_pred)
    bacc_pt = balanced_accuracy_score(y_true, y_pred)
    f1_pt = f1_score(y_true, y_pred, zero_division=0)
    auc_pt = roc_auc_score(y_true, y_prob)

    print()
    print("=" * 70)
    print(f"MODEL: {model_name} (Point Estimates & 10,000 Stratified Bootstraps)")
    print("=" * 70)
    print(f"  Point Accuracy        : {acc_pt:.4f}")
    print(f"  Point Balanced Accuracy: {bacc_pt:.4f}")
    print(f"  Point F1              : {f1_pt:.4f}")
    print(f"  Point ROC-AUC         : {auc_pt:.4f}")

    class_0_idx = np.where(y_true == 0)[0]
    class_1_idx = np.where(y_true == 1)[0]
    n0 = len(class_0_idx)
    n1 = len(class_1_idx)

    acc_boot = np.empty(N_BOOTSTRAPS)
    bacc_boot = np.empty(N_BOOTSTRAPS)
    f1_boot = np.empty(N_BOOTSTRAPS)
    auc_boot = np.empty(N_BOOTSTRAPS)

    for i in range(N_BOOTSTRAPS):
        s0 = rng.choice(class_0_idx, size=n0, replace=True)
        s1 = rng.choice(class_1_idx, size=n1, replace=True)
        idx = np.concatenate([s0, s1])

        yt_b = y_true[idx]
        yp_b = y_pred[idx]
        ypr_b = y_prob[idx]

        acc_boot[i] = accuracy_score(yt_b, yp_b)
        bacc_boot[i] = balanced_accuracy_score(yt_b, yp_b)
        f1_boot[i] = f1_score(yt_b, yp_b, zero_division=0)
        auc_boot[i] = roc_auc_score(yt_b, ypr_b)

    metrics = [
        ("Accuracy", acc_pt, acc_boot),
        ("Balanced_Accuracy", bacc_pt, bacc_boot),
        ("F1", f1_pt, f1_boot),
        ("ROC-AUC", auc_pt, auc_boot),
    ]

    for m_label, pt, b_arr in metrics:
        lo, hi = percentile_ci(b_arr)
        print(f"  {m_label:18s}: {pt:.4f} [95% CI: {lo:.4f}, {hi:.4f}]")
        all_ci_rows.append({
            "model": model_name,
            "metric": m_label,
            "point_estimate": pt,
            "ci_lower_95": lo,
            "ci_upper_95": hi,
            "n_bootstrap": N_BOOTSTRAPS
        })

ci_df = pd.DataFrame(all_ci_rows)

# Save combined top3 table
top3_ci_path = os.path.join(OUT_DIR, "top3_bootstrap_ci.csv")
ci_df.to_csv(top3_ci_path, index=False)

# Save RF table for backward compatibility
rf_df = ci_df[ci_df["model"] == "Random_Forest"]
if len(rf_df) > 0:
    rf_ci_path = os.path.join(OUT_DIR, "random_forest_bootstrap_ci.csv")
    rf_df.to_csv(rf_ci_path, index=False)

print()
print("=" * 70)
print("STEP 15 COMPLETE — BOOTSTRAP CONFIDENCE INTERVALS SAVED")
print("=" * 70)
print(f"Saved: {top3_ci_path}")
