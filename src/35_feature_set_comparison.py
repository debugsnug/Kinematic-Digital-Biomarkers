# ============================================================
# STEP 35 - FEATURE-SET COMPARISON (primary vs secondary vs position-error ablation)
# ============================================================
# Same data, same subjects, same outer/inner folds (asserted), same 16 models / grids / selection rule; only the
# predictor set differs:
#   coordinate_only               PRIMARY    key-point coordinates + target object   (results/...)
#   angle_enhanced                SECONDARY  + 2D-video-derived joint-angle features (results/feature_sets/angle_enhanced)
#   coordinate_no_position_error  ABLATION   coordinate_only without position_error_* (results/feature_sets/coordinate_no_position_error)
# Paired comparison of the complete nested selection procedures at subject level (each subject predicted once per set).
# These ablations do NOT remove site/acquisition confounding; they only show how much the result depends on
# angle-derived and target-object-derived inputs.

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import binomtest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(BASE, "results")
sys.path.insert(0, os.path.join(BASE, "src"))
SETS = {"coordinate_only": "", "angle_enhanced": "feature_sets/angle_enhanced",
        "coordinate_no_position_error": "feature_sets/coordinate_no_position_error",
        "coordinate_no_duration": "feature_sets/coordinate_no_duration",
        "coordinate_no_duration_no_position_error": "feature_sets/coordinate_no_duration_no_position_error",
        "coordinate_no_timing_coupled": "feature_sets/coordinate_no_timing_coupled"}
OUT = os.path.join(R, "feature_sets"); os.makedirs(OUT, exist_ok=True)

sizes = pd.read_csv(os.path.join(R, "feature_audit", "feature_set_sizes.csv")).set_index("feature_set")["n_predictors"]
data = {}
for name, sub in SETS.items():
    d = os.path.join(R, sub)
    data[name] = {
        "assign": pd.read_csv(os.path.join(d, "model_benchmark", "outer_fold_assignment.csv")),
        "bench": pd.read_csv(os.path.join(d, "model_benchmark", "model_benchmark_summary.csv")),
        "ci": pd.read_csv(os.path.join(d, "nested_selection", "nested_selection_ci.csv")).set_index("metric"),
        "folds": pd.read_csv(os.path.join(d, "nested_selection", "nested_selection_folds.csv")),
        "pred": pd.read_csv(os.path.join(d, "nested_selection", "nested_selection_predictions.csv")),
    }
base_assign = data["coordinate_only"]["assign"]
for name, v in data.items():                       # identical outer folds for all sets
    assert v["assign"].equals(base_assign), f"outer folds differ for {name}"
print("Outer-fold assignment identical across the three feature sets: PASS")

# ---- table 1: headline per set ----
rows = []
for name, v in data.items():
    ci = v["ci"]
    rows.append({"feature_set": name, "n_predictors": int(sizes[name]),
                 "nested_accuracy": ci.loc["Accuracy", "point_estimate"], "acc_ci_low": ci.loc["Accuracy", "ci_lower_95"], "acc_ci_high": ci.loc["Accuracy", "ci_upper_95"],
                 "nested_balanced_accuracy": ci.loc["Balanced_Accuracy", "point_estimate"],
                 "bacc_ci_low": ci.loc["Balanced_Accuracy", "ci_lower_95"], "bacc_ci_high": ci.loc["Balanced_Accuracy", "ci_upper_95"],
                 "nested_f1": ci.loc["F1", "point_estimate"], "nested_mcc": ci.loc["MCC", "point_estimate"],
                 "nested_sensitivity": ci.loc["Sensitivity", "point_estimate"], "nested_specificity": ci.loc["Specificity", "point_estimate"],
                 "mean_fold_roc_auc": v["folds"]["roc_auc"].mean(),
                 "top3_by_inner_cv": ", ".join(v["bench"].head(3)["model"]),
                 "best_single_model_outer_accuracy": v["bench"]["test_accuracy_mean"].max()})
H = pd.DataFrame(rows); H.to_csv(os.path.join(OUT, "feature_set_headline_comparison.csv"), index=False)

# ---- table 2: all 16 models x 3 sets (outer-fold mean accuracy, inner acc) ----
B = None
for name, v in data.items():
    t = v["bench"][["model", "inner_accuracy_mean", "test_accuracy_mean", "test_balanced_accuracy_mean", "test_roc_auc_mean"]].copy()
    t.columns = ["model"] + [f"{name}__{c}" for c in t.columns[1:]]
    B = t if B is None else B.merge(t, on="model")
B.to_csv(os.path.join(OUT, "benchmark_16_models_by_feature_set.csv"), index=False)

# ---- table 3: paired subject-level comparison of the nested procedures ----
def correct(name):
    p = data[name]["pred"].copy(); p["subject"] = p.subject.astype(int)
    return p.set_index("subject").sort_index().eval("true_target == predicted_target")
base = correct("coordinate_only"); rng = np.random.default_rng(42); prow = []
for other in [k for k in SETS if k != "coordinate_only"]:
    o = correct(other); assert (o.index == base.index).all()
    b01 = int(((~base) & o).sum()); b10 = int((base & (~o)).sum())       # subjects only 'other' / only primary gets right
    p = binomtest(b01, b01 + b10, 0.5).pvalue if b01 + b10 else 1.0
    diffs = [o.to_numpy()[i].mean() - base.to_numpy()[i].mean() for i in (rng.integers(0, len(base), len(base)) for _ in range(10000))]
    prow.append({"comparison": f"{other} minus coordinate_only (nested accuracy)", "accuracy_difference": o.mean() - base.mean(),
                 "bootstrap_ci95_low": np.percentile(diffs, 2.5), "bootstrap_ci95_high": np.percentile(diffs, 97.5),
                 "subjects_only_other_correct": b01, "subjects_only_primary_correct": b10, "exact_mcnemar_p": p})
Pc = pd.DataFrame(prow); Pc.to_csv(os.path.join(OUT, "paired_subject_level_comparison.csv"), index=False)

# ---- repeated-partition comparison (if the repeated runs exist) ----
rep = []
for name, sub in SETS.items():
    f = os.path.join(R, sub, "repeated_selection", "repeated_selection_summary.csv")
    if os.path.exists(f):
        s = pd.read_csv(f).set_index("metric")
        rep.append({"feature_set": name, "accuracy_mean_over_10_partitions": s.loc["accuracy", "mean"],
                    "accuracy_sd": s.loc["accuracy", "sd_across_partitions"], "accuracy_min": s.loc["accuracy", "min"], "accuracy_max": s.loc["accuracy", "max"],
                    "balanced_accuracy_mean": s.loc["balanced_accuracy", "mean"], "mcc_mean": s.loc["mcc", "mean"]})
if rep:
    pd.DataFrame(rep).to_csv(os.path.join(OUT, "feature_set_repeated_partition_comparison.csv"), index=False)

pd.set_option("display.width", 250)
print("\nHEADLINE (fully nested selection procedure, 51 subjects, identical folds):"); print(H.round(3).to_string(index=False))
print("\nPAIRED SUBJECT-LEVEL COMPARISON:"); print(Pc.round(3).to_string(index=False))
if rep:
    print("\nREPEATED PARTITIONS (10 seeds):"); print(pd.DataFrame(rep).round(3).to_string(index=False))
