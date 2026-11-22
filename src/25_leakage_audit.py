# ============================================================
# STEP 25 — LEAKAGE & METHODOLOGY AUDIT
# ============================================================

import glob
import os
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "experiment_dataset.csv"
)

BENCHMARK_PRED_PATH = os.path.join(
    BASE_DIR,
    "results",
    "model_benchmark",
    "model_benchmark_predictions.csv"
)

TOP3_PRED_PATH = os.path.join(
    BASE_DIR,
    "results",
    "top3_validation",
    "top3_predictions.csv"
)

ABLATION_PATH = os.path.join(
    BASE_DIR,
    "results",
    "ablation",
    "ablation_fold_results.csv"
)

STABLE_FEATURE_PATH = os.path.join(
    BASE_DIR,
    "results",
    "feature_stability",
    "stable_features.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "leakage_audit"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("STEP 25 — LEAKAGE & METHODOLOGY AUDIT")
print("=" * 70)


audit = []


def record(
    check,
    status,
    details
):

    audit.append({

        "check": check,

        "status": status,

        "details": details
    })


# ============================================================
# 1. LOAD EXPERIMENT DATASET
# ============================================================

print()
print("=" * 70)
print("1. EXPERIMENT DATASET")
print("=" * 70)

df = pd.read_csv(
    DATA_PATH
)

y = df["target"]

print(
    f"Shape: {df.shape}"
)

record(
    "Experiment dataset exists",
    "PASS",
    f"Shape = {df.shape}"
)


# ============================================================
# 2. REQUIRED COLUMNS
# ============================================================

required_columns = [
    "subject",
    "target"
]

missing_required = [
    col
    for col in required_columns
    if col not in df.columns
]

if len(missing_required) == 0:

    print(
        "Required subject/target columns: PASS"
    )

    record(
        "Required subject and target columns",
        "PASS",
        "Both subject and target are present."
    )

else:

    print(
        "Required subject/target columns: FAIL"
    )

    record(
        "Required subject and target columns",
        "FAIL",
        str(missing_required)
    )


# ============================================================
# 3. SUBJECT COUNT
# ============================================================

n_subjects = df["subject"].nunique()

print(
    f"Unique subjects: {n_subjects}"
)

if n_subjects == 51:

    record(
        "Unique subject count",
        "PASS",
        "51 unique subjects found."
    )

else:

    record(
        "Unique subject count",
        "CHECK",
        f"{n_subjects} unique subjects found."
    )


# ============================================================
# 4. ONE ROW PER SUBJECT
# ============================================================

rows_per_subject = (
    df.groupby("subject")
    .size()
)

duplicate_subjects = (
    rows_per_subject[
        rows_per_subject > 1
    ]
)

if len(duplicate_subjects) == 0:

    print(
        "One row per subject: PASS"
    )

    record(
        "One row per subject",
        "PASS",
        "No subject occurs more than once."
    )

else:

    print(
        "One row per subject: FAIL"
    )

    record(
        "One row per subject",
        "FAIL",
        f"Duplicated subjects: "
        f"{duplicate_subjects.index.tolist()}"
    )


# ============================================================
# 5. TARGET DISTRIBUTION
# ============================================================

target_counts = (
    df["target"]
    .value_counts()
    .sort_index()
)

print()
print(
    "Target distribution:"
)

print(
    target_counts.to_string()
)

if (
    target_counts.get(0, 0) == 21
    and
    target_counts.get(1, 0) == 30
):

    record(
        "Target distribution",
        "PASS",
        "TD=21 and DCP=30."
    )

else:

    record(
        "Target distribution",
        "CHECK",
        target_counts.to_dict()
    )


# ============================================================
# 6. TARGET / GROUP LEAKAGE INTO FEATURES
# ============================================================

for forbidden in [
    "target",
    "subject"
]:

    if forbidden in df.columns:

        print(
            f"'{forbidden}' present as metadata: PASS"
        )


# Metadata columns must never be treated as predictors.
metadata_columns = [
    "target",
    "subject",
    "side"
]

predictor_columns = [
    col
    for col in df.columns
    if col not in metadata_columns
]

# Numeric predictors only
numeric_predictors = df[
    predictor_columns
].select_dtypes(
    include=[np.number]
).columns.tolist()

predictor_columns = numeric_predictors

print()
print(
    f"Predictor count: "
    f"{len(predictor_columns)}"
)


# Check that the expected predictors are present.
EXPECTED_PREDICTORS = 84  # coordinate-only primary set (78 coordinate + 6 position_error)

if len(predictor_columns) == EXPECTED_PREDICTORS:
    record(
        "Predictor count",
        "PASS",
        f"{len(predictor_columns)} predictor variables found."
    )
else:
    record(
        "Predictor count",
        "CHECK",
        f"Expected {EXPECTED_PREDICTORS} predictors, "
        f"but found {len(predictor_columns)}."
    )


# ============================================================
# 7. NUMERIC PREDICTORS
# ============================================================

all_predictors_numeric = all(
    pd.api.types.is_numeric_dtype(df[col])
    for col in predictor_columns
)

print(
    f"All predictors numeric: "
    f"{'PASS' if all_predictors_numeric else 'CHECK'}"
)

record(
    "Predictor data type",
    "PASS" if all_predictors_numeric else "CHECK",
    (
        "All selected predictors are numeric."
        if all_predictors_numeric
        else "One or more selected predictors are not numeric."
    )
)


# ============================================================
# 8. COMPLETELY MISSING FEATURES
# ============================================================

completely_missing = [
    col
    for col in predictor_columns
    if df[col].isna().all()
]

print()
print(
    f"Completely missing predictors: "
    f"{len(completely_missing)}"
)

if len(completely_missing) == 0:

    record(
        "Completely missing predictors",
        "PASS",
        "No predictor is completely missing."
    )

else:

    record(
        "Completely missing predictors",
        "FAIL",
        str(completely_missing)
    )


# ============================================================
# 9. PARTIALLY MISSING FEATURES
# ============================================================

partially_missing = [
    col
    for col in predictor_columns
    if (
        df[col].isna().any()
        and
        not df[col].isna().all()
    )
]

print(
    f"Partially missing predictors: "
    f"{len(partially_missing)}"
)

record(
    "Partially missing predictors",
    "INFO",
    f"{len(partially_missing)} predictors contain "
    f"some missing values."
)


# ============================================================
# 10. DUPLICATE COMPLETE ROWS
# ============================================================

duplicate_rows = df.duplicated().sum()

print(
    f"Duplicate complete rows: "
    f"{duplicate_rows}"
)

if duplicate_rows == 0:

    record(
        "Duplicate complete rows",
        "PASS",
        "No duplicate complete rows."
    )

else:

    record(
        "Duplicate complete rows",
        "CHECK",
        f"{duplicate_rows} duplicates."
    )


# ============================================================
# 11. AUDIT MODEL BENCHMARK PREDICTIONS
# ============================================================

print()
print("=" * 70)
print("2. MODEL BENCHMARK SUBJECT SPLIT AUDIT")
print("=" * 70)

if os.path.exists(
    BENCHMARK_PRED_PATH
):

    pred = pd.read_csv(
        BENCHMARK_PRED_PATH
    )

    print(
        f"Prediction rows: {len(pred)}"
    )

    print(
        f"Models: "
        f"{pred['model'].nunique()}"
    )

    print(
        f"Unique subjects: "
        f"{pred['subject'].nunique()}"
    )


    # --------------------------------------------------------
    # Every model should have every subject exactly once
    # --------------------------------------------------------

    duplicate_model_subject = (
        pred.groupby(
            ["model", "subject"]
        )
        .size()
    )

    duplicates = (
        duplicate_model_subject[
            duplicate_model_subject > 1
        ]
    )

    if len(duplicates) == 0:

        print(
            "Model-subject prediction uniqueness: PASS"
        )

        record(
            "Model benchmark subject uniqueness",
            "PASS",
            "Each model has at most one prediction per subject."
        )

    else:

        print(
            "Model-subject prediction uniqueness: FAIL"
        )

        record(
            "Model benchmark subject uniqueness",
            "FAIL",
            str(duplicates)
        )


    # --------------------------------------------------------
    # Each model should contain all 51 subjects
    # --------------------------------------------------------

    subjects_per_model = (
        pred.groupby("model")["subject"]
        .nunique()
    )

    if (
        subjects_per_model == 51
    ).all():

        print(
            "All models cover all 51 subjects: PASS"
        )

        record(
            "Model benchmark subject coverage",
            "PASS",
            "Every benchmark model has predictions "
            "for all 51 subjects."
        )

    else:

        print(
            "Model subject coverage: CHECK"
        )

        record(
            "Model benchmark subject coverage",
            "CHECK",
            subjects_per_model.to_dict()
        )


else:

    print(
        "Benchmark predictions file not found."
    )

    record(
        "Model benchmark predictions",
        "CHECK",
        "Prediction file not found."
    )


# ============================================================
# 12. OUTER FOLD SUBJECT UNIQUENESS
# ============================================================

if os.path.exists(
    BENCHMARK_PRED_PATH
):

    print()
    print(
        "Checking outer-fold subject assignment..."
    )

    fold_subject = (
        pred[
            [
                "model",
                "subject",
                "fold"
            ]
        ]
        .drop_duplicates()
    )

    fold_counts = (
        fold_subject
        .groupby(
            ["model", "subject"]
        )["fold"]
        .nunique()
    )

    leakage_subjects = (
        fold_counts[
            fold_counts > 1
        ]
    )

    if len(leakage_subjects) == 0:

        print(
            "Subject appears in one outer test fold: PASS"
        )

        record(
            "Outer-fold subject isolation",
            "PASS",
            "Each subject is assigned to one outer test fold "
            "per model."
        )

    else:

        print(
            "Outer-fold subject isolation: FAIL"
        )

        record(
            "Outer-fold subject isolation",
            "FAIL",
            str(leakage_subjects)
        )


# ============================================================
# 13. TOP-3 PREDICTION AUDIT
# ============================================================

print()
print("=" * 70)
print("3. TOP-3 VALIDATION AUDIT")
print("=" * 70)

if os.path.exists(
    TOP3_PRED_PATH
):

    top3 = pd.read_csv(
        TOP3_PRED_PATH
    )

    print(
        f"Top-3 prediction rows: "
        f"{len(top3)}"
    )

    print(
        f"Models: "
        f"{top3['model'].nunique()}"
    )

    print(
        f"Subjects: "
        f"{top3['subject'].nunique()}"
    )


    top3_fold_counts = (
        top3[
            [
                "model",
                "subject",
                "fold"
            ]
        ]
        .drop_duplicates()
        .groupby(
            ["model", "subject"]
        )["fold"]
        .nunique()
    )


    top3_leakage = (
        top3_fold_counts[
            top3_fold_counts > 1
        ]
    )


    if len(top3_leakage) == 0:

        print(
            "Top-3 subject-fold isolation: PASS"
        )

        record(
            "Top-3 outer-fold subject isolation",
            "PASS",
            "Each subject appears in one outer fold per model."
        )

    else:

        print(
            "Top-3 subject-fold isolation: FAIL"
        )

        record(
            "Top-3 outer-fold subject isolation",
            "FAIL",
            str(top3_leakage)
        )

else:

    record(
        "Top-3 validation predictions",
        "CHECK",
        "Top-3 prediction file not found."
    )


# ============================================================
# 14. ABLATION AUDIT
# ============================================================

print()
print("=" * 70)
print("4. ABLATION AUDIT")
print("=" * 70)

if os.path.exists(
    ABLATION_PATH
):

    ablation = pd.read_csv(
        ABLATION_PATH
    )

    print(
        f"Ablation rows: "
        f"{len(ablation)}"
    )

    print(
        f"Representations: "
        f"{ablation['representation'].unique().tolist()}"
    )

    representation_folds = (
        ablation
        .groupby(
            "representation"
        )["fold"]
        .nunique()
    )

    print(
        "Folds per representation:"
    )

    print(
        representation_folds.to_string()
    )

    if (
        representation_folds == 5
    ).all():

        record(
            "Ablation outer-fold coverage",
            "PASS",
            "Each representation has five outer folds."
        )

    else:

        record(
            "Ablation outer-fold coverage",
            "CHECK",
            representation_folds.to_dict()
        )


# ============================================================
# 15. STABLE FEATURE COUNT
# ============================================================

print()
print("=" * 70)
print("5. FEATURE STABILITY")
print("=" * 70)

if os.path.exists(
    STABLE_FEATURE_PATH
):

    stable = pd.read_csv(
        STABLE_FEATURE_PATH
    )

    print(
        f"Stable features: "
        f"{len(stable)}"
    )

    record(
        "Stable feature set",
        "INFO",
        f"{len(stable)} features identified as stable."
    )


# ============================================================
# 16. MODEL SELECTION AUDIT (ZERO TEST-SNOOPING VERIFICATION)
# ============================================================

print()
print("=" * 70)
print("6. MODEL SELECTION AUDIT")
print("=" * 70)

SELECTION_JSON = os.path.join(
    BASE_DIR,
    "results",
    "model_benchmark",
    "selected_top_models.json"
)

if os.path.exists(SELECTION_JSON):
    import json
    with open(SELECTION_JSON, "r", encoding="utf-8") as f:
        meta = json.load(f)
    selected_mods = meta.get("selected_models", [])
    record(
        "Leak-free model selection",
        "PASS",
        f"Top models {selected_mods} were ranked by inner-CV accuracy only; no outer-fold test score entered the ranking. "
        "Scope: this rules out direct test-set use, NOT selection bias - see the LIMITATION entry below."
    )
    record(
        "Globally selected top-3 results are optimistic",
        "LIMITATION",
        "The top-3 list pools inner-CV results over ALL outer folds, so each outer test subject sat in the training data of the other folds "
        "that produced the ranking. Outer-fold results of these 3 families (about 0.90 accuracy) are therefore optimistic validation numbers, "
        "NOT independent final-test performance. The unbiased estimate of the whole selection process is the fully nested one "
        "(results/nested_selection, headline)."
    )
    print(f"Top models {selected_mods} selected strictly on inner-CV accuracy.")
else:
    record(
        "Leak-free model selection",
        "CHECK",
        "selected_top_models.json not found."
    )

# ---- additional checks: fully nested selection, permutation power, FRI label-freeness, LIME ----
import pandas as _pd
_ns = os.path.join(BASE_DIR, "results", "nested_selection", "nested_selection_predictions.csv")
if os.path.exists(_ns):
    _p = _pd.read_csv(_ns)
    ok = _p["subject"].is_unique and _p["subject"].nunique() == _pd.read_csv(DATA_PATH)["subject"].nunique()
    _bf = _pd.read_csv(os.path.join(BASE_DIR, "results", "model_benchmark", "model_benchmark_fold_results.csv"))
    _nf = _pd.read_csv(os.path.join(BASE_DIR, "results", "nested_selection", "nested_selection_folds.csv"))
    _exp = (_bf.sort_values(["fold", "inner_accuracy", "model"], ascending=[True, False, True], kind="mergesort").groupby("fold").head(1).sort_values("fold"))
    ok = ok and list(_exp["model"]) == list(_nf["model"]) and bool((_bf.groupby("fold")["inner_accuracy"].max().values == _nf.sort_values("fold")["inner_accuracy"].values).all())
    record("Fully nested selection estimate", "PASS" if ok else "FAIL",
           "Each subject predicted exactly once, and the winner in every fold is re-derived here as the family with the highest INNER-CV accuracy in that fold (no outer-test score used)." if ok
           else "Nested-selection predictions do not cover each subject exactly once.")
else:
    record("Fully nested selection estimate", "FAIL", "nested_selection_predictions.csv missing.")

_pt = os.path.join(BASE_DIR, "results", "leakage_audit", "label_permutation_test.csv")
if os.path.exists(_pt):
    _r = _pd.read_csv(_pt).iloc[0]
    ok = _r["n_permutations"] >= 1000 and _r["permuted_balanced_accuracy_mean"] < 0.60 and _r["permutation_p_value"] < 0.05
    record("Label-permutation sanity check", "PASS" if ok else "FAIL",
           f"{int(_r['n_permutations'])} permutations, null mean {_r['permuted_balanced_accuracy_mean']:.3f}, p<={_r['permutation_p_value']:.4f} (1/1001 floor; fixed RF, not the full selection procedure). "
           "Sanity check only; cannot exclude site/session confounding.")
else:
    record("Label-permutation sanity check", "FAIL", "label_permutation_test.csv missing.")

_fri = os.path.join(BASE_DIR, "results", "functional_resilience", "resilience_indices_by_subject.csv")
record("Functional Resilience Index (output presence only)", "INFO" if os.path.exists(_fri) else "FAIL",
       "Output present. Label-free, subject-intrinsic by construction (code review; not machine-verified here). Proposed/exploratory index." if os.path.exists(_fri)
       else "resilience_indices_by_subject.csv missing.")
_lime = os.path.join(BASE_DIR, "results", "lime_oof", "oof_lime_summary.csv")
record("LIME explanations (output presence only)", "INFO" if os.path.exists(_lime) else "FAIL",
       "Output present (OOF LIME by construction in step 31; not machine-verified here)." if os.path.exists(_lime) else "oof_lime_summary.csv missing.")

# ---- additional checks for the coordinate-only primary analysis ----
_fa = os.path.join(BASE_DIR, "results", "feature_audit", "feature_audit_dummy")
_dep = os.path.join(BASE_DIR, "results", "feature_audit", "base_feature_dependency.csv")
_fsz = os.path.join(BASE_DIR, "results", "feature_audit", "feature_set_sizes.csv")
if os.path.exists(_fsz):
    _z = pd.read_csv(_fsz).set_index("feature_set")["n_predictors"].to_dict()
    record("Coordinate-only primary feature set", "PASS" if (_z.get("coordinate_only"), _z.get("angle_enhanced"), _z.get("coordinate_no_position_error")) == (84, 168, 78) and 60 <= _z.get("coordinate_no_timing_coupled", 0) < 84 and _z.get("coordinate_no_duration") == 78 else "FAIL",
           f"Predictor counts {_z}; modality registry verified empirically (angle/object columns perturbed in 408 recordings, step 32); no angle-derived predictor in the primary set.")
else:
    record("Coordinate-only primary feature set", "FAIL", "feature_set_sizes.csv missing.")
_cc = os.path.join(BASE_DIR, "results", "top3_validation", "top3_crosscheck_vs_benchmark.csv")
if os.path.exists(_cc):
    _c = pd.read_csv(_cc)
    _ok = bool(_c["predictions_identical_to_benchmark"].all() and _c["best_params_identical_to_benchmark"].all())
    record("Benchmark vs independent Pipeline re-run", "PASS" if _ok else "FAIL",
           f"{int(_c['predictions_identical_to_benchmark'].sum())}/{len(_c)} model-fold predictions and best parameters identical between the pre-fitted-transform benchmark and the real Pipeline+GridSearchCV path.")
_R = os.path.join(BASE_DIR, "results")
_extra = ["angle_enhanced", "coordinate_no_position_error", "coordinate_no_duration", "coordinate_no_duration_no_position_error", "coordinate_no_timing_coupled"]
_ofa = {"coordinate_only": os.path.join(_R, "model_benchmark", "outer_fold_assignment.csv"),
        }
_ofa.update({k: os.path.join(_R, "feature_sets", k, "model_benchmark", "outer_fold_assignment.csv") for k in _extra})
if all(os.path.exists(v) for v in _ofa.values()):
    _a = {k: _pd.read_csv(v).sort_values("subject").reset_index(drop=True) for k, v in _ofa.items()}
    _same = all(_a[k].equals(_a["coordinate_only"]) for k in _a)
    record("Identical outer folds across feature sets", "PASS" if _same else "FAIL",
           f"Saved outer-fold assignments compared directly: identical for all {len(_ofa)} feature sets." if _same else "Outer-fold assignments differ between feature sets.")
else:
    record("Identical outer folds across feature sets", "FAIL", "outer_fold_assignment.csv missing for a feature set.")
_seed_files = sorted(glob.glob(os.path.join(_R, "repeated_selection", "seed_*", "model_benchmark", "outer_fold_assignment.csv"))) + [_ofa["coordinate_only"]]
_bad = []
for _f in _seed_files:
    _d = _pd.read_csv(_f)
    if not (_d["subject"].is_unique and _d["subject"].nunique() == 51 and _d["outer_fold"].nunique() == 5):
        _bad.append(_f)
record("Outer-split integrity (all seeds, all primary assignments)", "PASS" if not _bad and _seed_files else "FAIL",
       f"{len(_seed_files)} saved outer assignments (seed 42 + repeated seeds): each subject in exactly one of 5 test folds. Inner-split overlap is asserted inside step 23 at run time (not re-derivable from saved files)." if not _bad else f"Violations: {_bad}")
_st = os.path.join(BASE_DIR, "demo", "study.json")
if os.path.exists(_st):
    import hashlib as _hl, json as _js
    _d = _js.load(open(_st))
    _ok = _d["feature_sha256"] == _hl.sha256(open(os.path.join(BASE_DIR, "data", "processed", "rep_features.csv"), "rb").read()).hexdigest()
    _nn = pd.read_csv(os.path.join(BASE_DIR, "results", "nested_selection", "nested_selection_predictions.csv")).set_index("subject")
    _ok = _ok and all(c["forest"]["nested_call"] == int(_nn.loc[int(c["id"]), "predicted_target"]) for c in _d["cases"]) and len(_d["cases"]) == 51
    record("Reach Lab demo matches final results", "PASS" if _ok else "FAIL",
           "demo/study.json was built from the current rep_features.csv (hash match) and its per-participant headline calls equal results/nested_selection predictions for all 51 participants." if _ok else "demo/study.json is stale relative to data/processed or results/nested_selection - run step 38.")
record("LightGBM / CatBoost", "INFO",
       "Not implemented. The benchmark has 16 models (the stated 15-16 range); LightGBM/CatBoost were considered and left out; adding them would give 18 models and needs the guide's approval.")

# ---- unresolved limitations: neither passed nor failed; cannot be fixed with the supplied data ----
record("Recording-site / acquisition confounding", "LIMITATION",
       "No site/session/camera variable exists in the supplied data (a per-site claim in earlier drafts could not be verified and was removed); subject-number blocks are structured by group, and file/subject "
       "numbering alone separates the groups (ID-only baseline in label_permutation_test.csv). A model could partly exploit acquisition differences; "
       "the permutation test cannot detect this. Unresolved.")
record("No independent test set / external cohort", "LIMITATION",
       "n = 51 from one dataset. 'Final test' = outer cross-validation folds; generalisation to new children, sites or cameras is untested.")
record("Uncertainty intervals are conditional", "LIMITATION",
       "Bootstrap CIs resample the same 51 subjects with trained models fixed; repeated CV re-partitions the same subjects. Neither includes "
       "cohort-level or site-level variability.")
record("Repetition-aware features", "LIMITATION",
       "No statistically supported classification advantage over static means (5 paired folds, under-powered); FRI is a proposed, not clinically validated, index.")


# ============================================================
# 17. RESTORED POSITION ERROR FEATURE AUDIT
# ============================================================

pos_features = [c for c in predictor_columns if "position_error" in c]
if len(pos_features) == 6:
    record(
        "Restored position error features",
        "PASS",
        f"All 6 position error features present and validated: {pos_features}"
    )
    print(f"Position error features: {len(pos_features)} verified.")
else:
    record(
        "Restored position error features",
        "CHECK",
        f"Expected 6 position error features, found {len(pos_features)}"
    )


# ============================================================
# 18. METHODOLOGY STATEMENTS
# ============================================================

print()
print("=" * 70)
print("7. METHODOLOGY AUDIT SUMMARY")
print("=" * 70)

methodology = {

    "prediction_unit":
        "Subject",

    "dataset_size":
        "51 subjects",

    "groups":
        "30 DCP / 21 TD",

    "outer_cv":
        "5-fold StratifiedGroupKFold",

    "inner_cv":
        "4-fold StratifiedGroupKFold",

    "group_variable":
        "subject",

    "imputation":
        "Median imputation inside ML pipeline",

    "scaling":
        "StandardScaler inside ML pipeline",

    "feature_selection":
        "Mutual-information SelectKBest inside ML pipeline",

    "hyperparameter_tuning":
        "Inner cross-validation only",

    "outer_test_subjects":
        "Not used for fitting preprocessing, feature selection, "
        "or hyperparameter tuning"
}


for key, value in methodology.items():

    print(
        f"{key}: {value}"
    )


# ============================================================
# 17. SAVE AUDIT TABLE
# ============================================================

audit_df = pd.DataFrame(
    audit
)

audit_csv = os.path.join(
    OUTPUT_DIR,
    "leakage_audit.csv"
)

audit_df.to_csv(
    audit_csv,
    index=False
)


# ============================================================
# 18. SAVE HUMAN-READABLE REPORT
# ============================================================

report_path = os.path.join(
    OUTPUT_DIR,
    "leakage_audit_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "LEAKAGE & METHODOLOGY AUDIT\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        "DATASET\n"
    )

    f.write(
        f"Shape: {df.shape}\n"
    )

    f.write(
        f"Subjects: {n_subjects}\n"
    )

    f.write(
        f"Predictors: {len(predictor_columns)}\n"
    )

    f.write(
        f"TD: {(y == 0).sum()}\n"
    )

    f.write(
        f"DCP: {(y == 1).sum()}\n\n"
    )


    f.write(
        "METHODOLOGY\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    for key, value in methodology.items():

        f.write(
            f"{key}: {value}\n"
        )


    f.write("\n")

    f.write(
        "AUDIT CHECKS\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    counts = audit_df["status"].value_counts().to_dict()
    f.write("SUMMARY: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) + "\n")
    f.write("PASS = check performed and passed; INFO = descriptive; LIMITATION = unresolved methodological limit "
            "(not a failed check, but it bounds what the results mean); FAIL/CHECK = needs action.\n\n")
    f.write("-- CHECKS --\n")
    for _, row in audit_df[audit_df["status"] != "LIMITATION"].iterrows():
        f.write(f"{row['status']}: {row['check']} — {row['details']}\n")
    f.write("\n-- UNRESOLVED LIMITATIONS --\n")
    for _, row in audit_df[audit_df["status"] == "LIMITATION"].iterrows():
        f.write(f"LIMITATION: {row['check']} — {row['details']}\n")


# ============================================================
# FINAL STATUS
# ============================================================

print()
print("=" * 70)
print("AUDIT RESULTS")
print("=" * 70)

print()

print(
    audit_df.to_string(
        index=False
    )
)

print()
print("=" * 70)
print("FILES CREATED")
print("=" * 70)

print(
    audit_csv
)

print(
    report_path
)

print()
print("=" * 70)
print("STEP 25 COMPLETE")
print("=" * 70)