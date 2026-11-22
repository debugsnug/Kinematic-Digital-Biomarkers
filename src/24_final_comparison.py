# ============================================================
# STEP 24 – FINAL RESULTS COMPARISON & CONSOLIDATION
# ============================================================

import os
import json
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

ABLATION_DIR = os.path.join(BASE_DIR, "results", "ablation")
BENCHMARK_DIR = os.path.join(BASE_DIR, "results", "model_benchmark")
TOP3_DIR = os.path.join(BASE_DIR, "results", "top3_validation")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "final_comparison")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("STEP 24 – FINAL RESULTS CONSOLIDATION & COMPARISON")
print("=" * 70)


# ============================================================
# 1. LOAD BENCHMARK RESULTS (16 MODELS)
# ============================================================

benchmark_path = os.path.join(BENCHMARK_DIR, "model_benchmark_summary.csv")
benchmark = pd.read_csv(benchmark_path)

print()
print("1. COMPLETE 16-MODEL BENCHMARK (SORTED BY INNER VALIDATION ACCURACY)")
print("-" * 70)
print(
    benchmark[
        [
            "model",
            "inner_accuracy_mean",
            "test_accuracy_mean",
            "test_accuracy_std",
            "test_balanced_accuracy_mean",
            "test_f1_mean",
            "test_roc_auc_mean"
        ]
    ].to_string(index=False)
)


# ============================================================
# 2. LOAD TOP-3 SELECTION RECORD
# ============================================================

selection_path = os.path.join(BENCHMARK_DIR, "selected_top_models.json")
with open(selection_path, "r", encoding="utf-8") as f:
    selection_meta = json.load(f)

selected_models = selection_meta.get("selected_models", [])
print()
print("2. TOP-MODEL SELECTION")
print("-" * 70)
print(f"Selection Criterion: {selection_meta.get('selection_criterion')}")
print(f"Selected Top Models: {selected_models}")


# ============================================================
# 3. LOAD TOP-3 VALIDATION RESULTS
# ============================================================

top3_path = os.path.join(TOP3_DIR, "top3_summary.csv")
top3 = pd.read_csv(top3_path)

print()
print("3. TOP-3 MODEL DETAILED VALIDATION PERFORMANCE")
print("-" * 70)
print(top3.to_string(index=False))


# ============================================================
# 4. LOAD ABLATION RESULTS
# ============================================================

ablation_path = os.path.join(ABLATION_DIR, "ablation_summary.csv")
ablation = pd.read_csv(ablation_path)

print()
print("4. FEATURE REPRESENTATION ABLATION RESULTS")
print("-" * 70)
print(ablation.to_string(index=False))


# ============================================================
# 5. MODEL RANKING TABLE
# ============================================================

ranking = benchmark.copy()

ranking["inner_accuracy_rank"] = (
    ranking["inner_accuracy_mean"]
    .rank(ascending=False, method="min")
    .astype(int)
)

ranking["test_accuracy_rank"] = (
    ranking["test_accuracy_mean"]
    .rank(ascending=False, method="min")
    .astype(int)
)

ranking["test_balanced_acc_rank"] = (
    ranking["test_balanced_accuracy_mean"]
    .rank(ascending=False, method="min")
    .astype(int)
)

ranking["test_f1_rank"] = (
    ranking["test_f1_mean"]
    .rank(ascending=False, method="min")
    .astype(int)
)

ranking["test_roc_auc_rank"] = (
    ranking["test_roc_auc_mean"]
    .rank(ascending=False, method="min")
    .astype(int)
)

ranking = ranking.sort_values("inner_accuracy_mean", ascending=False)
ranking_path = os.path.join(OUTPUT_DIR, "model_benchmark_with_metric_ranks.csv")
ranking.to_csv(ranking_path, index=False)


# ============================================================
# 6. MASTER RESULTS TABLE
# ============================================================

master_path = os.path.join(OUTPUT_DIR, "final_results_master.csv")
ranking.to_csv(master_path, index=False)

rep_comp_path = os.path.join(OUTPUT_DIR, "representation_comparison.csv")
ablation.to_csv(rep_comp_path, index=False)


# ============================================================
# 7. GENERATE COMPREHENSIVE TEXT SUMMARY REPORT
# ============================================================

summary_lines = [
    "FINAL PROJECT RESULTS SUMMARY",
    "=" * 70,
    "",
    "1. COURSE & PROJECT CONTEXT",
    "Course: 23CSE365 Internet of Things – Amrita Vishwa Vidyapeetham",
    "Dataset: KU Leuven Markerless Motion Capture (DOI: 10.48804/BZM9SJ)",
    "Published Reference: Vanmechelen et al., Journal of Biomechanics, 2024",
    "",
    "2. TARGET VARIABLE & CLASS SPECIFICATION",
    "Prediction Task: Binary Classification of Upper-Limb Motor Impairment",
    "Target Variable: target (derived from clinical group)",
    "  - Class 1: Dyskinetic Cerebral Palsy (DCP), N = 30 (58.8%)",
    "  - Class 0: Typically Developing controls (TD), N = 21 (41.2%)",
    "Total Participants: 51 unique subjects (408 reaching-sideways trials, 8 reps/subject)",
    "Prediction Unit: 1 row per subject (no intra-subject data leakage)",
    "",
    "3. RIGOROUS SUBJECT-WISE DATA PARTITIONING & LEAKAGE PREVENTION",
    "Cross-Validation: 5-Fold Stratified Group Cross-Validation (StratifiedGroupKFold)",
    "Subject Isolation: Exactly 0 subjects shared across train and test partitions (one row per subject, grouped splits)",
    "Pipeline Encapsulation: Median Imputer, StandardScaler, and Mutual Information SelectKBest",
    "                      fit ONLY on training folds inside each cross-validation iteration.",
    "Temporal & Identifier Leakage: Subject IDs and metadata dropped; repetition trends computed strictly intra-subject.",
    "",
    "4. COMPLETE 16-MODEL BENCHMARK (Ranked by Inner-Validation Accuracy; Test* = outer-fold results; ranking itself pools inner CV over all folds):",
]

for rank, row in ranking.iterrows():
    summary_lines.append(
        f"  {row['inner_accuracy_rank']:02d}. {row['model']}: "
        f"InnerAcc = {row['inner_accuracy_mean']:.4f} +/- {row['inner_accuracy_std']:.4f} | "
        f"TestAcc = {row['test_accuracy_mean']:.4f} +/- {row['test_accuracy_std']:.4f} | "
        f"TestBAcc = {row['test_balanced_accuracy_mean']:.4f} +/- {row['test_balanced_accuracy_std']:.4f} | "
        f"TestF1 = {row['test_f1_mean']:.4f} +/- {row['test_f1_std']:.4f} | "
        f"TestAUC = {row['test_roc_auc_mean']:.4f} +/- {row['test_roc_auc_std']:.4f}"
    )

summary_lines.extend([
    "",
    "5. LEAK-FREE MODEL SELECTION & TOP-3 VALIDATION STAGE:",
    f"Selection Criterion: Highest inner-validation accuracy on training folds (no outer-test score used; ranking pools inner CV across folds, so outer-fold results are optimistic)",
    f"Selected Models: {', '.join(selected_models)}",
    "",
    "Out-of-Fold Validation Performance of Selected Models:",
])

for _, row in top3.iterrows():
    summary_lines.append(
        f"  * {row['model']}:"
        f" Accuracy = {row['accuracy_mean']:.4f} +/- {row['accuracy_std']:.4f},"
        f" Balanced Accuracy = {row['balanced_accuracy_mean']:.4f} +/- {row['balanced_accuracy_std']:.4f},"
        f" F1 = {row['f1_mean']:.4f} +/- {row['f1_std']:.4f},"
        f" Sensitivity = {row['recall_sensitivity_mean']:.4f} +/- {row['recall_sensitivity_std']:.4f},"
        f" Specificity = {row['specificity_mean']:.4f} +/- {row['specificity_std']:.4f},"
        f" ROC-AUC = {row['roc_auc_mean']:.4f} +/- {row['roc_auc_std']:.4f},"
        f" PR-AUC = {row['pr_auc_mean']:.4f} +/- {row['pr_auc_std']:.4f},"
        f" MCC = {row['mcc_mean']:.4f} +/- {row['mcc_std']:.4f},"
        f" Brier Score = {row['brier_score_mean']:.4f} +/- {row['brier_score_std']:.4f}"
    )

summary_lines.extend([
    "",
    "6. REPRESENTATION ABLATION STUDY:",
    "Compares Static Biomarkers vs. Repetition-Aware Descriptors vs. Stable Features:",
])

for _, row in ablation.iterrows():
    summary_lines.append(
        f"  - {row['representation']}:"
        f" Accuracy = {row['accuracy_mean']:.4f} +/- {row['accuracy_std']:.4f},"
        f" F1 = {row['f1_mean']:.4f} +/- {row['f1_std']:.4f},"
        f" ROC-AUC = {row['roc_auc_mean']:.4f} +/- {row['roc_auc_std']:.4f}"
    )

# ---- Section 7: findings generated from the numbers, not hard-coded claims ----
abl = ablation.set_index("representation")
static_acc = abl.loc["Static", "accuracy_mean"]
rep_acc = abl.loc["Repetition_Aware", "accuracy_mean"]
static_auc = abl.loc["Static", "roc_auc_mean"]
rep_auc = abl.loc["Repetition_Aware", "roc_auc_mean"]
abl_stats_path = os.path.join(ABLATION_DIR, "ablation_statistical_comparison.csv")
abl_p = pd.read_csv(abl_stats_path)["wilcoxon_p_fdr"].min() if os.path.exists(abl_stats_path) else float("nan")

if (rep_acc - static_acc) > 0.02 and abl_p < 0.05:
    abl_msg = "Repetition-aware descriptors improved accuracy significantly over static means."
else:
    abl_msg = (
        f"Repetition-aware descriptors did NOT show a statistically supported advantage over static "
        f"per-subject means (accuracy {rep_acc:.3f} vs {static_acc:.3f}; ROC-AUC {rep_auc:.3f} vs {static_auc:.3f}; "
        f"smallest FDR-adjusted Wilcoxon p = {abl_p:.2f}, n = 5 paired folds, i.e. under-powered). "
        f"The claim that repetition-derived features add discriminative value is therefore not demonstrated "
        f"for DCP-vs-TD classification on this dataset."
    )

summary_lines.extend([
    "",
    "7. KEY FINDINGS (generated from the results above):",
    f"- Top-3 by inner-validation accuracy: {', '.join(selected_models)}; tree ensembles and a linear SVM are competitive, "
    "and differences among them are within one fold-SD.",
    f"- {abl_msg}",
    "- Path-straightness features (path efficiency, lateral deviation) are the strongest univariate and model-based separators.",
])

# ---- Fully nested estimate of the whole selection procedure (step 29) ----
NS_DIR = os.path.join(BASE_DIR, "results", "nested_selection")
fl = pd.read_csv(os.path.join(NS_DIR, "nested_selection_folds.csv"))
ns = pd.read_csv(os.path.join(NS_DIR, "nested_selection_ci.csv")).set_index("metric")
fl_acc, fl_sd, fl_auc = fl["accuracy"].mean(), fl["accuracy"].std(), fl["roc_auc"].mean()
def _ns(m):
    return f"{ns.loc[m,'point_estimate']:.3f} [{ns.loc[m,'ci_lower_95']:.3f}, {ns.loc[m,'ci_upper_95']:.3f}]"
summary_lines.extend([
    "",
    "7b. FULLY NESTED ESTIMATE OF THE WHOLE SELECTION PROCEDURE (primary, selection-unbiased estimate):",
    "  Procedure: tune all 16 models by inner CV, keep the family with the best inner-CV accuracy, evaluate it on unseen subjects - "
    "with model family AND hyperparameters chosen separately INSIDE every outer training fold (winners per fold: "
    f"{', '.join(fl['model'])}).",
    f"  Pooled over the 51 held-out subjects (95% subject-level bootstrap CI): Accuracy = {_ns('Accuracy')}; "
    f"Balanced accuracy = {_ns('Balanced_Accuracy')}; F1 = {_ns('F1')}; MCC = {_ns('MCC')}; "
    f"Sensitivity = {_ns('Sensitivity')}; Specificity = {_ns('Specificity')}.",
    f"  Per-fold accuracy mean +/- SD = {fl_acc:.3f} +/- {fl_sd:.3f}.",
    "  The top-3 families of Table 2 were ranked with inner-CV scores averaged over ALL outer folds, i.e. a ranking that indirectly "
    "used every subject; their outer-fold numbers are therefore optimistic for 'the procedure' and must NOT be quoted as an unbiased "
    "estimate. They answer a different question: how well do these three fixed model families work (validation stage required by the "
    "project). Quote the figure above as the expected performance of the end-to-end procedure.",
])

# ---- Section 8: repeated subject-wise nested CV ----
rep_path = os.path.join(BASE_DIR, "results", "repeated_cv", "repeated_cv_summary.csv")
if os.path.exists(rep_path):
    rep = pd.read_csv(rep_path)
    summary_lines.extend(["", "8. REPEATED SUBJECT-WISE NESTED CV (10 re-partitions of the same 51 subjects; mean [95% CI of repeat means]; captures split randomness only; it is NOT generalisation uncertainty - see the uncertainty note below):"])
    for _, r in rep.iterrows():
        summary_lines.append(
            f"  * {r['model']}: "
            f"Acc = {r['accuracy_mean']:.3f} [{r['accuracy_ci95_low']:.3f}, {r['accuracy_ci95_high']:.3f}], "
            f"BalAcc = {r['balanced_accuracy_mean']:.3f} [{r['balanced_accuracy_ci95_low']:.3f}, {r['balanced_accuracy_ci95_high']:.3f}], "
            f"F1 = {r['f1_mean']:.3f} [{r['f1_ci95_low']:.3f}, {r['f1_ci95_high']:.3f}], "
            f"AUC = {r['roc_auc_mean']:.3f} [{r['roc_auc_ci95_low']:.3f}, {r['roc_auc_ci95_high']:.3f}]"
        )

# ---- Section 9: empirical leakage check ----
perm_path = os.path.join(BASE_DIR, "results", "leakage_audit", "label_permutation_test.csv")
if os.path.exists(perm_path):
    pr = pd.read_csv(perm_path).iloc[0]
    summary_lines.extend([
        "",
        "9. EMPIRICAL LEAKAGE CHECK (label permutation, same subject-wise pipeline):",
        f"  Real balanced accuracy = {pr['real_balanced_accuracy']:.3f}; permuted-label mean = "
        f"{pr['permuted_balanced_accuracy_mean']:.3f} (SD {pr['permuted_balanced_accuracy_sd']:.3f}, "
        f"95th pct {pr['permuted_balanced_accuracy_95th_pct']:.3f}); permutation p <= {pr['permutation_p_value']:.4f} (1/1001 floor; single fixed Random Forest, not the 16-model selection) "
        f"({int(pr['n_permutations'])} permutations).",
        f"  Metadata-only baselines (excluded from predictors): subject-ID number alone reaches "
        f"balanced accuracy {pr['metadata_only_subject_id_balanced_acc']:.3f} (identifier leakage risk if used); "
        f"arm side alone {pr['metadata_only_arm_side_balanced_acc']:.3f}.",
    ])

# ---- Section 9b: Functional Resilience Index ----
FRI_DIR = os.path.join(BASE_DIR, "results", "functional_resilience")
fri = pd.read_csv(os.path.join(FRI_DIR, "resilience_group_comparison.csv"))
fri_inc = pd.read_csv(os.path.join(FRI_DIR, "resilience_incremental_summary.csv"))
summary_lines.extend(["", "9b. FUNCTIONAL RESILIENCE MODEL (rep1-rep8; subject-intrinsic, label-free indices):"])
for _, r in fri.iterrows():
    summary_lines.append(
        f"  * {r['index']}: TD median {r['TD_median']:.3f} vs DCP {r['DCP_median']:.3f}; Cohen d (DCP-TD) = {r['cohens_d_DCP_minus_TD']:.2f} "
        f"[{r['d_ci95_low']:.2f}, {r['d_ci95_high']:.2f}]; stand-alone AUC (higher = TD) = {r['AUC_TD_gt_DCP']:.2f} "
        f"[{r['auc_ci95_low']:.2f}, {r['auc_ci95_high']:.2f}]; BH-FDR p = {r['p_fdr_bh']:.4f}")
for _, r in fri_inc.iterrows():
    summary_lines.append(f"  - {r['feature_set']}: acc {r['acc_mean']:.3f} +/- {r['acc_sd']:.3f}, AUC {r['auc_mean']:.3f} {r['note']}")
_sens = pd.read_csv(os.path.join(FRI_DIR, "resilience_weight_sensitivity.csv"))
summary_lines.append("  FRI is a PROPOSED index (equal weights for consistency and maintenance; not clinically validated). Weight sensitivity "
                     "(consistency weight 0.25/0.50/0.75): Cohen d = " + " / ".join(f"{v:.2f}" for v in _sens["cohens_d_DCP_minus_TD"]) +
                     "; AUC = " + " / ".join(f"{v:.2f}" for v in _sens["AUC_TD_gt_DCP"]) + ".")
summary_lines.append("  Interpretation: DCP children show lower consistency/resilience across repetitions (group difference), but adding the indices "
                     "to the static per-subject means does not measurably improve classification; resilience is reported as a descriptive/"
                     "explanatory biomarker, not as a better classifier input.")

# ---- Section 9c: LIME vs SHAP ----
lime_ag = pd.read_csv(os.path.join(BASE_DIR, "results", "lime_oof", "lime_shap_agreement.csv")).iloc[0]
lime_top = pd.read_csv(os.path.join(BASE_DIR, "results", "lime_oof", "oof_lime_summary.csv")).head(5)["feature"].tolist()
summary_lines.extend(["", "9c. LIME (out-of-fold) vs SHAP:",
    f"  Spearman rho over all features = {lime_ag['spearman_rho_all_features']:.2f}; top-15 overlap = {int(lime_ag['top15_overlap'])}/15 "
    f"(Jaccard {lime_ag['top15_jaccard']:.2f}); top-5 LIME features: {', '.join(lime_top)}."])

# ---- Section 9e: feature-set comparison, repeated whole-procedure, confound (coordinate-only primary) ----
_fs = os.path.join(BASE_DIR, "results", "feature_sets")
_rs = os.path.join(BASE_DIR, "results", "repeated_selection", "repeated_selection_summary.csv")
summary_lines.extend(["", "9e. FEATURE-SET COMPARISON (same outer folds; fully nested whole-procedure estimate; PRIMARY = coordinate_only, 84 predictors):"])
if os.path.exists(os.path.join(_fs, "feature_set_headline_comparison.csv")):
    _h = pd.read_csv(os.path.join(_fs, "feature_set_headline_comparison.csv"))
    for _, r in _h.iterrows():
        summary_lines.append(f"  * {r['feature_set']} ({int(r['n_predictors'])} predictors): nested Acc = {r['nested_accuracy']:.3f} [{r['acc_ci_low']:.3f}, {r['acc_ci_high']:.3f}], "
                             f"BalAcc = {r['nested_balanced_accuracy']:.3f}, F1 = {r['nested_f1']:.3f}, MCC = {r['nested_mcc']:.3f}; top-3 by inner CV: {r['top3_by_inner_cv']}")
if os.path.exists(os.path.join(_fs, "paired_subject_level_comparison.csv")):
    for _, r in pd.read_csv(os.path.join(_fs, "paired_subject_level_comparison.csv")).iterrows():
        summary_lines.append(f"  - {r['comparison']}: diff = {r['accuracy_difference']:+.3f} (bootstrap 95% CI [{r['bootstrap_ci95_low']:+.3f}, {r['bootstrap_ci95_high']:+.3f}], exact McNemar p = {r['exact_mcnemar_p']:.2f})")
summary_lines.append("  Ablations: coordinate_no_position_error (target-object feature), coordinate_no_duration (movement_duration_s = recording clip length), "
                     "coordinate_no_duration_no_position_error, coordinate_no_timing_coupled (also drops normalised jerk and SPARC, which track duration). "
                     "Compare the rows above: differences between sets are not statistically distinguishable at n = 51 unless the paired CI above excludes 0. "
                     "The 168-predictor angle_enhanced set uses 2D video-derived joint angles (dataset paper 2.4) and is SECONDARY.")
_dur = os.path.join(BASE_DIR, "results", "duration_audit", "duration_audit_summary.txt")
if os.path.exists(_dur):
    summary_lines.extend(["", "9e-2. RECORDING-DURATION AUDIT (results/duration_audit):"] + ["  " + l for l in open(_dur, encoding="utf-8").read().splitlines()[:3]])
_t3 = os.path.join(BASE_DIR, "results", "nested_top3_per_fold", "per_fold_top3_pooled_metrics.csv")
if os.path.exists(_t3):
    _m = pd.read_csv(_t3); _m = _m[_m.metric == "Accuracy"]
    summary_lines.extend(["", "9e-3. PER-OUTER-FOLD TOP-3 (selection-unbiased version of requirement 6; top-3 chosen from inner CV inside each outer training fold):"] +
        [f"  {r.procedure}: accuracy {r.point_estimate:.3f} [{r.ci_lower_95:.3f}, {r.ci_upper_95:.3f}]" for r in _m.itertuples()] +
        ["  (rank slots = model ranked 1st/2nd/3rd in each fold; families differ between folds; majority vote = label vote of the three selected models.)"])
summary_lines.append("  HISTORICAL: an earlier 168-predictor full-feature run is superseded; the angle_enhanced row above is its current, re-run equivalent.")
if os.path.exists(_rs):
    _r = pd.read_csv(_rs).set_index("metric")
    summary_lines.extend(["", "9f. WHOLE SELECTION PROCEDURE REPEATED OVER 10 SUBJECT-WISE PARTITIONS (primary set; each repetition re-runs the complete 16-model nested selection):",
        f"  Accuracy mean = {_r.loc['accuracy','mean']:.3f} (SD across partitions {_r.loc['accuracy','sd_across_partitions']:.3f}; min {_r.loc['accuracy','min']:.3f}, max {_r.loc['accuracy','max']:.3f}); "
        f"balanced accuracy = {_r.loc['balanced_accuracy','mean']:.3f}; MCC = {_r.loc['mcc','mean']:.3f}; mean fold AUC = {_r.loc['mean_fold_roc_auc','mean']:.3f}.",
        "  The seed-42 partition is the fixed default headline (set before inspecting results); it is at the low end of the ten partitions (tied lowest), so the repeated mean is shown alongside rather than substituted. Same 51 subjects: split sensitivity only."])
summary_lines.extend(["", "9g. ACQUISITION CONFOUND: see results/confound_analysis/confound_summary.txt. Group and recording batch cannot be separated; high accuracy may partly reflect recording setup; independent site-balanced validation required."])

# ---- Section 9d: what each uncertainty number does and does not mean ----
summary_lines.extend(["", "9d. HOW TO READ THE UNCERTAINTY NUMBERS:",
    "- Fold SD (+/-): spread of 5 outer folds of ~10 subjects each; coarse (10-point steps), not a CI.",
    "- Repeated-split variability (Table 3): re-partitions the SAME 51 subjects; shows sensitivity to the split only. Intervals are too narrow to be read as generalisation uncertainty.",
    "- Bootstrap CIs on pooled out-of-fold predictions (Tables 1b, 2): resample the 51 subjects, so they reflect subject-sampling noise in the test predictions, but "
    "they hold the trained models fixed and (for Table 2) the model-family choice fixed; they do not include training-set or model-selection variability. Table 1b includes selection inside every fold but still not the variability of re-drawing a different training cohort.",
    "- None of them measure generalisation to new sites, scanners, clinicians or populations: all data come from one dataset (site/acquisition confound unresolved).",
    "- 'Final test' = outer cross-validation folds. There is NO independent hold-out or external test set (impossible with n = 51 without discarding ~20% of subjects)."])

summary_lines.extend([
    "",
    "10. LIMITATIONS",
    "- n = 51 subjects (about 10 test subjects per fold): fold-level metrics move in ~10-point steps; "
    "report CIs/variance, not point values. All estimates are pilot-level.",
    "- Acquisition confound: the supplied data contain no site/session/camera variable and subject-number blocks are structured by group (see results/confound_analysis/confound_summary.txt); "
    "the dataset paper describes per-lab DeepLabCut models, so recording setup cannot be separated from group effects. High accuracy may partly reflect recording setup; an independent, site-balanced cohort is needed.",
    "- Pixel-to-mm conversion uses a fixed 2 mm/pixel (paper reports it is approximated from target size); absolute-length features are approximate.",
    "- DCP-vs-TD is an easy, large-effect task; high scores do not imply clinical severity grading or generalisation to other populations.",
    "- Model ranking by inner-CV accuracy is itself noisy at this n; the top-3 are statistically indistinguishable.",
    "- No external test cohort and no independent hold-out set: nested subject-wise CV is the evaluation; recording-site/acquisition confounding "
    "remains UNRESOLVED (no site variable to adjust for; a subject-ID-only baseline already separates the groups).",
])

# ---- Markdown results table (all 16 models + selected models' validation) ----
md = ["# Results summary", "",
      "Task: binary classification, DCP (1, n=30) vs TD (0, n=21); unit = subject; subject-wise nested CV "
      "(outer 5-fold / inner 4-fold StratifiedGroupKFold, seed 42).", "",
      "## Table 1 - all 16 models (outer-fold test metrics, mean +/- SD over 5 folds; ranked by inner-CV accuracy)", "",
      "| Rank | Model | Inner-CV acc | Test acc | Test bal. acc | Test F1 | Test ROC-AUC | Selected |",
      "|---|---|---|---|---|---|---|---|"]
for _, r in ranking.iterrows():
    sel = "YES" if r["model"] in selected_models else ""
    md.append(f"| {int(r['inner_accuracy_rank'])} | {r['model']} | {r['inner_accuracy_mean']:.3f} +/- {r['inner_accuracy_std']:.3f} | "
              f"{r['test_accuracy_mean']:.3f} +/- {r['test_accuracy_std']:.3f} | "
              f"{r['test_balanced_accuracy_mean']:.3f} +/- {r['test_balanced_accuracy_std']:.3f} | "
              f"{r['test_f1_mean']:.3f} +/- {r['test_f1_std']:.3f} | "
              f"{r['test_roc_auc_mean']:.3f} +/- {r['test_roc_auc_std']:.3f} | {sel} |")
ci_path = os.path.join(BASE_DIR, "results", "bootstrap_ci", "top3_bootstrap_ci.csv")
md += ["", "## Table 1b - PRIMARY unbiased estimate: whole procedure (16 models tuned by inner CV, best family by inner-CV accuracy) selected INSIDE each outer training fold", "",
       "| Fold | Winner (inner-CV) | Inner acc | Test acc | Test ROC-AUC |", "|---|---|---|---|---|"]
for _, r in fl.iterrows():
    md.append(f"| {int(r['fold'])} | {r['model']} | {r['inner_accuracy']:.3f} | {r['accuracy']:.3f} | {r['roc_auc']:.3f} |")
md.append(f"| **Mean +/- SD** | | | **{fl_acc:.3f} +/- {fl_sd:.3f}** | **{fl_auc:.3f}** |")
md += ["", "Pooled over the 51 held-out subjects, 95% subject-level bootstrap CI (10,000 resamples):", "",
       "| Metric | Estimate [95% CI] |", "|---|---|"]
for m_ in ["Accuracy", "Balanced_Accuracy", "F1", "Sensitivity", "Specificity", "MCC"]:
    md.append(f"| {m_} | {_ns(m_)} |")
md += ["", "## Table 2 - the 3 selected model families, validation stage (pooled out-of-fold predictions, 10,000 subject-level bootstrap resamples, 95% CI). OPTIMISTIC as an estimate of the selection procedure (families ranked with all folds); see Table 1b.", "",
       "| Model | Accuracy | Balanced acc | F1 | ROC-AUC |", "|---|---|---|---|---|"]
if os.path.exists(ci_path):
    ci = pd.read_csv(ci_path)
    for mname in selected_models:
        g = ci[ci["model"] == mname].set_index("metric")
        cells = [f"{g.loc[m,'point_estimate']:.3f} [{g.loc[m,'ci_lower_95']:.3f}, {g.loc[m,'ci_upper_95']:.3f}]"
                 for m in ["Accuracy", "Balanced_Accuracy", "F1", "ROC-AUC"]]
        md.append(f"| {mname} | " + " | ".join(cells) + " |")
if os.path.exists(rep_path):
    md += ["", "## Table 3 - selected models, repeated subject-wise nested CV (10 partitions; mean [95% CI of repeat means])", "",
           "_Caveat: the 10 repeats re-partition the same 51 subjects, so these intervals capture split randomness only and are much narrower than the true generalisation uncertainty; they are not confidence intervals for new participants._", "",
           "| Model | Accuracy | Balanced acc | F1 | ROC-AUC |", "|---|---|---|---|---|"]
    for _, r in rep.iterrows():
        md.append(f"| {r['model']} | " + " | ".join(
            f"{r[m+'_mean']:.3f} [{r[m+'_ci95_low']:.3f}, {r[m+'_ci95_high']:.3f}]"
            for m in ["accuracy", "balanced_accuracy", "f1", "roc_auc"]) + " |")
md += ["", "## Table 4 - representation ablation (single Random Forest per representation, nested CV)", "",
       "| Representation | Accuracy | F1 | ROC-AUC |", "|---|---|---|---|"]
for _, r in ablation.iterrows():
    md.append(f"| {r['representation']} | {r['accuracy_mean']:.3f} +/- {r['accuracy_std']:.3f} | "
              f"{r['f1_mean']:.3f} +/- {r['f1_std']:.3f} | {r['roc_auc_mean']:.3f} +/- {r['roc_auc_std']:.3f} |")
md += ["", f"Ablation conclusion: {abl_msg}"]
md += ["", "## Table 5 - Functional Resilience Index and related repetition indices (TD vs DCP)", "",
       "| Index | TD median | DCP median | Cohen d (DCP-TD) [95% CI] | AUC, higher=TD [95% CI] | BH-FDR p |", "|---|---|---|---|---|---|"]
for _, r in fri.iterrows():
    md.append(f"| {r['index']} | {r['TD_median']:.3f} | {r['DCP_median']:.3f} | {r['cohens_d_DCP_minus_TD']:.2f} [{r['d_ci95_low']:.2f}, {r['d_ci95_high']:.2f}] | "
              f"{r['AUC_TD_gt_DCP']:.2f} [{r['auc_ci95_low']:.2f}, {r['auc_ci95_high']:.2f}] | {r['p_fdr_bh']:.4f} |")
md += ["", "Incremental classification value (10 repeated subject-wise 5-fold partitions, RF, same pipeline):", "",
       "| Feature set | Accuracy | ROC-AUC | Note |", "|---|---|---|---|"]
for _, r in fri_inc.iterrows():
    md.append(f"| {r['feature_set']} | {r['acc_mean']:.3f} +/- {r['acc_sd']:.3f} | {r['auc_mean']:.3f} | {r['note'] if isinstance(r['note'], str) else ''} |")
md += ["", "FRI is a proposed index (equal weights for consistency and maintenance; not clinically validated). Weight sensitivity (consistency weight 0.25 / 0.50 / 0.75): "
       "Cohen d = " + " / ".join(f"{v:.2f}" for v in _sens["cohens_d_DCP_minus_TD"]) + "; AUC = " + " / ".join(f"{v:.2f}" for v in _sens["AUC_TD_gt_DCP"]) + "."]
md += ["", "## Table 6 - Out-of-fold LIME vs SHAP agreement", "",
       f"Spearman rho (all features) = {lime_ag['spearman_rho_all_features']:.2f}; top-15 overlap = {int(lime_ag['top15_overlap'])}/15 "
       f"(Jaccard {lime_ag['top15_jaccard']:.2f}). Top LIME features: {', '.join(lime_top)}.", "",
       "## How to read the uncertainty numbers", ""] + [l for l in summary_lines[summary_lines.index("9d. HOW TO READ THE UNCERTAINTY NUMBERS:")+1:summary_lines.index("10. LIMITATIONS")-1]]

with open(os.path.join(OUTPUT_DIR, "results_table.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(md) + "\n")

summary_text = "\n".join(summary_lines)
summary_txt_path = os.path.join(OUTPUT_DIR, "final_results_summary.txt")
with open(summary_txt_path, "w", encoding="utf-8") as f:
    f.write(summary_text)

print()
print("=" * 70)
print("FILES CREATED")
print("=" * 70)
print(ranking_path)
print(master_path)
print(rep_comp_path)
print(summary_txt_path)
print("\nSTEP 24 COMPLETE")
