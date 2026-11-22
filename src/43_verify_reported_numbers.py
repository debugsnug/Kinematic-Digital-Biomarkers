# ============================================================
# STEP 43 - VERIFY THAT REPORTED HEADLINE NUMBERS MATCH THE SAVED CSVs
# ============================================================
# The checklist numbers (README / report / slides) are asserted here against the result files. Exit code 1 on any mismatch.
import os, sys, pandas as pd
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
rd = lambda *p: pd.read_csv(os.path.join(R, *p)); bad = []
def chk(name, got, want, tol=5e-4):
    ok = abs(got - want) <= tol; print(("OK   " if ok else "FAIL ") + f"{name}: csv={got:.4f} reported={want}"); ok or bad.append(name)
ci = rd("nested_selection", "nested_selection_ci.csv").set_index("metric").point_estimate
for k, v in {"Accuracy": .824, "Balanced_Accuracy": .814, "Sensitivity": .867, "Specificity": .762, "F1": .852}.items(): chk("nested " + k, ci[k], v)
pf = rd("nested_top3_per_fold", "per_fold_top3_pooled_metrics.csv"); chk("per-fold top-3 vote accuracy", pf[(pf.procedure == "majority_vote_of_top3") & (pf.metric == "Accuracy")].point_estimate.iloc[0], .863)
pr = rd("nested_selection", "nested_selection_predictions.csv"); chk("nested correct subjects (of 51)", (pr.true_target == pr.predicted_target).sum() / 51, 42 / 51)
H = rd("feature_sets", "feature_set_headline_comparison.csv").set_index("feature_set")
for k, v in {"coordinate_only": .824, "angle_enhanced": .863, "coordinate_no_position_error": .882, "coordinate_no_duration": .843, "coordinate_no_duration_no_position_error": .824, "coordinate_no_timing_coupled": .843}.items(): chk("ablation " + k, H.nested_accuracy[k], v)
F = rd("functional_resilience", "fri_component_analysis.csv").set_index("name"); chk("FRI AUC", F.loc["FRI (5 metrics)", "AUC_TD_gt_DCP"], .754); chk("FRI AUC without sparc+jerk", F.loc["FRI without sparc+norm_jerk", "AUC_TD_gt_DCP"], .643)
chk("FRI FDR p (step 30 family of 7 indices)", rd("functional_resilience", "resilience_group_comparison.csv").set_index("index").loc["Functional_Resilience_Index", "p_fdr_bh"], .0068, 5e-5)
print("\nALL REPORTED NUMBERS MATCH" if not bad else "\nMISMATCH: " + ", ".join(bad)); sys.exit(1 if bad else 0)
