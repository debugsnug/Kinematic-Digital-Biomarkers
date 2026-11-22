# ============================================================
# STEP 42 - FINAL FIGURE SET (reads saved CSVs only; no model is refitted)
# ============================================================
# results/figures_final/ is the curated set for the report; results/figures/ keeps the older per-step plots (diagnostic,
# several explore the exploratory Random Forest). Captions with caveats are written to captions.md.
import os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); R = os.path.join(BASE, "results")
OUT = os.path.join(R, "figures_final"); os.makedirs(OUT, exist_ok=True)
rd = lambda *p: pd.read_csv(os.path.join(R, *p))
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})
C = {"nested": "#1f5fa8", "other": "#8a8f98", "bad": "#c4572b", "good": "#2a8c6a"}
def save(f, n): f.tight_layout(); f.savefig(os.path.join(OUT, n + ".png"), dpi=200); plt.close(f)
caps = {}

# F1 evidence hierarchy
ci = rd("nested_selection", "nested_selection_ci.csv").set_index("metric").loc["Accuracy"]
pf = rd("nested_top3_per_fold", "per_fold_top3_pooled_metrics.csv"); vote = pf[(pf.procedure == "majority_vote_of_top3") & (pf.metric == "Accuracy")].iloc[0]
rep = rd("repeated_selection", "repeated_selection_summary.csv").set_index("metric").loc["accuracy"]
g3 = rd("top3_validation", "top3_classification_metrics.csv").accuracy
items = [("Fully nested selection\n(PRIMARY, 51 subjects)", ci.point_estimate, ci.ci_lower_95, ci.ci_upper_95, C["nested"]),
         ("Per-fold top-3 majority vote\n(selection-unbiased)", float(vote.point_estimate), float(vote.ci_lower_95), float(vote.ci_upper_95), C["other"]),
         ("Repeated nested, 10 partitions\n(mean; bar = min-max)", rep["mean"], rep["min"], rep["max"], C["other"]),
         ("Global top-3, best model\n(OPTIMISTIC: ranked on pooled data)", g3.max(), np.nan, np.nan, C["bad"])]
f, ax = plt.subplots(figsize=(8, 3.8))
for i, (n, v, lo, hi, c) in enumerate(items):
    ax.barh(i, v, color=c, height=.55)
    if np.isfinite(lo): ax.errorbar(v, i, xerr=[[v - lo], [hi - v]], color="k", capsize=3)
    ax.text(0.02, i, f"{v:.3f}", va="center", color="w", fontweight="bold")
ax.axvline(30 / 51, ls="--", c="k", lw=1, label="always predict DCP (0.588)"); ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=8)
ax.set_yticks(range(4)); ax.set_yticklabels([i[0] for i in items], fontsize=8.5); ax.invert_yaxis(); ax.set_xlim(0, 1.02)
ax.set_xlabel("Subject-level accuracy"); ax.set_title("Evidence hierarchy: only the nested estimate is a primary claim", fontsize=10)
save(f, "F1_evidence_hierarchy")
caps["F1"] = "Accuracy under four evaluation procedures. Blue = primary (fold-wise nested selection; 95% subject bootstrap CI). Red = global top-3 ranking, which uses pooled inner scores and is optimistic. Dashed = majority-class rate. Single cohort, 51 subjects: no external validation."

# F2 fold-wise
nf = rd("nested_selection", "nested_selection_folds.csv")
f, ax = plt.subplots(figsize=(7, 3.6)); x = np.arange(len(nf))
ax.bar(x - .2, nf.accuracy, .4, label="Accuracy", color=C["nested"]); ax.bar(x + .2, nf.balanced_accuracy, .4, label="Balanced accuracy", color=C["other"])
ax.set_xticks(x); ax.set_xticklabels([f"Fold {a}\n{b}" for a, b in zip(nf.fold, nf.model)], fontsize=8); ax.set_ylim(0, 1.18); ax.legend(frameon=False, ncol=2, loc="upper center")
ax.set_ylabel("Held-out score (about 10 subjects per fold)"); ax.set_title("Fold-wise performance of the nested winners", fontsize=10)
save(f, "F2_foldwise_performance")
caps["F2"] = "Each outer fold chooses its own model family and hyper-parameters from training subjects only; ~10 test subjects per fold, so single-fold values move in steps of ~0.1."

# F3 ablation
H = rd("feature_sets", "feature_set_headline_comparison.csv")
f, ax = plt.subplots(figsize=(8, 3.8)); y = np.arange(len(H))
ax.errorbar(H.nested_accuracy, y, xerr=[H.nested_accuracy - H.acc_ci_low, H.acc_ci_high - H.nested_accuracy], fmt="o", color=C["nested"], capsize=3)
ax.set_yticks(y); ax.set_yticklabels([f"{a} ({b})" for a, b in zip(H.feature_set, H.n_predictors)], fontsize=8.5); ax.invert_yaxis()
ax.axvline(30 / 51, ls="--", c="k", lw=1, label="always predict DCP (0.588)"); ax.legend(frameon=False, loc="lower left", fontsize=8); ax.set_xlim(.5, 1); ax.set_xlabel("Nested accuracy (95% bootstrap CI)"); ax.set_title("Feature-set ablation: all intervals overlap", fontsize=10)
save(f, "F3_feature_set_ablation")
caps["F3"] = "Same nested procedure on six feature sets (predictor counts in brackets). Removing duration, position error or both timing-coupled smoothness measures does not lower accuracy beyond the CI width; the differences are not statistically distinguishable at n=51. Accuracy is not carried by duration alone, but this does not remove the acquisition confound."

# F4 acquisition proxies
A = rd("confound_analysis", "acquisition_proxy_classifiers.csv"); A["p"] = A.proxy_set.str.split(" ").str[0]
A = A.groupby("p").balanced_accuracy_mean.max().sort_values()
f, ax = plt.subplots(figsize=(7, 3.2)); ax.barh(A.index, A.values, color=C["bad"]); ax.axvline(.5, ls="--", c="k", lw=1)
for i, v in enumerate(A.values): ax.text(v + .01, i, f"{v:.2f}", va="center")
ax.set_xlim(.4, 1); ax.set_xlabel("Best balanced accuracy of a classifier using only this variable"); ax.set_title("Acquisition proxies separate the groups without any movement features", fontsize=10)
save(f, "F4_acquisition_proxies")
caps["F4"] = "Clip length (equal to movement_duration_s) and subject number alone separate DCP from TD far above chance (dashed = 0.5). The supplied data has no site/camera variable, so the confound cannot be resolved here."

# F5 FRI components
F = rd("functional_resilience", "fri_component_analysis.csv")
S = F[F.analysis.isin(["full", "single metric FRI", "leave one out", "timing removed"])].copy()
S["label"] = S.analysis.map({"full": "", "single metric FRI": "only: ", "leave one out": "", "timing removed": ""}) + S.name
f, ax = plt.subplots(figsize=(8, 5)); y = np.arange(len(S))
cols = [C["nested"] if a == "full" else C["bad"] if a == "timing removed" else C["other"] for a in S.analysis]
ax.barh(y, S.AUC_TD_gt_DCP, color=cols); ax.axvline(.5, ls="--", c="k", lw=1)
for i, (v, p) in enumerate(zip(S.AUC_TD_gt_DCP, S.p_fdr_bh)): ax.text(v + .01, i, f"AUC {v:.2f}, FDR p " + ("<0.001" if p < .001 else f"={p:.3f}"), va="center", fontsize=7.5)
ax.set_yticks(y); ax.set_yticklabels(S.label, fontsize=8); ax.invert_yaxis(); ax.set_xlim(0, 1.25)
ax.set_xlabel("AUC (TD index > DCP index; 0.5 = no separation)"); ax.set_title("FRI components: AUC falls when timing-coupled metrics are removed", fontsize=10)
save(f, "F5_FRI_components")
caps["F5"] = "AUC of the exploratory FRI, of FRI built from one metric, and with one metric left out. Removing SPARC and normalised jerk (red) reduces AUC to 0.64 (BH-FDR p=0.13); max lateral deviation works against the index (leaving it out raises AUC). The FRI is not clinically validated."

# F6 importance across sets
I = rd("importance_across_feature_sets", "importance_by_set.csv")
top = I[I.feature_set == "coordinate_only"].nsmallest(8, "rank").base_feature.tolist()
P = I[I.base_feature.isin(top)].pivot(index="base_feature", columns="feature_set", values="mean_abs_shap_sum").reindex(top)
f, ax = plt.subplots(figsize=(8.5, 4.2)); w = .8 / P.shape[1]
for j, c in enumerate(P.columns): ax.bar(np.arange(len(P)) + j * w, P[c].fillna(0), w, label=c)
ax.set_xticks(np.arange(len(P)) + .4 - w / 2); ax.set_xticklabels(P.index, rotation=30, ha="right", fontsize=8); ax.set_ylabel("Out-of-fold mean |SHAP| (summed over descriptors)")
ax.legend(fontsize=6.5, frameon=False); ax.set_title("Feature importance is stable across feature sets (exploratory Random Forest)", fontsize=10)
save(f, "F6_importance_across_feature_sets")
caps["F6"] = "SHAP of a fixed Random Forest (k=40) per feature set, held-out subjects only. Rank agreement between sets is high (Spearman 0.96-1.00; rank_agreement_between_sets.csv). Explains this exploratory forest, not the nested winners."

# F7 nested importance
N = rd("nested_pipeline_explanations", "nested_pipeline_importance.csv").head(10)
f, ax = plt.subplots(figsize=(7.5, 3.8)); ax.barh(N.base_feature, N.mean_importance, xerr=N.sd_across_folds, color=C["nested"], capsize=2); ax.invert_yaxis(); ax.axvline(0, c="k", lw=.8)
ax.set_xlabel("Held-out accuracy drop when feature is permuted (mean ± SD over 5 folds)"); ax.set_title("Permutation importance of the nested winners: weak and unstable", fontsize=10)
save(f, "F7_nested_pipeline_importance")
caps["F7"] = "Each fold's winning pipeline permuted on ~10 held-out subjects. Effects are small relative to their spread and disagree with the SHAP ranking; no single feature is established as driving the nested model. Treat as descriptive."

# F8 univariate effects
U = rd("statistical_analysis", "significant_features.csv").assign(a=lambda d: d.cohens_d.abs()).nlargest(12, "a")
f, ax = plt.subplots(figsize=(8, 4.5)); ax.barh(U.feature, U.cohens_d, color=[C["bad"] if v > 0 else C["nested"] for v in U.cohens_d]); ax.invert_yaxis(); ax.axvline(0, c="k", lw=.8)
for i, p in enumerate(U.p_adjusted): ax.text(0.02 if U.cohens_d.iloc[i] < 0 else -0.02, i, f"FDR p={p:.1e}", va="center", ha="left" if U.cohens_d.iloc[i] < 0 else "right", fontsize=7)
ax.set_xlabel("Cohen's d (DCP minus TD)"); ax.set_title("Largest univariate group differences (Mann-Whitney, BH-FDR)", fontsize=10)
save(f, "F8_univariate_effects")
caps["F8"] = "Twelve largest |d| among the FDR-significant features. Group differences are descriptive; duration-related features (clip length) are confounded with acquisition."
with open(os.path.join(OUT, "captions.md"), "w") as f: f.write("# Final figure captions\n\n" + "\n\n".join(f"**{k}.** {v}" for k, v in caps.items()) + "\n")
print("figures:", sorted(os.listdir(OUT)))
