# ============================================================
# STEP 30 - FUNCTIONAL RESILIENCE MODEL (rep1-rep8)
# ============================================================
# Project objective 6: model functional resilience from the repeatability of the
# eight repetitions. All indices below are SUBJECT-INTRINSIC: each is computed from
# one child's own eight repetitions, uses no label and no cross-subject statistic
# (no scaling by the cohort), so they cannot leak the target and need no fold-wise fitting.
#
# Primary quality metrics (sign-oriented so that larger = better movement quality):
#   path_efficiency(+) position_error_mm(-) max_lateral_deviation_mm(-)
#   smoothness_sparc(+) smoothness_norm_jerk(-)
# Recording duration is deliberately EXCLUDED from every primary FRI component:
# it is exactly clip length ((frames-1)/FPS) and is strongly group-associated.
# Joint-angle metrics for stability: elbow_rom_deg, wrist_rom_deg, elbow/wrist_angle_sd
# (shoulder angles are missing for most subjects and are not used).
#
#   Consistency Score  CS   = mean_k 1/(1 + CV_k)            CV across the 8 reps of metric k
#   Movement Stability MSI  = mean_j 1/(1 + CV_j)            CV across reps of joint-angle metric j
#   Adaptive Motor Control AMCI = mean_k (late - early)/scale_k   early = reps 1-4, late = reps 5-8;
#                                 >0 means quality improved, <0 means it deteriorated
#   Functional Resilience Index  FRI = 0.5*CS + 0.5*mean_k exp(min(0, (late-early)/scale_k))
#                                 i.e. consistency plus the ability to MAINTAIN quality (no penalty
#                                 for improvement, exponential penalty for deterioration); FRI in (0,1].
#   scale_k = mean |value| of metric k over the subject's 8 reps.
#
# WHY THIS FORMULA (a proposed index, NOT a clinically validated measure)
#   * Construct: "resilience" = keeping movement quality steady AND not letting it decline as the
#     task is repeated. Consistency (low between-repetition spread) captures steadiness only; a child can
#     be consistently bad or consistently get worse. Maintenance (no early->late decline) captures
#     decline only; a child can be erratic yet end level. FRI needs both, so it averages them.
#   * Equal weights: no clinical evidence or training label is available to justify any other weighting,
#     and fitting weights to the DCP/TD label would make the index label-dependent (circular, leaky).
#     0.5/0.5 is the prior-free choice; the script reports a sensitivity check with weights 0.25/0.75
#     (resilience_weight_sensitivity.csv) so the reader can see the group result does not hinge on it.
#   * Maintenance uses exp(min(0, d)): improvement is not rewarded beyond 1 (a child who warms up is not
#     "more resilient" than one who stays level), deterioration is penalised smoothly; d is relative to
#     the child's own mean so units/magnitudes of different metrics are comparable without cohort scaling.
#   * Missing data: each metric is NaN-aware. A metric with <3 valid repetitions (CV) or no valid early or
#     late repetition (maintenance) is skipped and the index averages over the remaining metrics (nanmean);
#     the number of metrics actually used per child is saved (n_quality_metrics_used, n_joint_metrics_used).
#     A child with no usable metric would get NaN (none occur here; asserted below). Shoulder angles are
#     missing for most children and are excluded by design.
#   * Limits: the six quality metrics and the 4-vs-4 early/late split are expert choices, not validated
#     against clinical scales (e.g. MACS); the index is descriptive/exploratory.
#
# INPUT MODALITY (verified by 32_feature_audit.py): Consistency Score, Adaptive Motor Control Index, the Maintenance
# score and the FRI use only coordinate-derived quality metrics (path efficiency, max lateral deviation, SPARC, jerk)
# plus position_error_mm, which depends on the tracked target-object key point. Recording duration is excluded. The Movement Stability
# Index uses joint-angle metrics (2D video-derived angles) and is therefore ANGLE-DERIVED: it is reported for completeness
# but is not part of the FRI and is excluded from the coordinate-only classification experiment. Because position_error is
# the one target-dependent input, FRI is also reported without it (Functional_Resilience_Index_no_position_error).
#
# Evaluation: (a) TD vs DCP comparison with effect sizes, bootstrap CIs, BH-FDR;
# (b) AUC of each index as a stand-alone score with direction fixed a priori (higher = TD);
# (c) incremental value for classification: static means vs static means + the 4 indices,
#     identical pipeline, 10 repeated subject-wise 5-fold partitions.

import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "results", "functional_resilience")
os.makedirs(OUT, exist_ok=True)
SEED, N_BOOT, N_REPEATS = 42, 10000, 10

long = pd.read_csv(os.path.join(BASE, "data", "processed", "repetition_trajectories_long.csv"))
exp = pd.read_csv(os.path.join(BASE, "data", "processed", "experiment_dataset.csv"))

QUALITY = {"path_efficiency": +1, "position_error_mm": -1, "max_lateral_deviation_mm": -1,
           "smoothness_sparc": +1, "smoothness_norm_jerk": -1}
assert "movement_duration_s" not in QUALITY, "Recording duration must not enter the primary FRI"
JOINT = ["elbow_rom_deg", "wrist_rom_deg", "elbow_angle_sd", "wrist_angle_sd"]

def cv_term(v):
    v = v[np.isfinite(v)]
    if len(v) < 3 or abs(v.mean()) < 1e-12:
        return np.nan
    return 1.0 / (1.0 + v.std(ddof=0) / abs(v.mean()))

rows = []
for subj, g in long.sort_values(["subject", "rep"]).groupby("subject"):
    assert len(g) == 8
    cs, ret, amci = {}, {}, {}                      # per quality metric (NaN-aware; missing metrics are skipped)
    for m, sgn in QUALITY.items():
        v = g[m].to_numpy(float)
        cs[m] = cv_term(v)
        ok = np.isfinite(v)
        early, late = v[:4][ok[:4]], v[4:][ok[4:]]
        scale = np.abs(v[ok]).mean() if ok.any() else np.nan
        if len(early) and len(late) and scale > 1e-12:
            d = sgn * (late.mean() - early.mean()) / scale
            amci[m] = d
            ret[m] = np.exp(min(0.0, d))
        else:
            amci[m] = ret[m] = np.nan
    msi = [cv_term(g[j].to_numpy(float)) for j in JOINT]
    n_q, n_j = int(np.sum(np.isfinite(list(cs.values())))), int(np.sum(np.isfinite(msi)))
    nope = [m for m in QUALITY if m != "position_error_mm"]
    CS, MSI = np.nanmean(list(cs.values())), np.nanmean(msi)
    AMCI, RET = np.nanmean(list(amci.values())), np.nanmean(list(ret.values()))
    CS2, RET2 = np.nanmean([cs[m] for m in nope]), np.nanmean([ret[m] for m in nope])
    ntc = [m for m in QUALITY if m not in ("smoothness_sparc", "smoothness_norm_jerk")]   # also drop duration-coupled smoothness metrics (step 37)
    CS3, RET3 = np.nanmean([cs[m] for m in ntc]), np.nanmean([ret[m] for m in ntc])
    # Legacy comparator only: reproduces the former duration-inclusive construction for audit transparency.
    duration = g["movement_duration_s"].to_numpy(float)
    duration_cs = cv_term(duration)
    duration_ok = np.isfinite(duration)
    duration_early, duration_late = duration[:4][duration_ok[:4]], duration[4:][duration_ok[4:]]
    duration_scale = np.abs(duration[duration_ok]).mean() if duration_ok.any() else np.nan
    duration_ret = (np.exp(min(0.0, -(duration_late.mean() - duration_early.mean()) / duration_scale))
                    if len(duration_early) and len(duration_late) and duration_scale > 1e-12 else np.nan)
    legacy_cs = np.nanmean([*cs.values(), duration_cs])
    legacy_ret = np.nanmean([*ret.values(), duration_ret])
    rows.append({"subject": subj, "group": g.group.iloc[0],
                 "Consistency_Score": CS, "Movement_Stability_Index": MSI,
                 "Adaptive_Motor_Control_Index": AMCI,
                 "Maintenance_Score": RET,
                 "n_quality_metrics_used": n_q, "n_joint_metrics_used": n_j,
                 "Functional_Resilience_Index": 0.5 * CS + 0.5 * RET,
                 "Functional_Resilience_Index_no_position_error": 0.5 * CS2 + 0.5 * RET2,
                 "FRI_no_timing_coupled_sensitivity": 0.5 * CS3 + 0.5 * RET3,
                 "FRI_duration_included_legacy_sensitivity_only": 0.5 * legacy_cs + 0.5 * legacy_ret})
idx = pd.DataFrame(rows)
idx["target"] = (idx.group == "DCP").astype(int)
assert idx[["Consistency_Score", "Maintenance_Score", "Functional_Resilience_Index", "Functional_Resilience_Index_no_position_error"]].notna().all().all(), "a child has no usable metric"
print(f"Metrics used per child (of {len(QUALITY)} quality / {len(JOINT)} joint): "
      f"quality min {idx.n_quality_metrics_used.min()}, joint min {idx.n_joint_metrics_used.min()}")
idx.to_csv(os.path.join(OUT, "resilience_indices_by_subject.csv"), index=False)
assert set(idx.subject) == set(exp.subject)

NAMES = ["Functional_Resilience_Index", "Functional_Resilience_Index_no_position_error", "Consistency_Score",
         "Movement_Stability_Index", "Adaptive_Motor_Control_Index"]
# The duration-inclusive legacy comparator is diagnostic only and is never used in primary group or ML analyses.
# Movement_Stability_Index is angle-derived: excluded from the coordinate-only classification experiment below
COORD_INDICES = [n for n in NAMES if n != "Movement_Stability_Index"]
rng = np.random.default_rng(SEED)
td, dcp = idx[idx.target == 0], idx[idx.target == 1]

def cohen_d(a, b):
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (b.mean() - a.mean()) / sp                      # DCP - TD

res = []
for nme in NAMES + ["FRI_no_timing_coupled_sensitivity", "FRI_duration_included_legacy_sensitivity_only"]:
    a, b = td[nme].dropna().to_numpy(), dcp[nme].dropna().to_numpy()
    p = mannwhitneyu(a, b, alternative="two-sided").pvalue
    ds, aucs = [], []
    for _ in range(N_BOOT):
        aa, bb = rng.choice(a, len(a)), rng.choice(b, len(b))
        ds.append(cohen_d(aa, bb))
        aucs.append((aa[:, None] > bb[None, :]).mean() + 0.5 * (aa[:, None] == bb[None, :]).mean())
    auc = (a[:, None] > b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean()   # P(TD index > DCP index)
    res.append({"index": nme, "TD_median": np.median(a), "DCP_median": np.median(b),
                "cohens_d_DCP_minus_TD": cohen_d(a, b),
                "d_ci95_low": np.percentile(ds, 2.5), "d_ci95_high": np.percentile(ds, 97.5),
                "AUC_TD_gt_DCP": auc, "auc_ci95_low": np.percentile(aucs, 2.5),
                "auc_ci95_high": np.percentile(aucs, 97.5), "mannwhitney_p": p})
R = pd.DataFrame(res)
m = len(R); order = np.argsort(R.mannwhitney_p.values)
adj = np.minimum.accumulate((R.mannwhitney_p.values[order] * m / (np.arange(m) + 1))[::-1])[::-1]
R.loc[order, "p_fdr_bh"] = np.minimum(adj, 1.0)
R.to_csv(os.path.join(OUT, "resilience_group_comparison.csv"), index=False)

# ---- weight sensitivity of the FRI (consistency weight w, maintenance weight 1-w) ----
sens = []
for w in (0.25, 0.5, 0.75):
    f = w * idx.Consistency_Score + (1 - w) * idx.Maintenance_Score
    a_, b_ = f[idx.target == 0].to_numpy(), f[idx.target == 1].to_numpy()
    sens.append({"consistency_weight": w, "maintenance_weight": 1 - w, "cohens_d_DCP_minus_TD": cohen_d(a_, b_),
                 "AUC_TD_gt_DCP": (a_[:, None] > b_[None, :]).mean() + 0.5 * (a_[:, None] == b_[None, :]).mean(),
                 "mannwhitney_p": mannwhitneyu(a_, b_, alternative="two-sided").pvalue})
pd.DataFrame(sens).to_csv(os.path.join(OUT, "resilience_weight_sensitivity.csv"), index=False)
print("\nFRI WEIGHT SENSITIVITY"); print(pd.DataFrame(sens).round(4).to_string(index=False))

# ---- (c) incremental value for classification --------------------------------
data = exp.merge(idx[["subject"] + NAMES], on="subject")
assert "FRI_duration_included_legacy_sensitivity_only" not in data.columns and "FRI_no_timing_coupled_sensitivity" not in data.columns
y = data.target.to_numpy(); grp = data.subject.to_numpy()
static_cols = [c for c in exp.columns if c.endswith("_mean")]
sets = {"Static_means": static_cols, "Static_means+Resilience_indices": static_cols + COORD_INDICES,
        "Resilience_indices_only": COORD_INDICES}
def pipe(k):
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                     ("sel", SelectKBest(lambda a, b: mutual_info_classif(a, b, random_state=SEED), k=k)),
                     ("rf", RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                                   random_state=SEED, n_jobs=1))])
inc = []
for r in range(N_REPEATS):
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=SEED + r)
    for name, cols in sets.items():
        yp = np.zeros(len(y)); ps = np.zeros(len(y))
        for tr, te in cv.split(data, y, grp):
            mdl = pipe(min(20, len(cols))).fit(data.iloc[tr][cols], y[tr])
            yp[te] = mdl.predict(data.iloc[te][cols]); ps[te] = mdl.predict_proba(data.iloc[te][cols])[:, 1]
        inc.append({"repeat": r, "feature_set": name,
                    "accuracy": (yp == y).mean(), "balanced_accuracy": balanced_accuracy_score(y, yp),
                    "roc_auc": roc_auc_score(y, ps)})
I = pd.DataFrame(inc); I.to_csv(os.path.join(OUT, "resilience_incremental_repeats.csv"), index=False)
S = I.groupby("feature_set").agg(acc_mean=("accuracy", "mean"), acc_sd=("accuracy", "std"),
                                 bacc_mean=("balanced_accuracy", "mean"), auc_mean=("roc_auc", "mean"),
                                 auc_sd=("roc_auc", "std")).reset_index()
pv = I.pivot(index="repeat", columns="feature_set", values="balanced_accuracy")
diff = pv["Static_means+Resilience_indices"] - pv["Static_means"]
S["note"] = ""
S.loc[S.feature_set == "Static_means+Resilience_indices", "note"] = (
    f"paired diff in balanced acc vs static: {diff.mean():+.3f} (SD over repeats {diff.std(ddof=1):.3f}; repeats share the same 51 subjects)")
S.to_csv(os.path.join(OUT, "resilience_incremental_summary.csv"), index=False)

print("FUNCTIONAL RESILIENCE INDICES - TD vs DCP"); print(R.round(4).to_string(index=False))
print("\nINCREMENTAL VALUE (10 repeated subject-wise 5-fold partitions)"); print(S.round(4).to_string(index=False))
