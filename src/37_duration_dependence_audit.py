# ============================================================
# STEP 37 - RECORDING-DURATION DEPENDENCE AUDIT
# ============================================================
# movement_duration_s = (n_frames - 1) / 25 fps, so it equals recording clip length. This step (a) confirms that identity
# on the repetition-level data, and (b) reports, for every primary predictor, its subject-level Spearman correlation with
# mean clip length and the group difference in clip length. It does NOT decide whether duration is clinical or an artefact:
# with the supplied metadata that cannot be separated; the duration-excluded ablations (steps 23/29/35) quantify the reliance.
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rep = pd.read_csv(os.path.join(BASE, "data", "processed", "rep_features.csv"))
ds = pd.read_csv(os.path.join(BASE, "data", "processed", "experiment_dataset.csv"))
out = os.path.join(BASE, "results", "duration_audit"); os.makedirs(out, exist_ok=True)

exact = np.allclose(rep["movement_duration_s"], (rep["n_frames"] - 1) / 25.0, equal_nan=True)
r_rep = float(np.corrcoef(rep["movement_duration_s"].dropna(), rep.loc[rep["movement_duration_s"].notna(), "n_frames"])[0, 1])
clip = rep.groupby("subject")["n_frames"].mean().rename("clip_frames_mean")
S = ds[["subject", "target"]].merge(clip.reset_index(), on="subject")
assert len(S) == 51
dur_mean = ds.set_index("subject")["movement_duration_s_mean"]
r_subj = float(np.corrcoef(S.set_index("subject").loc[dur_mean.index, "clip_frames_mean"], dur_mean)[0, 1])
g1, g0 = S[S.target == 1].clip_frames_mean, S[S.target == 0].clip_frames_mean
u = mannwhitneyu(g1, g0)
rows = []
for c in [c for c in ds.columns if c not in ("subject", "side", "group", "target")]:
    x = ds.set_index("subject").loc[S.subject, c]; ok = x.notna().to_numpy()
    rho = spearmanr(x[ok], S.clip_frames_mean[ok]).statistic
    rows.append({"predictor": c, "spearman_with_clip_length": rho, "abs_rho": abs(rho)})
T = pd.DataFrame(rows).sort_values("abs_rho", ascending=False); T.to_csv(os.path.join(out, "predictor_vs_clip_length.csv"), index=False)
txt = [f"movement_duration_s == (n_frames-1)/25 on all repetitions: {exact}; repetition-level Pearson r = {r_rep:.4f}; subject-level mean-duration vs mean-clip-length r = {r_subj:.4f}",
       f"Mean clip length (frames): DCP {g1.mean():.1f} (SD {g1.std():.1f}), TD {g0.mean():.1f} (SD {g0.std():.1f}); Mann-Whitney p = {u.pvalue:.2g}",
       f"Primary predictors with |Spearman| >= 0.7 vs clip length: {(T.abs_rho >= 0.7).sum()} of {len(T)}",
       "Top 12 by |rho|:"] + [f"  {r.predictor}: {r.spearman_with_clip_length:+.2f}" for r in T.head(12).itertuples()] + [
       "Interpretation: duration is a legitimate clinical timing measure OR a recording/trimming artefact; the supplied metadata cannot separate the two.",
       "Many other predictors are correlated with duration, so excluding movement_duration_s alone does not remove all timing information."]
open(os.path.join(out, "duration_audit_summary.txt"), "w", encoding="utf-8").write("\n".join(txt))
print("\n".join(txt))
