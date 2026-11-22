# ============================================================
# STEP 41 - FRI VERIFICATION AND COMPONENT ANALYSIS (checklist 1)
# ============================================================
# Independently recomputes the FRI from the repetition-level data, asserts equality with step 30, then reports for each
# quality metric: consistency / maintenance contribution, single-metric AUC, leave-one-out FRI AUC, and the FRI built only
# from non-timing metrics. Direction fixed a priori (higher = TD). Descriptive; not clinically validated.
import os, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "results", "functional_resilience")
long = pd.read_csv(os.path.join(BASE, "data", "processed", "repetition_trajectories_long.csv"))
saved = pd.read_csv(os.path.join(OUT, "resilience_indices_by_subject.csv"))
key = lambda s: str(int(s)).zfill(2)
saved["k"] = saved.subject.map(key); saved = saved.set_index("k")
Q = {"path_efficiency": 1, "position_error_mm": -1, "max_lateral_deviation_mm": -1, "smoothness_sparc": 1, "smoothness_norm_jerk": -1}
def cvt(v):
    v = v[np.isfinite(v)]
    return np.nan if len(v) < 3 or abs(v.mean()) < 1e-12 else 1 / (1 + v.std(ddof=0) / abs(v.mean()))
cs, ms, grp, ln = {}, {}, {}, {}
for s, g in long.sort_values(["subject", "rep"]).groupby("subject"):
    k = key(s); grp[k] = int(g.group.iloc[0] == "DCP"); ln[k] = float(g["movement_duration_s"].mean())
    for m, sg in Q.items():
        v = g[m].to_numpy(float); ok = np.isfinite(v); e, l = v[:4][ok[:4]], v[4:][ok[4:]]; sc = np.abs(v[ok]).mean() if ok.any() else np.nan
        cs[(k, m)] = cvt(v)
        ms[(k, m)] = np.exp(min(0, sg * (l.mean() - e.mean()) / sc)) if len(e) and len(l) and sc > 1e-12 else np.nan
subs = sorted(grp); y = np.array([grp[k] for k in subs])
def fri(metrics):
    c = np.array([np.nanmean([cs[(k, m)] for m in metrics]) for k in subs]); r = np.array([np.nanmean([ms[(k, m)] for m in metrics]) for k in subs])
    return 0.5 * c + 0.5 * r, c, r
full, c_all, r_all = fri(list(Q))
assert np.allclose(full, saved.loc[subs, "Functional_Resilience_Index"].to_numpy(), atol=1e-12), "FRI recomputation differs from step 30"
print("FRI recomputation matches step 30 for all", len(subs), "subjects")
def auc(sc):
    a, b = sc[y == 0], sc[y == 1]; a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    return (a[:, None] > b[None, :]).mean() + .5 * (a[:, None] == b[None, :]).mean(), mannwhitneyu(a, b).pvalue, \
        (b.mean() - a.mean()) / np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
def sp(x):
    ok = np.isfinite(x); return spearmanr(x[ok], np.array([ln[k] for k in subs])[ok])[0]
rows = []
def add(kind, name, sc):
    a, p, d = auc(sc); rows.append({"analysis": kind, "name": name, "AUC_TD_gt_DCP": a, "mannwhitney_p": p, "cohens_d_DCP_minus_TD": d, "spearman_with_clip_length": sp(sc)})
add("full", "FRI (5 metrics)", full)
for m in Q:
    c = np.array([cs[(k, m)] for k in subs]); r = np.array([ms[(k, m)] for k in subs])
    add("single metric consistency", m, c); add("single metric maintenance", m, r); add("single metric FRI", m, 0.5 * c + 0.5 * r)
    add("leave one out", f"FRI without {m}", fri([x for x in Q if x != m])[0])
add("timing removed", "FRI without sparc+norm_jerk", fri(["path_efficiency", "position_error_mm", "max_lateral_deviation_mm"])[0])
add("component", "Consistency only", c_all); add("component", "Maintenance only", r_all)
R = pd.DataFrame(rows); p = R.mannwhitney_p.to_numpy(); o = np.argsort(p); m = len(p)
adj = np.minimum.accumulate((p[o] * m / (np.arange(m) + 1))[::-1])[::-1]; R.loc[o, "p_fdr_bh"] = np.minimum(adj, 1)
R.to_csv(os.path.join(OUT, "fri_component_analysis.csv"), index=False); print(R.round(3).to_string(index=False))
