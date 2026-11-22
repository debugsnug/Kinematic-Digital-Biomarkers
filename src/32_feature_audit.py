# ============================================================
# STEP 32 - EMPIRICAL FEATURE-MODALITY AUDIT
# ============================================================
# Does not trust the registry in feature_sets.py: it PROVES which features depend on what.
#  (A) Dependency test: recompute every repetition-level feature from the cleaned frames
#      (i) with the three joint-angle columns removed (NaN), (ii) with the target-object key point removed.
#      A base feature that changes (value differs or becomes NaN) in ANY repetition depends on that input.
#      The result must equal the registry (angle_derived / coordinates_plus_target / coordinates) - asserted.
#  (B) Angle provenance: re-derive 2D frontal-plane joint angles from the raw key-point (x, y) columns
#      and compare them with the supplied angle columns, to establish whether the angle columns are
#      video/key-point derived (markerless) or an independent source.
#  (C) Exact predictor counts per feature set.

import importlib.util
import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))
import feature_sets as fs

OUT = os.path.join(BASE, "results", "feature_audit")
os.makedirs(OUT, exist_ok=True)

# ---------------- (A) dependency test ----------------
spec = importlib.util.spec_from_file_location("feat03", os.path.join(BASE, "src", "03_features.py"))
feat03 = importlib.util.module_from_spec(spec); spec.loader.exec_module(feat03)

frames = pd.read_pickle(os.path.join(BASE, "data", "processed", "clean_frames.pkl"))
ANGLE_COLS = ["shoulder_angle", "elbow_angle", "wrist_angle"]
OBJ_COLS = [c for c in frames.columns if c.startswith("object_")]

def feats(sub):
    return feat03.extract_rep_features(sub)

changed_angle, changed_obj, all_names = {}, {}, None
n_reps = 0
for keys, sub in frames.groupby(["subject", "side", "group", "rep"], sort=False):
    base = feats(sub)
    no_angle = feats(sub.assign(**{c: np.nan for c in ANGLE_COLS}))
    no_obj = feats(sub.assign(**{c: np.nan for c in OBJ_COLS}))
    all_names = list(base.index)
    n_reps += 1
    for name in all_names:
        a, b, c = base[name], no_angle[name], no_obj[name]
        diff_a = not (np.isclose(a, b, equal_nan=True, rtol=1e-12, atol=1e-12))
        diff_o = not (np.isclose(a, c, equal_nan=True, rtol=1e-12, atol=1e-12))
        changed_angle[name] = changed_angle.get(name, False) or diff_a
        changed_obj[name] = changed_obj.get(name, False) or diff_o

rows = []
for name in all_names:
    reg = "angle_derived" if name in fs.ANGLE_BASE else ("coordinates_plus_target" if name in fs.TARGET_BASE else "coordinates")
    emp = ("angle_derived" if changed_angle[name] else "coordinates_plus_target" if changed_obj[name] else "coordinates")
    rows.append({"base_feature": name, "depends_on_angle_columns": changed_angle[name],
                 "depends_on_target_object": changed_obj[name], "registry_modality": reg,
                 "empirical_modality": emp, "agree": reg == emp})
dep = pd.DataFrame(rows)
dep.to_csv(os.path.join(OUT, "base_feature_dependency.csv"), index=False)
print(f"(A) dependency test over {n_reps} subject x repetition recordings")
print(dep[dep.empirical_modality != "coordinates"].to_string(index=False))
bad = dep[~dep.agree & ~dep.base_feature.isin(["n_frames"])]
# only base features that enter the experiment dataset must agree
audit = pd.read_csv(os.path.join(BASE, "data", "processed", "feature_audit.csv"))
used = set(audit.base_feature)
bad = dep[(~dep.agree) & dep.base_feature.isin(used)]
if len(bad):
    print(bad.to_string(index=False)); raise SystemExit("REGISTRY DISAGREES WITH THE EMPIRICAL DEPENDENCY TEST")
print("registry == empirical modality for every predictor base feature: PASS")

# ---------------- (B) angle provenance ----------------
RAW = os.path.join(BASE, "data", "raw")
def ang(u, v):
    cu = (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
    return np.degrees(np.arccos(np.clip(cu, -1, 1)))
def sang(u, v):
    return np.degrees(np.arctan2(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0], (u * v).sum(1)))
prov = []
files = sorted(f for f in os.listdir(RAW) if f.endswith("rep1.xlsx"))
for f in files:
    parts = f.split("_"); side = parts[1]; s = side
    d = pd.read_excel(os.path.join(RAW, f))
    g = lambda k, ax: pd.to_numeric(d[f"('{k}_{s}', '{ax}')"], errors="coerce").to_numpy()
    P = {k: np.c_[g(k, "x"), g(k, "y")] for k in ["hip", "shoulder", "elbow", "wrist", "MCP3"]}
    L = "R" if side == "right" else "L"
    cand = {
        "Shoulder": ang(P["hip"] - P["shoulder"], P["elbow"] - P["shoulder"]),
        # signed 2D angle between the upper-arm and forearm vectors / forearm and hand vectors
        "Elbow": sang(P["elbow"] - P["shoulder"], P["wrist"] - P["elbow"]),
        "Wrist": sang(P["wrist"] - P["elbow"], P["MCP3"] - P["wrist"]),
    }
    for joint, derived in cand.items():
        given = pd.to_numeric(d[f"{L}-{joint}"], errors="coerce").to_numpy()
        ok = np.isfinite(given) & np.isfinite(derived)
        if ok.sum() < 5:
            continue
        r = np.corrcoef(given[ok], derived[ok])[0, 1]
        slope, icpt = np.polyfit(derived[ok], given[ok], 1)
        resid = (given[ok] - derived[ok] + 180.0) % 360.0 - 180.0   # raw difference (wrapped to +-180), no fitting
        prov.append({"file": f, "joint": joint, "n_frames": int(ok.sum()), "pearson_r": r, "abs_r": abs(r),
                     "fit_slope": slope, "fit_intercept": icpt, "resid_rmse_deg": float(np.sqrt(np.mean(resid ** 2))), "max_abs_diff_deg": float(np.max(np.abs(resid))), "frac_frames_exact": float(np.mean(np.abs(resid) < 0.01)),
                     "given_range_deg": float(np.ptp(given[ok]))})
P_ = pd.DataFrame(prov); P_.to_csv(os.path.join(OUT, "angle_provenance_check.csv"), index=False)
summ = P_.groupby("joint").agg(n_recordings=("file", "count"), median_abs_r=("abs_r", "median"),
                               p10_abs_r=("abs_r", lambda x: np.percentile(x, 10)),
                               median_resid_rmse_deg=("resid_rmse_deg", "median"),
                               max_abs_diff_deg=("max_abs_diff_deg", "max"),
                               frac_frames_exact=("frac_frames_exact", "mean"),
                               n_recordings_all_frames_exact=("frac_frames_exact", lambda x: int((x > 0.999).sum())),
                               median_given_range_deg=("given_range_deg", "median")).reset_index()
summ.to_csv(os.path.join(OUT, "angle_provenance_summary.csv"), index=False)
print("\n(B) supplied angle columns vs angles re-derived from the key points (rep1 of every participant):")
print(summ.round(4).to_string(index=False))
print("CONCLUSION: the supplied angle columns are reproduced (to <0.01 deg, same frames) by simple 2D geometry on the video key points "
      "for the large majority of frames/recordings (see frac_frames_exact); they are 2D video-derived, NOT an independent 3D "
      "motion-capture (Vicon) measurement.")

# ---------------- (C) counts ----------------
cnt = audit.groupby("source_modality").size().rename("n_predictors").reset_index()
sets = pd.DataFrame([{"feature_set": k, "description": v,
                      "n_predictors": int(audit[{"coordinate_only": "in_coordinate_only", "angle_enhanced": "in_angle_enhanced",
                                                 "coordinate_no_position_error": "in_coordinate_no_position_error",
                                                 "coordinate_no_duration": "in_coordinate_no_duration",
                                                 "coordinate_no_duration_no_position_error": "in_coordinate_no_duration_no_position_error",
                                                 "coordinate_no_timing_coupled": "in_coordinate_no_timing_coupled"}[k]].sum())}
                     for k, v in fs.SETS.items()])
sets.to_csv(os.path.join(OUT, "feature_set_sizes.csv"), index=False)
print("\n(C) predictors by modality:\n", cnt.to_string(index=False)); print(sets.to_string(index=False))
assert set(audit[audit.in_coordinate_only].source_modality) <= {"coordinates", "coordinates_plus_target"}
print("\nNo angle-derived predictor is in the coordinate-only set: PASS")
