# ============================================================
# STEP 33 - RECORDING-SITE / ACQUISITION CONFOUND INVESTIGATION
# ============================================================
# The supplied data contain NO site, session, camera or calibration variable, so site confounding cannot be
# removed or adjusted for. This step documents what CAN be established and measures how much group information is
# carried by acquisition-related proxies that are NOT clinical movement features. Nothing here claims to fix
# the confound; it quantifies and bounds it.
#
#  (1) What the documentation/metadata say (paper + data files) and what they do not.
#  (2) Filename / subject-number structure vs group (block table, ID-only classifier).
#  (3) Excel-file metadata (creator / dates) as batch proxy.
#  (4) Acquisition-geometry proxies from the raw key-point pixels (absolute camera-frame positions, apparent body
#      size in pixels, clip length): can they predict group? (subject-wise CV; NOT used as model predictors)
#  (5) Stratified check: does the primary nested-selection model do equally well in each subject-number block?

import os
import re
import sys
import warnings
import numpy as np
import pandas as pd
import openpyxl
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, "data", "raw")
OUT = os.path.join(BASE, "results", "confound_analysis")
os.makedirs(OUT, exist_ok=True)
SEED = 42
exp = pd.read_csv(os.path.join(BASE, "data", "processed", "experiment_dataset.csv"))
meta = exp[["subject", "group", "side", "target"]].copy()
meta["subject"] = meta["subject"].astype(int)

# ---- (1)+(2) filename structure ----
names = sorted(os.listdir(RAW))
pat = re.compile(r"^(\d+)_(left|right)_(TD|DCP)_rep([1-8])\.xlsx$")
bad = [n for n in names if not pat.match(n)]
print(f"(2) {len(names)} raw files; every filename matches Nr_arm_group_rep: {not bad}  "
      "=> filenames encode only case number, performing arm, group and repetition (no site/session token).")

bins = [(1, 24), (25, 42), (43, 66), (67, 89)]
rows = []
for lo, hi in bins:
    m = meta[(meta.subject >= lo) & (meta.subject <= hi)]
    rows.append({"subject_number_block": f"{lo}-{hi}", "n_TD": int((m.group == "TD").sum()), "n_DCP": int((m.group == "DCP").sum()),
                 "subject_numbers_TD": ",".join(map(str, sorted(m[m.group == "TD"].subject))),
                 "fraction_DCP": float((m.group == "DCP").mean()) if len(m) else np.nan})
blocks = pd.DataFrame(rows); blocks.to_csv(os.path.join(OUT, "subject_number_blocks.csv"), index=False)
print("\nSubject-number blocks vs group (case numbers are assigned in recruitment/processing order):")
print(blocks[["subject_number_block", "n_TD", "n_DCP", "fraction_DCP"]].to_string(index=False))

side_tab = pd.crosstab(meta.group, meta.side); side_tab.to_csv(os.path.join(OUT, "group_by_arm_side.csv"))
print("\nGroup x performing arm (from filenames):\n", side_tab.to_string())
print("Paper (Vanmechelen 2024, sect. 3.1) reports DCP 18 left / 12 right and TD 18 left / 3 right; "
      "TD agrees, the DCP split in the filenames is 20 left / 10 right (documentation discrepancy, unresolved).")

# ---- (3) Excel metadata ----
mrows = []
for f in names:
    if not f.endswith("rep1.xlsx"):
        continue
    p = openpyxl.load_workbook(os.path.join(RAW, f)).properties
    mrows.append({"subject": int(f.split("_")[0]), "creator": p.creator, "last_modified_by": p.lastModifiedBy,
                  "created": str(p.created)[:10], "modified": str(p.modified)[:10]})
M = pd.DataFrame(mrows).merge(meta, on="subject"); M.to_csv(os.path.join(OUT, "excel_file_metadata.csv"), index=False)
print("\n(3) Excel metadata: creator values", sorted(set(M.creator)), "| distinct created dates:", M.created.nunique(),
      "| files with a last-modified-by user:", int(M.last_modified_by.notna().sum()),
      f"(subjects {sorted(M[M.last_modified_by.notna()].subject)}) -> no site/camera information; at most a processing-history trace on 2 files.")

# ---- (4) acquisition-geometry proxies from raw pixels ----
prox = []
for f in names:
    if not f.endswith("_rep1.xlsx"):
        continue
    nr, side, grp = f.split("_")[0], f.split("_")[1], f.split("_")[2]
    reps = [pd.read_excel(os.path.join(RAW, f.replace("rep1", f"rep{r}"))) for r in range(1, 9)]
    g = lambda d, k, ax: pd.to_numeric(d[f"('{k}_{side}', '{ax}')"], errors="coerce").to_numpy()
    o = lambda d, ax: pd.to_numeric(d[f"('object', '{ax}')"], errors="coerce").to_numpy()
    feats = {"subject": int(nr)}
    d0 = reps[0]
    feats["hip_x_px"] = np.nanmean([g(d, "hip", "x")[0] for d in reps]); feats["hip_y_px"] = np.nanmean([g(d, "hip", "y")[0] for d in reps])
    feats["knee_x_px"] = np.nanmean([g(d, "knee", "x")[0] for d in reps]); feats["knee_y_px"] = np.nanmean([g(d, "knee", "y")[0] for d in reps])
    feats["shoulder_x_px"] = np.nanmean([g(d, "shoulder", "x")[0] for d in reps]); feats["shoulder_y_px"] = np.nanmean([g(d, "shoulder", "y")[0] for d in reps])
    feats["object_x_px"] = np.nanmean([o(d, "x")[0] for d in reps]); feats["object_y_px"] = np.nanmean([o(d, "y")[0] for d in reps])
    feats["torso_len_px"] = np.nanmean([np.hypot(g(d, "shoulder", "x")[0] - g(d, "hip", "x")[0], g(d, "shoulder", "y")[0] - g(d, "hip", "y")[0]) for d in reps])
    feats["thigh_len_px"] = np.nanmean([np.hypot(g(d, "hip", "x")[0] - g(d, "knee", "x")[0], g(d, "hip", "y")[0] - g(d, "knee", "y")[0]) for d in reps])
    feats["clip_frames_mean"] = float(np.mean([len(d) for d in reps]))
    prox.append(feats)
P = pd.DataFrame(prox).merge(meta, on="subject"); P.to_csv(os.path.join(OUT, "acquisition_geometry_proxies.csv"), index=False)
sets = {
    "camera_frame_positions (absolute pixel x,y of hip/knee/shoulder/target)": [c for c in P.columns if c.endswith("_px") and c.split("_")[0] in ("hip", "knee", "shoulder", "object")],
    "apparent_body_size (torso + thigh length in pixels)": ["torso_len_px", "thigh_len_px"],
    "clip_length (frames per repetition)": ["clip_frames_mean"],
    "subject_number_only": None,
}
yv, gv = P.target.to_numpy(), P.subject.to_numpy()
res = []
for label, cols in sets.items():
    Xp = P[["subject"]].to_numpy(float) if cols is None else P[cols].to_numpy(float)
    for mname, mk in [("logistic", lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))),
                      ("random_forest", lambda: make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=SEED)))]:
        accs = []
        for r in range(10):
            pred = np.zeros(len(yv), int)
            for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEED + r).split(Xp, yv, gv):
                pred[te] = mk().fit(Xp[tr], yv[tr]).predict(Xp[te])
            accs.append(balanced_accuracy_score(yv, pred))
        res.append({"proxy_set": label, "model": mname, "n_inputs": Xp.shape[1], "balanced_accuracy_mean": np.mean(accs),
                    "balanced_accuracy_sd_over_10_partitions": np.std(accs, ddof=1)})
R = pd.DataFrame(res); R.to_csv(os.path.join(OUT, "acquisition_proxy_classifiers.csv"), index=False)
print("\n(4) Group predictability from acquisition/body-size proxies ONLY (subject-wise 5-fold CV x10, chance = 0.50):")
print(R.round(3).to_string(index=False))

# ---- (5) primary nested-selection model by subject-number block ----
ns = pd.read_csv(os.path.join(BASE, "results", "nested_selection", "nested_selection_predictions.csv"))
ns["subject"] = ns.subject.astype(int); ns = ns.merge(meta[["subject", "group"]], on="subject")
ns["correct"] = (ns.true_target == ns.predicted_target).astype(int)
def blk(n):
    for lo, hi in bins:
        if lo <= n <= hi:
            return f"{lo}-{hi}"
ns["block"] = ns.subject.map(blk)
S = ns.groupby(["block", "group"]).agg(n=("correct", "size"), accuracy=("correct", "mean")).reset_index()
S.to_csv(os.path.join(OUT, "primary_model_accuracy_by_block_and_group.csv"), index=False)
print("\n(5) Primary nested-selection model, accuracy by subject-number block x group (small cells; descriptive only):")
print(S.round(3).to_string(index=False))

with open(os.path.join(OUT, "confound_summary.txt"), "w", encoding="utf-8") as f:
    f.write("RECORDING-SITE / ACQUISITION CONFOUND - WHAT IS AND IS NOT ESTABLISHED\n")
    f.write("- The dataset documentation (README, paper) states participants were recruited at laboratories in Belgium and the Netherlands and that "
            "each laboratory trained its own DeepLabCut model (federated processing). It does NOT list which child was recorded where, nor any "
            "camera/calibration/session variable. The statement 'all TD in Belgium, 6 DCP in Amsterdam' that appeared in earlier project text is NOT "
            "verifiable from the supplied documents and has been removed.\n")
    f.write(f"- Subject numbers are strongly structured by group (blocks 25-42 and 67-89 contain only DCP; TD occupy 1-2 and 43-66); a classifier "
            f"given only the subject number reaches balanced accuracy {R[(R.proxy_set=='subject_number_only')].balanced_accuracy_mean.max():.2f}. "
            "Case numbers are therefore a proxy for recruitment/processing batch and must never be a predictor.\n")
    f.write("- Best balanced accuracy of each acquisition proxy (max over classifiers; 10 partitions): " + "; ".join(
        f"{k.split(' (')[0]} {v:.2f}" for k, v in R.groupby("proxy_set").balanced_accuracy_mean.max().items()) + ". "
        "Clip length (frames per repetition) is the strongest non-ID proxy; movement_duration_s equals clip length (step 37), so clinical slowness and "
        "recording/trimming differences cannot be separated. Duration-excluded ablations are in results/feature_sets.\n")
    f.write("- Excel metadata and filenames contain no site information.\n")
    f.write("- Group is also unbalanced on arm side (TD 18 left / 3 right) and, per the paper, sex (DCP 10 F / 20 M; TD 16 F / 5 M). "
            "Left-arm videos were mirrored to look like right-arm videos.\n")
    f.write("- CONCLUSION: group and acquisition batch cannot be separated with the supplied data. High accuracy may partly reflect recording "
            "setup. Validation on an independent, site-balanced cohort is required before any clinical claim.\n")
print("\nSaved", os.path.join(OUT, "confound_summary.txt"))
