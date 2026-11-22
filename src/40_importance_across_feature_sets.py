# ============================================================
# STEP 40 - FEATURE IMPORTANCE ACROSS FEATURE CONFIGURATIONS (checklist 3 / 5)
# ============================================================
# For each of the 6 feature sets: out-of-fold SHAP of the catalog Random Forest (same pipeline as step 20; per-fold
# k and RF hyper-parameters taken from that set's saved nested-selection results is NOT used - a fixed RF
# (k=40, catalog defaults) is explained so that sets are comparable). SHAP is computed on held-out subjects only,
# aggregated to base features (sum of mean|SHAP| over the 6 descriptor columns). EXPLORATORY: explains this RF.
import os, sys, warnings
import numpy as np, pandas as pd, shap
from itertools import combinations
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))
import model_catalog as mc, feature_sets as fs
OUT = os.path.join(BASE, "results", "importance_across_feature_sets"); os.makedirs(OUT, exist_ok=True)
FILES = {"coordinate_only": "experiment_dataset.csv", "angle_enhanced": "experiment_dataset_angle_enhanced.csv",
         "coordinate_no_position_error": "experiment_dataset_coord_no_position_error.csv",
         "coordinate_no_duration": "experiment_dataset_coord_no_duration.csv",
         "coordinate_no_duration_no_position_error": "experiment_dataset_coord_no_duration_no_position_error.csv",
         "coordinate_no_timing_coupled": "experiment_dataset_coord_no_timing_coupled.csv"}
ANGLE = set(fs.ANGLE_BASE)
imp_rows, mod_rows = [], []
for name, fn in FILES.items():
    df = pd.read_csv(os.path.join(BASE, "data", "processed", fn))
    y = df.target.to_numpy(); g = df.subject.to_numpy()
    Xd = df.drop(columns=["target", "subject", "side", "group"], errors="ignore").select_dtypes(include=[np.number])
    cols = np.array(Xd.columns); X = Xd.to_numpy(float)
    outer, _ = mc.make_cv(); acc = {}
    for fold, (tr, te) in enumerate(outer.split(X, y, g), start=1):
        pipe = mc.make_pipeline("Random_Forest"); pipe.set_params(selector__k=40); pipe.fit(X[tr], y[tr])
        keep = pipe.named_steps["selector"].get_support()
        Xte = pipe.named_steps["scaler"].transform(pipe.named_steps["imputer"].transform(X[te]))[:, keep]
        sv = shap.TreeExplainer(pipe.named_steps["model"]).shap_values(Xte)
        sv = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if sv.ndim == 3 else sv)
        for c, v in zip(cols[keep], np.abs(sv).mean(axis=0)):
            b = fs.split_name(c)[0]; acc[b] = acc.get(b, 0.0) + v / 5
    s = pd.Series(acc).sort_values(ascending=False)
    for r, (b, v) in enumerate(s.items(), 1):
        imp_rows.append({"feature_set": name, "base_feature": b, "mean_abs_shap_sum": v, "rank": r})
    tot = s.sum()
    mod = {"angle": sum(v for b, v in s.items() if b in ANGLE), "target(position_error)": s.get("position_error_mm", 0.0),
           "timing_coupled": sum(v for b, v in s.items() if b in fs.TIMING_COUPLED_BASE)}
    mod["other_coordinate"] = tot - sum(mod.values())
    for k, v in mod.items(): mod_rows.append({"feature_set": name, "modality": k, "share": v / tot})
I = pd.DataFrame(imp_rows); I.to_csv(os.path.join(OUT, "importance_by_set.csv"), index=False)
pd.DataFrame(mod_rows).to_csv(os.path.join(OUT, "modality_share_by_set.csv"), index=False)
ag = []
for a, b in combinations(FILES, 2):
    A = I[I.feature_set == a].set_index("base_feature").mean_abs_shap_sum; B = I[I.feature_set == b].set_index("base_feature").mean_abs_shap_sum
    com = A.index.intersection(B.index)
    ag.append({"set_a": a, "set_b": b, "n_common_features": len(com), "spearman": spearmanr(A[com], B[com])[0],
               "top5_overlap": len(set(A.nlargest(5).index) & set(B.nlargest(5).index))})
pd.DataFrame(ag).to_csv(os.path.join(OUT, "rank_agreement_between_sets.csv"), index=False)
with open(os.path.join(OUT, "summary.txt"), "w") as f:
    f.write("Top-5 base features by out-of-fold SHAP (exploratory RF, k=40) per feature set\n")
    for n in FILES: f.write(f"\n{n}: " + ", ".join(I[(I.feature_set == n) & (I["rank"] <= 5)].base_feature) + "\n")
    f.write("\nRank agreement between sets:\n" + pd.DataFrame(ag).round(3).to_string(index=False) + "\n")
print(open(os.path.join(OUT, "summary.txt")).read())
