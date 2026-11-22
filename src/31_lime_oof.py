# ============================================================
# STEP 31 - OUT-OF-FOLD LIME EXPLANATIONS (project objective 7: SHAP *and* LIME)
# ============================================================
# Same subject-wise outer 5-fold CV as the SHAP step. In every outer fold a Random-Forest
# pipeline (imputer -> scaler -> mutual-information SelectKBest -> RF) is tuned by inner CV on the
# outer-TRAINING subjects only, then LIME explains the predictions for the held-out subjects.
# The LIME background distribution is the outer-training data only (never the test subject).
# LIME is local and stochastic, so it is used to CROSS-CHECK the SHAP ranking, not to replace it:
# agreement is quantified (Spearman rank correlation, top-15 overlap).

import os
import warnings
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "results", "lime_oof")
FIG = os.path.join(BASE, "results", "figures")
os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)
sys.path.insert(0, os.path.join(BASE, "src"))
import model_catalog as mc
SEED, TOP, NUM_SAMPLES = 42, 15, 1000

df = pd.read_csv(os.path.join(BASE, "data", "processed", "experiment_dataset.csv"))
y = df["target"]; groups = df["subject"]
X = df.drop(columns=["target", "subject", "side", "group"], errors="ignore").select_dtypes(include=[np.number])
feats = X.columns.tolist()

outer, inner = mc.make_cv(SEED)

records, fold_rows = [], []
for fold, (tr, te) in enumerate(outer.split(X, y, groups), start=1):
    gs = mc.fit_nested("Random_Forest", X.iloc[tr], y.iloc[tr], groups.iloc[tr], inner)   # catalog definition
    best = gs.best_estimator_
    imp = best.named_steps["imputer"]
    Xtr = imp.transform(X.iloc[tr]); Xte = imp.transform(X.iloc[te])
    rest = Pipeline(best.steps[1:])                       # fitted scaler -> selector -> model

    def predict_fn(Z):
        return rest.predict_proba(pd.DataFrame(Z, columns=feats))

    explainer = LimeTabularExplainer(Xtr, feature_names=feats, class_names=["TD", "DCP"],
                                     discretize_continuous=True, mode="classification",
                                     random_state=SEED)
    acc = (best.predict(X.iloc[te]) == y.iloc[te].to_numpy()).mean()
    fold_rows.append({"fold": fold, "best_params": str(gs.best_params_), "outer_accuracy": acc})
    print(f"fold {fold}: {gs.best_params_}  outer acc {acc:.3f}")
    for i, row in enumerate(Xte):
        exp = explainer.explain_instance(row, predict_fn, num_features=len(feats), labels=(1,),
                                         num_samples=NUM_SAMPLES)
        w = dict(exp.as_map()[1])
        for j, f in enumerate(feats):
            records.append({"fold": fold, "subject": groups.iloc[te].iloc[i], "feature": f,
                            "lime_weight": w.get(j, 0.0)})

L = pd.DataFrame(records)
L.to_csv(os.path.join(OUT, "oof_lime_weights.csv"), index=False)
pd.DataFrame(fold_rows).to_csv(os.path.join(OUT, "lime_fold_summary.csv"), index=False)
S = (L.assign(abs_w=L.lime_weight.abs()).groupby("feature")
       .agg(mean_abs_lime=("abs_w", "mean"), mean_lime=("lime_weight", "mean")).reset_index()
       .sort_values("mean_abs_lime", ascending=False).reset_index(drop=True))
shap = pd.read_csv(os.path.join(BASE, "results", "shap_oof", "oof_shap_summary.csv"))[["feature", "mean_abs_shap"]]
M = S.merge(shap, on="feature", how="left").fillna({"mean_abs_shap": 0.0})
M["lime_rank"] = M.mean_abs_lime.rank(ascending=False, method="first")
M["shap_rank"] = M.mean_abs_shap.rank(ascending=False, method="first")
M.to_csv(os.path.join(OUT, "oof_lime_summary.csv"), index=False)

rho = spearmanr(M.mean_abs_lime, M.mean_abs_shap)
top_l = set(M.nsmallest(TOP, "lime_rank").feature); top_s = set(M.nsmallest(TOP, "shap_rank").feature)
agree = pd.DataFrame([{"spearman_rho_all_features": rho.statistic, "spearman_p": rho.pvalue,
                       "top15_overlap": len(top_l & top_s), "top15_jaccard": len(top_l & top_s) / len(top_l | top_s),
                       "n_explained_subjects": L.subject.nunique(), "lime_num_samples": NUM_SAMPLES}])
agree.to_csv(os.path.join(OUT, "lime_shap_agreement.csv"), index=False)

top = M.nsmallest(TOP, "lime_rank").iloc[::-1]
fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(top.feature, top.mean_abs_lime)
ax.set_xlabel("Mean |LIME weight| over out-of-fold subjects"); ax.set_title("Out-of-fold LIME feature importance")
ax.grid(axis="x", alpha=0.3); fig.tight_layout()
fig.savefig(os.path.join(FIG, "Figure_10_lime_importance.png"), bbox_inches="tight", dpi=200); plt.close(fig)

print(M.head(TOP)[["feature", "mean_abs_lime", "mean_abs_shap", "lime_rank", "shap_rank"]].to_string(index=False))
print(agree.T.to_string(header=False))
