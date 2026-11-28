# ============================================================
# STEP 47 — DEDICATED SHAP + LIME FOR THE HIGHEST-PERFORMING MODEL ONLY
# ============================================================
# Added as an extra script (without removing step 45) per guide feedback:
#   "use LIME and SHAP only on highest performance model individually not for all 5"
#
# This script:
#   1. Selects the single highest-performing model from results/model_benchmark/
#      model_benchmark_summary.csv (by inner_accuracy_mean -> MLP).
#   2. Runs leak-free nested CV (outer 5-fold StratifiedGroupKFold, inner 4-fold
#      GridSearchCV via model_catalog.fit_nested) for that single model only.
#   3. Computes out-of-fold SHAP (KernelExplainer on outer-training background)
#      and out-of-fold LIME (LimeTabularExplainer on outer-training background)
#      for every held-out subject.
#   4. Quantifies SHAP vs LIME agreement (Spearman rho, top-15 overlap).
#   5. Saves global bar/beeswarm figures AND individual-subject SHAP/LIME
#      explanation figures for representative subjects.
#
# Outputs in results/explainability_best_model/:
#   best_model_info.csv
#   oof_shap_values.csv
#   oof_shap_summary.csv
#   oof_lime_weights.csv
#   oof_lime_summary.csv
#   shap_lime_agreement.csv
#   figures/shap_bar_<Model>.png
#   figures/shap_beeswarm_<Model>.png
#   figures/lime_bar_<Model>.png
#   figures/individual_shap_subject_<id>.png
#   figures/individual_lime_subject_<id>.png

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import spearmanr
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
import model_catalog as mc

OUT_DIR = os.path.join(BASE_DIR, "results", "explainability_best_model")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

SEED = 42
TOP_FEATURES = 15
LIME_SAMPLES = 1000

print("=" * 70)
print("STEP 47 — SHAP + LIME FOR HIGHEST-PERFORMING MODEL ONLY")
print("=" * 70)

# ----------------------------------------------------------------
# 1. Identify single highest-performing model from benchmark
# ----------------------------------------------------------------
summary_path = os.path.join(BASE_DIR, "results", "model_benchmark", "model_benchmark_summary.csv")
if not os.path.exists(summary_path):
    raise FileNotFoundError(f"Run step 23 first: {summary_path}")

bm = pd.read_csv(summary_path).sort_values("inner_accuracy_mean", ascending=False).reset_index(drop=True)
best_row = bm.iloc[0]
best_name = str(best_row["model"])
best_row.to_frame().T.to_csv(os.path.join(OUT_DIR, "best_model_info.csv"), index=False)

print(f"Highest-performing model (by inner-CV accuracy): {best_name}")
print(f"  Inner CV Acc : {best_row['inner_accuracy_mean']:.4f}")
print(f"  Outer Test Acc: {best_row['test_accuracy_mean']:.4f}")
print(f"  Balanced Acc : {best_row['test_balanced_accuracy_mean']:.4f}")
print(f"  F1-Score     : {best_row['test_f1_mean']:.4f}")
print(f"  ROC-AUC      : {best_row['test_roc_auc_mean']:.4f}")

# ----------------------------------------------------------------
# 2. Load preprocessed dataset
# ----------------------------------------------------------------
DATA_PATH = os.path.join(
    BASE_DIR, "data", "processed",
    "final_preprocessed_features_for_model_training_and_prediction.csv"
)
df = pd.read_csv(DATA_PATH)
y = df["target"]
groups = df["subject"]
X = df.drop(columns=["target", "subject", "side", "group"], errors="ignore").select_dtypes(include=[np.number])
feature_names = X.columns.tolist()
print(f"\nDataset: {df.shape} | subjects={groups.nunique()} | predictors={len(feature_names)}")

outer_cv, inner_cv = mc.make_cv(SEED)

# ----------------------------------------------------------------
# 3. Outer CV: tune best_name on training subjects, explain held-out subjects
# ----------------------------------------------------------------
shap_records = []
lime_records = []
pred_records = []

for fold_idx, (tr, te) in enumerate(outer_cv.split(X, y, groups), start=1):
    X_tr, X_te = X.iloc[tr], X.iloc[te]
    y_tr, y_te = y.iloc[tr], y.iloc[te]
    g_tr, g_te = groups.iloc[tr], groups.iloc[te]

    gs = mc.fit_nested(best_name, X_tr, y_tr, g_tr, inner_cv)
    best_pipe = gs.best_estimator_

    imp = best_pipe.named_steps["imputer"]
    sc = best_pipe.named_steps["scaler"]
    sel = best_pipe.named_steps["selector"]
    model = best_pipe.named_steps["model"]

    # Transform to selected feature space (fitted on training subjects only)
    X_tr_sel = sel.transform(sc.transform(imp.transform(X_tr)))
    X_te_sel = sel.transform(sc.transform(imp.transform(X_te)))
    sel_feats = [feature_names[i] for i, s in enumerate(sel.get_support()) if s]

    y_pred = best_pipe.predict(X_te)
    scores, _ = mc.score_of(best_pipe, X_te)
    fold_acc = float((y_pred == y_te.to_numpy()).mean())
    print(f"\nFold {fold_idx}/5: best_params={gs.best_params_} | n_sel={len(sel_feats)} | outer_acc={fold_acc:.3f}")

    for s, yt, yp, sc_val in zip(g_te.tolist(), y_te.tolist(), y_pred.tolist(), scores.tolist()):
        pred_records.append({
            "fold": fold_idx, "subject": s, "true_target": yt,
            "predicted_target": yp, "predicted_score": sc_val,
        })

    # --- SHAP (outer-training background only) ---
    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_te_sel)
    except Exception:
        bg = shap.sample(X_tr_sel, min(50, len(X_tr_sel)), random_state=SEED)
        predict_shap = (
            (lambda Z: model.predict_proba(Z)[:, 1])
            if hasattr(model, "predict_proba")
            else (lambda Z: model.decision_function(Z))
        )
        explainer = shap.KernelExplainer(predict_shap, bg)
        sv = explainer.shap_values(X_te_sel, nsamples=150)

    sv_arr = np.asarray(sv)
    if isinstance(sv, list) and len(sv) >= 2:
        sv_pos = np.asarray(sv[1])
    elif sv_arr.ndim == 3:
        sv_pos = sv_arr[:, :, 1]
    else:
        sv_pos = sv_arr

    for i, subj in enumerate(g_te.tolist()):
        for j, feat in enumerate(sel_feats):
            shap_records.append({
                "model": best_name, "fold": fold_idx, "subject": subj,
                "true_target": int(y_te.iloc[i]),
                "predicted_target": int(y_pred[i]),
                "feature": feat,
                "feature_value": float(X_te_sel[i, j]),
                "shap_value": float(sv_pos[i, j]),
                "abs_shap": abs(float(sv_pos[i, j])),
            })

    # --- LIME (outer-training background only) ---
    X_tr_imp = imp.transform(X_tr)
    X_te_imp = imp.transform(X_te)
    rest_pipe = Pipeline(best_pipe.steps[1:])  # scaler -> selector -> model

    def lime_predict_fn(Z):
        df_z = pd.DataFrame(Z, columns=feature_names)
        if hasattr(rest_pipe, "predict_proba"):
            return rest_pipe.predict_proba(df_z)
        preds = rest_pipe.predict(df_z)
        return np.column_stack([1 - preds, preds])

    lime_exp = LimeTabularExplainer(
        X_tr_imp, feature_names=feature_names,
        class_names=["TD", "DCP"],
        discretize_continuous=True, mode="classification",
        random_state=SEED,
    )

    for i, row in enumerate(X_te_imp):
        exp = lime_exp.explain_instance(
            row, lime_predict_fn, num_features=len(feature_names),
            labels=(1,), num_samples=LIME_SAMPLES,
        )
        w = dict(exp.as_map()[1])
        for j, feat in enumerate(feature_names):
            lime_records.append({
                "model": best_name, "fold": fold_idx,
                "subject": g_te.iloc[i],
                "true_target": int(y_te.iloc[i]),
                "predicted_target": int(y_pred[i]),
                "feature": feat,
                "lime_weight": float(w.get(j, 0.0)),
            })

# ----------------------------------------------------------------
# 4. Save CSVs and compute SHAP vs LIME agreement
# ----------------------------------------------------------------
shap_df = pd.DataFrame(shap_records)
lime_df = pd.DataFrame(lime_records)
pred_df = pd.DataFrame(pred_records)

shap_df.to_csv(os.path.join(OUT_DIR, "oof_shap_values.csv"), index=False)
lime_df.to_csv(os.path.join(OUT_DIR, "oof_lime_weights.csv"), index=False)
pred_df.to_csv(os.path.join(OUT_DIR, "oof_predictions.csv"), index=False)

shap_summary = (
    shap_df.groupby("feature")
    .agg(mean_abs_shap=("abs_shap", "mean"),
         mean_shap=("shap_value", "mean"),
         std_abs_shap=("abs_shap", "std"),
         n_folds=("fold", "nunique"))
    .reset_index()
    .sort_values("mean_abs_shap", ascending=False)
    .reset_index(drop=True)
)
shap_summary.to_csv(os.path.join(OUT_DIR, "oof_shap_summary.csv"), index=False)

lime_summary = (
    lime_df.assign(abs_w=lime_df["lime_weight"].abs())
    .groupby("feature")
    .agg(mean_abs_lime=("abs_w", "mean"),
         mean_lime=("lime_weight", "mean"))
    .reset_index()
    .sort_values("mean_abs_lime", ascending=False)
    .reset_index(drop=True)
)
lime_summary.to_csv(os.path.join(OUT_DIR, "oof_lime_summary.csv"), index=False)

merged = lime_summary.merge(
    shap_summary[["feature", "mean_abs_shap"]], on="feature", how="left"
).fillna({"mean_abs_shap": 0.0})
merged["lime_rank"] = merged["mean_abs_lime"].rank(ascending=False, method="first")
merged["shap_rank"] = merged["mean_abs_shap"].rank(ascending=False, method="first")

rho = spearmanr(merged["mean_abs_lime"], merged["mean_abs_shap"])
top_l = set(merged.nsmallest(TOP_FEATURES, "lime_rank")["feature"])
top_s = set(merged.nsmallest(TOP_FEATURES, "shap_rank")["feature"])

agree_df = pd.DataFrame([{
    "model": best_name,
    "spearman_rho_all_features": float(rho.statistic),
    "spearman_p": float(rho.pvalue),
    "top15_overlap": len(top_l & top_s),
    "top15_jaccard": len(top_l & top_s) / len(top_l | top_s),
    "n_subjects": int(pred_df["subject"].nunique()),
}])
agree_df.to_csv(os.path.join(OUT_DIR, "shap_lime_agreement.csv"), index=False)

# ----------------------------------------------------------------
# 5. Global figures (SHAP bar, SHAP beeswarm, LIME bar)
# ----------------------------------------------------------------
top_shap = shap_summary.head(TOP_FEATURES).sort_values("mean_abs_shap")
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(top_shap["feature"], top_shap["mean_abs_shap"], color="#4C72B0")
ax.set_xlabel("Mean |SHAP value|")
ax.set_title(f"Out-of-Fold SHAP Importance — {best_name} (Highest-Performing Model)")
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, f"shap_bar_{best_name}.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

top_feats = shap_summary.head(TOP_FEATURES)["feature"].tolist()
bee_df = shap_df[shap_df["feature"].isin(top_feats)]
feat_order = top_shap["feature"].tolist()
feat_y = {f: i for i, f in enumerate(feat_order)}
rng = np.random.default_rng(SEED)
fig, ax = plt.subplots(figsize=(10, 7))
for feat in feat_order:
    fd = bee_df[bee_df["feature"] == feat]
    jitter = rng.normal(0, 0.07, len(fd))
    ax.scatter(
        fd["shap_value"], np.full(len(fd), feat_y[feat]) + jitter,
        s=22, alpha=0.7, c=fd["shap_value"], cmap="RdBu_r", vmin=-0.15, vmax=0.15
    )
ax.axvline(0, linestyle="--", linewidth=1, color="grey")
ax.set_yticks(range(len(feat_order)))
ax.set_yticklabels(feat_order)
ax.set_xlabel("SHAP value (positive → DCP)")
ax.set_title(f"Out-of-Fold SHAP Beeswarm — {best_name} (Highest-Performing Model)")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, f"shap_beeswarm_{best_name}.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

top_lime = lime_summary.head(TOP_FEATURES).sort_values("mean_abs_lime")
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(top_lime["feature"], top_lime["mean_abs_lime"], color="#DD8452")
ax.set_xlabel("Mean |LIME weight|")
ax.set_title(f"Out-of-Fold LIME Importance — {best_name} (Highest-Performing Model)")
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, f"lime_bar_{best_name}.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------------
# 6. Individual-subject explanations (1 DCP, 1 TD, 1 error if present)
# ----------------------------------------------------------------
samples = []
tp_dcp = pred_df[(pred_df["true_target"] == 1) & (pred_df["predicted_target"] == 1)]
tn_td = pred_df[(pred_df["true_target"] == 0) & (pred_df["predicted_target"] == 0)]
errs = pred_df[pred_df["true_target"] != pred_df["predicted_target"]]
if len(tp_dcp):
    samples.append(("Correct_DCP", tp_dcp.iloc[0]))
if len(tn_td):
    samples.append(("Correct_TD", tn_td.iloc[0]))
if len(errs):
    samples.append(("Misclassified", errs.iloc[0]))

for tag, row in samples:
    subj = row["subject"]
    lbl = f"Subject {subj} ({tag}: true={row['true_target']}, pred={row['predicted_target']})"

    # Individual SHAP
    s_sub = shap_df[shap_df["subject"] == subj].copy()
    s_sub = s_sub.sort_values("abs_shap", ascending=False).head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cols = ["#C44E52" if v > 0 else "#4C72B0" for v in s_sub["shap_value"]]
    ax.barh(s_sub["feature"], s_sub["shap_value"], color=cols)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP contribution (+ → DCP, - → TD)")
    ax.set_title(f"{best_name} Individual SHAP — {lbl}")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"individual_shap_subject_{subj}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Individual LIME
    l_sub = lime_df[lime_df["subject"] == subj].copy()
    l_sub["abs_w"] = l_sub["lime_weight"].abs()
    l_sub = l_sub.sort_values("abs_w", ascending=False).head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cols = ["#C44E52" if v > 0 else "#4C72B0" for v in l_sub["lime_weight"]]
    ax.barh(l_sub["feature"], l_sub["lime_weight"], color=cols)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("LIME weight (+ → DCP, - → TD)")
    ax.set_title(f"{best_name} Individual LIME — {lbl}")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"individual_lime_subject_{subj}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

print("\n" + "=" * 70)
print("STEP 47 COMPLETE")
print(f"  Best model : {best_name}")
print(f"  SHAP-LIME Spearman rho = {agree_df['spearman_rho_all_features'].iloc[0]:.3f}, "
      f"Top-15 overlap = {agree_df['top15_overlap'].iloc[0]}/15")
print(f"  Outputs    : results/explainability_best_model/")
print("=" * 70)
