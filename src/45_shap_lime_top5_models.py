# ============================================================
# STEP 45 — SHAP + LIME EXPLAINABILITY FOR TOP-5 BEST MODELS
# ============================================================
# Guide requirement: "For the best-performing model: LIME + SHAP individual
# predictions. For overall model analysis: feature-importance across all 5 models."
#
# This script:
#   1. Loads the top-5 models from the benchmark summary (step 23).
#   2. Runs out-of-fold SHAP (TreeExplainer for tree models, KernelExplainer
#      otherwise) on each of the 5 models.
#   3. Runs out-of-fold LIME on the best single model.
#   4. Builds a cross-model feature-importance comparison table.
#   5. Saves beeswarm / bar figures for each model and a combined rank table.
#
# Outputs in results/explainability_top5/:
#   shap_<ModelName>.csv        — per-subject per-feature SHAP values
#   shap_summary_<ModelName>.csv — aggregated mean|SHAP| per feature
#   lime_best_model.csv          — LIME weights for the #1 model
#   cross_model_importance.csv   — unified rank table (all 5 models)
#   figures/shap_bar_<Model>.png
#   figures/shap_beeswarm_<Model>.png (top model only)
#   figures/cross_model_importance.png

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
from sklearn.feature_selection import SelectKBest
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
import model_catalog as mc

OUT_DIR = os.path.join(BASE_DIR, "results", "explainability_top5")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

SEED = 42
TOP_N = 5         # top models to explain
TOP_FEATURES = 15
LIME_SAMPLES = 500

print("=" * 70)
print("STEP 45 — SHAP + LIME FOR TOP-5 MODELS")
print("=" * 70)

# ----------------------------------------------------------------
# Load data
# ----------------------------------------------------------------
DATA_PATH = os.path.join(BASE_DIR, "data", "processed",
                         "final_preprocessed_features_for_model_training_and_prediction.csv")
df = pd.read_csv(DATA_PATH)
y = df["target"]
groups = df["subject"]
X = df.drop(columns=["target", "subject", "side", "group"], errors="ignore") \
       .select_dtypes(include=[np.number])
feature_names = X.columns.tolist()
print(f"Dataset: {df.shape}  subjects={groups.nunique()}  features={len(feature_names)}")

# ----------------------------------------------------------------
# Top-5 models from step-23 benchmark summary
# ----------------------------------------------------------------
summary_path = os.path.join(BASE_DIR, "results", "model_benchmark", "model_benchmark_summary.csv")
if not os.path.exists(summary_path):
    raise FileNotFoundError(
        f"Run step 23 first:\n{summary_path}"
    )
bm = pd.read_csv(summary_path).sort_values("inner_accuracy_mean", ascending=False)
top5_names = bm.head(TOP_N)["model"].tolist()
print(f"\nTop-{TOP_N} models (by inner-CV accuracy): {top5_names}")

# ----------------------------------------------------------------
# CV splitters
# ----------------------------------------------------------------
outer_cv, inner_cv = mc.make_cv(SEED)
all_models_dict = mc.get_candidate_models()

# ----------------------------------------------------------------
# Helper — preprocess fold data
# ----------------------------------------------------------------

def preprocess(X_tr, y_tr, X_te, k):
    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    sel = SelectKBest(score_func=mc.mi_score, k=k)
    X_tr_t = sel.fit_transform(sc.fit_transform(imp.fit_transform(X_tr)), y_tr)
    X_te_t = sel.transform(sc.transform(imp.transform(X_te)))
    sel_feats = [feature_names[i] for i, s in enumerate(sel.get_support()) if s]
    return X_tr_t, X_te_t, sel_feats, imp, sc, sel

# ----------------------------------------------------------------
# SHAP loop — all top-5 models
# ----------------------------------------------------------------
cross_model_records = []

for rank, model_name in enumerate(top5_names, start=1):
    print(f"\n{'=' * 60}")
    print(f"[{rank}/{TOP_N}] SHAP for {model_name}")
    print("=" * 60)

    cls, fixed, grid = all_models_dict[model_name]
    combos = list(ParameterGrid(mc.sk_grid(model_name)))
    default_k, default_kw = mc.split_params(combos[0])

    shap_records = []
    for fold_idx, (tr, te) in enumerate(outer_cv.split(X, y, groups), start=1):
        X_tr_t, X_te_t, sel_feats, imp, sc, sel = preprocess(
            X.iloc[tr].to_numpy(dtype=float),
            y.iloc[tr].to_numpy(),
            X.iloc[te].to_numpy(dtype=float),
            default_k
        )
        model = cls(**{**fixed, **default_kw})
        model.fit(X_tr_t, y.iloc[tr].to_numpy())

        # Choose explainer based on model type
        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_te_t)
        except Exception:
            # Fallback: KernelExplainer with a small background sample
            bg = shap.sample(X_tr_t, min(50, len(X_tr_t)), random_state=SEED)
            explainer = shap.KernelExplainer(
                model.predict_proba if hasattr(model, "predict_proba")
                else model.predict,
                bg
            )
            sv = explainer.shap_values(X_te_t, nsamples=100)

        # Normalise SHAP output to (n_samples, n_features) for class-1
        sv_arr = np.asarray(sv)
        if isinstance(sv, list) and len(sv) >= 2:
            sv_pos = np.asarray(sv[1])
        elif sv_arr.ndim == 3:
            sv_pos = sv_arr[:, :, 1]
        elif sv_arr.ndim == 2:
            sv_pos = sv_arr
        else:
            sv_pos = sv_arr

        subj_te = groups.iloc[te].tolist()
        y_te = y.iloc[te].tolist()
        for i, subj in enumerate(subj_te):
            for j, feat in enumerate(sel_feats):
                shap_records.append({
                    "model": model_name, "fold": fold_idx,
                    "subject": subj, "true_target": y_te[i],
                    "feature": feat,
                    "shap_value": float(sv_pos[i, j]),
                    "abs_shap": abs(float(sv_pos[i, j])),
                })

    shap_df = pd.DataFrame(shap_records)
    shap_df.to_csv(os.path.join(OUT_DIR, f"shap_{model_name}.csv"), index=False)

    # Aggregate
    agg = (shap_df.groupby("feature")
           .agg(mean_abs_shap=("abs_shap", "mean"),
                mean_shap=("shap_value", "mean"),
                std_abs_shap=("abs_shap", "std"),
                n_appearances=("fold", "nunique"))
           .reset_index()
           .sort_values("mean_abs_shap", ascending=False)
           .reset_index(drop=True))
    agg["model"] = model_name
    agg["rank"] = agg.index + 1
    agg.to_csv(os.path.join(OUT_DIR, f"shap_summary_{model_name}.csv"), index=False)
    cross_model_records.append(agg)

    # Bar figure
    top_plot = agg.head(TOP_FEATURES).sort_values("mean_abs_shap")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top_plot["feature"], top_plot["mean_abs_shap"], color="#4C72B0")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"SHAP Feature Importance — {model_name}")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"shap_bar_{model_name}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved shap_bar_{model_name}.png")

# ----------------------------------------------------------------
# Beeswarm for the top-1 model only (most interpretable)
# ----------------------------------------------------------------
best_name = top5_names[0]
best_shap_df = pd.read_csv(os.path.join(OUT_DIR, f"shap_{best_name}.csv"))
best_agg = pd.read_csv(os.path.join(OUT_DIR, f"shap_summary_{best_name}.csv"))
top_feats = best_agg.head(TOP_FEATURES)["feature"].tolist()
bee_df = best_shap_df[best_shap_df["feature"].isin(top_feats)]
feat_order = best_agg.head(TOP_FEATURES).sort_values("mean_abs_shap")["feature"].tolist()
feat_y = {f: i for i, f in enumerate(feat_order)}
rng = np.random.default_rng(SEED)
fig, ax = plt.subplots(figsize=(10, 7))
for feat in feat_order:
    fd = bee_df[bee_df["feature"] == feat]
    jitter = rng.normal(0, 0.07, len(fd))
    ax.scatter(fd["shap_value"], np.full(len(fd), feat_y[feat]) + jitter,
               s=22, alpha=0.65, c=fd["shap_value"],
               cmap="RdBu_r", vmin=-0.15, vmax=0.15)
ax.axvline(0, linestyle="--", linewidth=1, color="grey")
ax.set_yticks(range(len(feat_order)))
ax.set_yticklabels(feat_order)
ax.set_xlabel("SHAP value (positive → DCP)")
ax.set_title(f"SHAP Beeswarm — {best_name} (best model)")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, f"shap_beeswarm_{best_name}.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"\nBeeswarm saved for best model: {best_name}")

# ----------------------------------------------------------------
# Cross-model importance comparison table
# ----------------------------------------------------------------
all_agg = pd.concat(cross_model_records, ignore_index=True)
pivot = all_agg.pivot_table(
    index="feature", columns="model",
    values="mean_abs_shap", aggfunc="mean"
).fillna(0)
pivot["mean_across_models"] = pivot.mean(axis=1)
pivot = pivot.sort_values("mean_across_models", ascending=False).reset_index()
pivot.to_csv(os.path.join(OUT_DIR, "cross_model_importance.csv"), index=False)

# Plot top-15 features across all 5 models
top15 = pivot.head(15).copy()
x = np.arange(len(top15))
width = 0.15
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
fig, ax = plt.subplots(figsize=(14, 6))
for i, mname in enumerate(top5_names):
    if mname in top15.columns:
        ax.bar(x + i * width, top15[mname], width=width, label=mname, color=colors[i % len(colors)])
ax.set_xticks(x + width * 2)
ax.set_xticklabels(top15["feature"], rotation=35, ha="right", fontsize=8)
ax.set_ylabel("Mean |SHAP value|")
ax.set_title("Feature Importance Comparison Across Top-5 Models (SHAP)")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "cross_model_importance.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------------
# LIME for the best model
# ----------------------------------------------------------------
print(f"\n{'=' * 60}\nLIME for best model: {best_name}\n{'=' * 60}")
cls_b, fixed_b, grid_b = all_models_dict[best_name]
combos_b = list(ParameterGrid(mc.sk_grid(best_name)))
k_b, kw_b = mc.split_params(combos_b[0])

lime_records = []
for fold_idx, (tr, te) in enumerate(outer_cv.split(X, y, groups), start=1):
    X_tr_np = X.iloc[tr].to_numpy(dtype=float)
    X_te_np = X.iloc[te].to_numpy(dtype=float)
    y_tr_np = y.iloc[tr].to_numpy()

    imp = SimpleImputer(strategy="median")
    X_tr_imp = imp.fit_transform(X_tr_np)
    X_te_imp = imp.transform(X_te_np)

    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr_imp)
    X_te_sc = sc.transform(X_te_imp)

    sel = SelectKBest(score_func=mc.mi_score, k=k_b)
    X_tr_sel = sel.fit_transform(X_tr_sc, y_tr_np)
    X_te_sel = sel.transform(X_te_sc)
    sel_feats_b = [feature_names[i] for i, s in enumerate(sel.get_support()) if s]

    model_b = cls_b(**{**fixed_b, **kw_b})
    model_b.fit(X_tr_sel, y_tr_np)

    predict_fn = (
        (lambda m: lambda Z: m.predict_proba(Z))(model_b)
        if hasattr(model_b, "predict_proba")
        else (lambda m: lambda Z: np.column_stack([1 - m.predict(Z), m.predict(Z)]))(model_b)
    )

    lime_exp = LimeTabularExplainer(
        X_tr_sel, feature_names=sel_feats_b,
        class_names=["TD", "DCP"],
        discretize_continuous=True, mode="classification",
        random_state=SEED
    )

    subj_te = groups.iloc[te].tolist()
    for i, row in enumerate(X_te_sel):
        exp = lime_exp.explain_instance(
            row, predict_fn, num_features=len(sel_feats_b),
            labels=(1,), num_samples=LIME_SAMPLES
        )
        w = dict(exp.as_map()[1])
        for j, feat in enumerate(sel_feats_b):
            lime_records.append({
                "fold": fold_idx, "subject": subj_te[i],
                "feature": feat, "lime_weight": w.get(j, 0.0),
            })
    print(f"  fold {fold_idx}: {len(X_te_sel)} test subjects explained")

lime_df = pd.DataFrame(lime_records)
lime_df.to_csv(os.path.join(OUT_DIR, "lime_best_model.csv"), index=False)

lime_agg = (lime_df.assign(abs_w=lime_df["lime_weight"].abs())
            .groupby("feature")
            .agg(mean_abs_lime=("abs_w", "mean"), mean_lime=("lime_weight", "mean"))
            .reset_index()
            .sort_values("mean_abs_lime", ascending=False).reset_index(drop=True))
lime_agg.to_csv(os.path.join(OUT_DIR, "lime_summary_best_model.csv"), index=False)

top_lime = lime_agg.head(TOP_FEATURES).sort_values("mean_abs_lime")
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(top_lime["feature"], top_lime["mean_abs_lime"], color="#DD8452")
ax.set_xlabel("Mean |LIME weight|")
ax.set_title(f"LIME Feature Importance — {best_name} (best model)")
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, f"lime_bar_{best_name}.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

print("\n" + "=" * 70)
print("STEP 45 COMPLETE")
print(f"  → results/explainability_top5/shap_*.csv")
print(f"  → results/explainability_top5/cross_model_importance.csv")
print(f"  → results/explainability_top5/lime_best_model.csv")
print(f"  → results/explainability_top5/figures/")
print("=" * 70)
