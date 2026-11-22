# ============================================================
# STEP 39 - EXPLANATIONS OF THE NESTED MODEL-SELECTION PIPELINE (checklist 3)
# ============================================================
# SHAP/LIME (steps 20, 31) explain an EXPLORATORY Random Forest. They do NOT explain the fold-specific
# winners of the nested procedure (step 29). This step does: each outer fold's winning pipeline
# (family + k + hyper-parameters chosen on that fold's training subjects only) is refitted on the training
# subjects with the shared catalog, asserted to reproduce the saved fold accuracy, and then explained by
# permutation importance on that fold's HELD-OUT subjects (model-agnostic, works for every family).
# Importance = drop in held-out accuracy when one base feature's 6 descriptor columns are permuted together.
import ast, os, sys, warnings
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))
import model_catalog as mc, feature_sets as fs
OUT = os.path.join(BASE, "results", "nested_pipeline_explanations"); os.makedirs(OUT, exist_ok=True)
N_REP = 50

df = pd.read_csv(os.path.join(BASE, "data", "processed", "experiment_dataset.csv"))
y = df.target.to_numpy(); groups = df.subject.to_numpy()
Xdf = df.drop(columns=["target", "subject", "side", "group"], errors="ignore").select_dtypes(include=[np.number])
cols = Xdf.columns.tolist(); X = Xdf.to_numpy(float)
base_of = {c: fs.split_name(c)[0] for c in cols}
bases = sorted(set(base_of.values()))
win = pd.read_csv(os.path.join(BASE, "results", "nested_selection", "nested_selection_folds.csv")).set_index("fold")
outer, _ = mc.make_cv()
rows, repro = [], []
for fold, (tr, te) in enumerate(outer.split(X, y, groups), start=1):
    w = win.loc[fold]; prm = ast.literal_eval(w.best_parameters)
    pipe = mc.make_pipeline(w.model)
    pipe.set_params(selector__k=prm["k"], **{f"model__{k}": v for k, v in prm.items() if k != "k"})
    pipe.fit(X[tr], y[tr])
    acc = accuracy_score(y[te], pipe.predict(X[te]))
    assert abs(acc - w.accuracy) < 1e-9, f"fold {fold}: refit accuracy {acc} != saved {w.accuracy}"
    repro.append({"fold": fold, "model": w.model, "k": prm["k"], "saved_accuracy": w.accuracy, "refit_accuracy": acc, "match": True})
    rng = np.random.default_rng(42 + fold)
    sel = np.array(cols)[pipe.named_steps["selector"].get_support()]
    for b in bases:
        idx = [i for i, c in enumerate(cols) if base_of[c] == b]
        drops = []
        for _ in range(N_REP):
            Xp = X[te].copy(); perm = rng.permutation(len(te)); Xp[:, idx] = X[te][perm][:, idx]
            drops.append(acc - accuracy_score(y[te], pipe.predict(Xp)))
        rows.append({"fold": fold, "model": w.model, "base_feature": b, "importance": float(np.mean(drops)),
                     "n_cols_selected": int(sum(base_of[c] == b for c in sel))})
P = pd.DataFrame(rows); P.to_csv(os.path.join(OUT, "perm_importance_by_fold.csv"), index=False)
pd.DataFrame(repro).to_csv(os.path.join(OUT, "fold_winners_reproduced.csv"), index=False)
S = (P.groupby("base_feature").agg(mean_importance=("importance", "mean"), sd_across_folds=("importance", "std"),
     folds_positive=("importance", lambda s: int((s > 0).sum())), mean_cols_selected=("n_cols_selected", "mean"))
     .sort_values("mean_importance", ascending=False).reset_index())
S.to_csv(os.path.join(OUT, "nested_pipeline_importance.csv"), index=False)
with open(os.path.join(OUT, "summary.txt"), "w") as f:
    f.write("Permutation importance (accuracy drop, base feature = 6 descriptor columns permuted jointly) of each outer fold's\n"
            "nested winner on held-out subjects. Each fold has ~10 test subjects: values are noisy; read the ranking, not decimals.\n"
            "These explain the nested pipeline (fold-specific winners), not the exploratory Random Forest used for SHAP/LIME.\n\n")
    f.write(pd.DataFrame(repro).to_string(index=False) + "\n\n" + S.head(15).round(4).to_string(index=False) + "\n")
print(open(os.path.join(OUT, "summary.txt")).read())
