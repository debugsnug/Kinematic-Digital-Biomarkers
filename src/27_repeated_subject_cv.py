# ============================================================
# STEP 27 - REPEATED SUBJECT-WISE NESTED CV OF THE SELECTED FAMILIES
# ============================================================
# The families selected in step 23 are re-evaluated over 10 different subject-wise partitions (seeds 42..51),
# each with its own inner-CV hyperparameter search (shared model_catalog.py definitions).
# CAVEAT: the family list itself was chosen with the global ranking (optimistic), and every repeat re-partitions
# the SAME 51 subjects, so this shows sensitivity to the split only. The repeated evaluation of the WHOLE
# selection procedure is step 34. ROC-AUC = mean of the per-fold AUCs (decision values of different fits are
# not on one scale, so no pooled AUC).

import json
import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef, roc_auc_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
import model_catalog as mc

warnings.filterwarnings("ignore")
OUT_DIR = os.path.join(BASE_DIR, "results", "repeated_cv")
os.makedirs(OUT_DIR, exist_ok=True)
N_REPEATS = 10

df = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "experiment_dataset.csv"))
y = df["target"].to_numpy(); groups = df["subject"].to_numpy()
X = df.drop(columns=["target", "subject", "side", "group"], errors="ignore").select_dtypes(include=[np.number])
selected = json.load(open(os.path.join(BASE_DIR, "results", "model_benchmark", "selected_top_models.json")))["selected_models"]

rows = []
for name in selected:
    for rep in range(N_REPEATS):
        outer, inner = mc.make_cv(mc.SEED + rep)
        yt, yp, aucs = [], [], []
        for tr, te in outer.split(X, y, groups):
            gs = mc.fit_nested(name, X.iloc[tr], y[tr], groups[tr], inner)
            sc, _ = mc.score_of(gs.best_estimator_, X.iloc[te])
            yt += list(y[te]); yp += list(gs.predict(X.iloc[te])); aucs.append(roc_auc_score(y[te], sc))
        rows.append({"model": name, "repeat": rep, "seed": mc.SEED + rep, "accuracy": accuracy_score(yt, yp),
                     "balanced_accuracy": balanced_accuracy_score(yt, yp), "f1": f1_score(yt, yp),
                     "mcc": matthews_corrcoef(yt, yp), "roc_auc": float(np.mean(aucs))})
        print(f"{name} repeat {rep+1}/{N_REPEATS}: acc={rows[-1]['accuracy']:.3f} auc={rows[-1]['roc_auc']:.3f}", flush=True)

res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT_DIR, "repeated_cv_per_repeat.csv"), index=False)
summ = []
for name, g in res.groupby("model"):
    row = {"model": name, "n_repeats": len(g)}
    for m in ["accuracy", "balanced_accuracy", "f1", "mcc", "roc_auc"]:
        v = g[m].to_numpy()
        half = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
        row.update({f"{m}_mean": v.mean(), f"{m}_sd_across_repeats": v.std(ddof=1),
                    f"{m}_ci95_low": v.mean() - half, f"{m}_ci95_high": v.mean() + half})
    summ.append(row)
summ = pd.DataFrame(summ)
summ.to_csv(os.path.join(OUT_DIR, "repeated_cv_summary.csv"), index=False)
print(summ.round(3).T.to_string())
