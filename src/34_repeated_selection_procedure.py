# ============================================================
# STEP 34 - REPEATED EVALUATION OF THE WHOLE SELECTION PROCEDURE
# ============================================================
# Repeats steps 23 + 29 (16-model benchmark, per-fold family + hyperparameter selection by inner CV, scoring on the
# unseen subjects) over 10 different subject-wise partitions (CV seeds 42..51). This gives the variability of the
# FULL procedure across splits (not just of three pre-selected families). Seed 42 must reproduce the primary result.
# CAVEAT: all repeats reuse the same 51 subjects; this is split variability, not generalisation uncertainty.

import os
import subprocess
import sys
import numpy as np
import pandas as pd
from scipy import stats

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# FEATURE_SET_DIR ("" = primary coordinate-only; e.g. "feature_sets/angle_enhanced") and EXPERIMENT_DATASET are passed by the runner.
FSD = os.environ.get("FEATURE_SET_DIR", "")
OUT = os.path.join(BASE, "results", FSD, "repeated_selection")
os.makedirs(OUT, exist_ok=True)
SEEDS = list(range(42, 52))

rows, winners = [], []
for seed in SEEDS:
    sub = os.path.join(FSD, "repeated_selection", f"seed_{seed}", "model_benchmark")
    env = {**os.environ, "CV_SEED": str(seed), "RESULTS_SUBDIR": sub}
    for script in ("23_model_benchmark.py", "29_nested_selection_estimate.py"):
        r = subprocess.run([sys.executable, os.path.join(BASE, "src", script)], env=env, cwd=BASE,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if r.returncode != 0:
            raise SystemExit(f"{script} failed for seed {seed}:\n{r.stderr[-2000:]}")
    nd = os.path.join(BASE, "results", FSD, "repeated_selection", f"seed_{seed}", "nested_selection")
    ci = pd.read_csv(os.path.join(nd, "nested_selection_ci.csv")).set_index("metric")["point_estimate"]
    fl = pd.read_csv(os.path.join(nd, "nested_selection_folds.csv"))
    rows.append({"seed": seed, "accuracy": ci["Accuracy"], "balanced_accuracy": ci["Balanced_Accuracy"],
                 "f1": ci["F1"], "mcc": ci["MCC"], "mean_fold_roc_auc": fl["roc_auc"].mean()})
    winners += [{"seed": seed, "fold": f.fold, "winner": f.model} for f in fl.itertuples()]
    print(f"seed {seed}: nested accuracy={rows[-1]['accuracy']:.3f} balanced={rows[-1]['balanced_accuracy']:.3f}", flush=True)

R = pd.DataFrame(rows); R.to_csv(os.path.join(OUT, "repeated_selection_per_seed.csv"), index=False)
W = pd.DataFrame(winners); W.to_csv(os.path.join(OUT, "winner_per_fold_per_seed.csv"), index=False)
W.groupby("winner").size().rename("times_selected").sort_values(ascending=False).reset_index().to_csv(
    os.path.join(OUT, "winner_frequency.csv"), index=False)

primary = pd.read_csv(os.path.join(BASE, "results", FSD, "nested_selection", "nested_selection_ci.csv")).set_index("metric")["point_estimate"]
assert abs(R.loc[R.seed == 42, "accuracy"].iloc[0] - primary["Accuracy"]) < 1e-12, "seed 42 must reproduce the primary nested estimate"
summ = []
for m in ["accuracy", "balanced_accuracy", "f1", "mcc", "mean_fold_roc_auc"]:
    v = R[m].to_numpy(); half = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    summ.append({"metric": m, "mean": v.mean(), "sd_across_partitions": v.std(ddof=1), "min": v.min(), "max": v.max(),
                 "ci95_low_of_mean": v.mean() - half, "ci95_high_of_mean": v.mean() + half, "n_partitions": len(v)})
pd.DataFrame(summ).to_csv(os.path.join(OUT, "repeated_selection_summary.csv"), index=False)
print(pd.DataFrame(summ).round(3).to_string(index=False))
print("\nWinner frequency over 10 partitions x 5 folds:\n", W.groupby("winner").size().sort_values(ascending=False).to_string())
