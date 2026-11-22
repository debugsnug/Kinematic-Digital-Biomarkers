# ============================================================
# STEP 26 - EMPIRICAL LEAKAGE CHECK: LABEL-PERMUTATION TEST
# ============================================================
# If any leakage existed in the pipeline (features, preprocessing,
# feature selection, or CV splitting), models trained on SHUFFLED
# labels would still score above chance. The same subject-wise
# pipeline is therefore run with permuted labels; accuracy must
# collapse to ~chance and the real score must sit far in the tail.
# Also reports how well metadata alone (subject ID number, arm side)
# predicts the label, to justify excluding them from predictors.

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "experiment_dataset.csv")
OUT_DIR = os.path.join(BASE_DIR, "results", "leakage_audit")
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
N_PERM = 1000

df = pd.read_csv(DATA_PATH)
y = df["target"].to_numpy()
groups = df["subject"].to_numpy()
X = df.drop(columns=["target", "subject", "side", "group"], errors="ignore").select_dtypes(include=[np.number])

def make_pipe():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("sel", SelectKBest(lambda a, b: mutual_info_classif(a, b, random_state=SEED), k=20)),
        ("rf", RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=1)),
    ])

def cv_score(Xm, yv, seed):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    pred = cross_val_predict(make_pipe(), Xm, yv, cv=cv, groups=groups)
    return balanced_accuracy_score(yv, pred), accuracy_score(yv, pred)

real_bacc, real_acc = cv_score(X, y, SEED)

rng = np.random.default_rng(SEED)
# Labels are constant within subject (one row per subject), so shuffling rows permutes
# subject-level labels without breaking the grouping. Permuted labels are drawn up front so the
# result does not depend on how the work is split across processes.
y_perms = [rng.permutation(y) for _ in range(N_PERM)]
perm = Parallel(n_jobs=-1)(delayed(lambda yp, i: cv_score(X, yp, SEED + i)[0])(yp, i)
                           for i, yp in enumerate(y_perms))
perm = np.array(perm)
p_value = (1 + np.sum(perm >= real_bacc)) / (1 + N_PERM)

# Metadata-only baselines (must NOT be used as predictors)
sid = pd.to_numeric(df["subject"], errors="coerce").to_numpy().reshape(-1, 1)
side = (df["side"] == "right").astype(int).to_numpy().reshape(-1, 1)
meta = {}
for name, M in [("subject_id_number", sid), ("arm_side", side)]:
    pipe = Pipeline([("rf", RandomForestClassifier(n_estimators=200, random_state=SEED))])
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    pred = cross_val_predict(pipe, M, y, cv=cv, groups=groups)
    meta[name] = balanced_accuracy_score(y, pred)

out = pd.DataFrame([{
    "real_balanced_accuracy": real_bacc,
    "real_accuracy": real_acc,
    "permuted_balanced_accuracy_mean": perm.mean(),
    "permuted_balanced_accuracy_sd": perm.std(ddof=1),
    "permuted_balanced_accuracy_95th_pct": np.percentile(perm, 95),
    "permutation_p_value": p_value,
    "n_permutations": N_PERM,
    "metadata_only_subject_id_balanced_acc": meta["subject_id_number"],
    "metadata_only_arm_side_balanced_acc": meta["arm_side"],
}])
out.to_csv(os.path.join(OUT_DIR, "label_permutation_test.csv"), index=False)
print(out.T.to_string(header=False))
print("NOTE: a sanity check, not proof of the absence of all leakage: it detects leakage that lets a model fit "
      "arbitrary labels (preprocessing/selection/split leaks); it cannot detect a feature that legitimately encodes "
      "recording site or session.")
verdict = "PASS" if perm.mean() < 0.60 and p_value < 0.05 else "FAIL"
print(f"\nLABEL-PERMUTATION LEAKAGE CHECK: {verdict} (null mean {perm.mean():.3f}, real {real_bacc:.3f}, p={p_value:.4f})")
if verdict == "FAIL":
    raise SystemExit(1)
