# ============================================================
# STEP 46 — DEEP LEARNING EXTENSIONS (EXPLORATORY / FUTURE WORK)
# ============================================================
# Guide requirement: "Explore adding two additional DL concepts for future
# extensions."
#
# IMPORTANT LABEL: These are EXPLORATORY stubs. They are NOT part of the
# validated pipeline and results must NOT be quoted alongside the CV estimates.
# They demonstrate two DL directions for a future, larger dataset.
#
# DL concept 1: Tabular MLP (sklearn MLPClassifier — already in the 16-model
#   catalog as "MLP"). Here we train a deeper/wider MLP with more epochs and
#   report its hold-out metrics for direct comparison.
#
# DL concept 2: 1D Convolutional Neural Network stub (conceptual / future).
#   Because the raw per-frame trajectories are time-series, a 1D-CNN is the
#   natural next step. This section shows the architecture using PyTorch if
#   available, otherwise prints the architecture description and skips training.
#   This is explicitly labelled FUTURE WORK.
#
# Outputs in results/dl_extensions/:
#   dl_mlp_extended.csv    — MLP hold-out results at 70:30, 80:20, 90:10
#   dl_architecture_cnn.txt — 1D-CNN architecture description
#   figures/dl_mlp_comparison.png

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    recall_score, roc_auc_score, confusion_matrix
)

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
import model_catalog as mc

OUT_DIR = os.path.join(BASE_DIR, "results", "dl_extensions")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

SEED = 42

print("=" * 70)
print("STEP 46 — DEEP LEARNING EXTENSIONS (EXPLORATORY / FUTURE WORK)")
print("=" * 70)

# ----------------------------------------------------------------
# Load data
# ----------------------------------------------------------------
DATA_PATH = os.path.join(BASE_DIR, "data", "processed",
                         "final_preprocessed_features_for_model_training_and_prediction.csv")
df = pd.read_csv(DATA_PATH)
y = df["target"].to_numpy()
X_df = df.drop(columns=["target", "subject", "side", "group"], errors="ignore") \
          .select_dtypes(include=[np.number])
X = X_df.to_numpy(dtype=float)
print(f"Subjects: {len(y)}  Predictors: {X.shape[1]}")

# ----------------------------------------------------------------
# DL CONCEPT 1 — Extended MLP (deeper, more epochs, dropout via
# early stopping).  Tabular MLP is widely used in DL-on-tabular.
# ----------------------------------------------------------------
print("\n--- DL Concept 1: Extended Tabular MLP ---")
print("Architecture: (84) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(1, Sigmoid)")
print("Implemented via sklearn MLPClassifier with extended hidden layers & epochs.\n")

MLP_CONFIGS = [
    {"hidden_layer_sizes": (64, 32), "max_iter": 1000, "alpha": 0.001,
     "early_stopping": True, "random_state": SEED, "label": "MLP-64-32"},
    {"hidden_layer_sizes": (128, 64, 32), "max_iter": 1500, "alpha": 0.0001,
     "early_stopping": True, "random_state": SEED, "label": "MLP-128-64-32"},
]

SPLIT_FRACS = [0.30, 0.20, 0.10]

mlp_rows = []
for cfg in MLP_CONFIGS:
    label = cfg.pop("label")
    for test_frac in SPLIT_FRACS:
        ratio_label = f"{int((1-test_frac)*100)}:{int(test_frac*100)}"
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=SEED)
        tr_idx, te_idx = next(sss.split(X, y))
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # Preprocess
        imp = SimpleImputer(strategy="median")
        sc = StandardScaler()
        sel = SelectKBest(score_func=mc.mi_score, k=40)
        X_tr_t = sel.fit_transform(sc.fit_transform(imp.fit_transform(X_tr)), y_tr)
        X_te_t = sel.transform(sc.transform(imp.transform(X_te)))

        model = MLPClassifier(**cfg)
        model.fit(X_tr_t, y_tr)
        y_pred = model.predict(X_te_t)
        y_prob = model.predict_proba(X_te_t)[:, 1]

        tn, fp, fn, tp = confusion_matrix(y_te, y_pred, labels=[0, 1]).ravel()
        spec = tn / (tn + fp) if (tn + fp) else np.nan

        row = {
            "model": label,
            "split_ratio": ratio_label,
            "n_train": len(y_tr), "n_test": len(y_te),
            "accuracy": accuracy_score(y_te, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_te, y_pred),
            "f1": f1_score(y_te, y_pred, zero_division=0),
            "sensitivity": recall_score(y_te, y_pred, zero_division=0),
            "specificity": spec,
            "roc_auc": roc_auc_score(y_te, y_prob),
        }
        mlp_rows.append(row)
        print(f"  {label} | {ratio_label}: Acc={row['accuracy']:.3f}  "
              f"BalAcc={row['balanced_accuracy']:.3f}  F1={row['f1']:.3f}  "
              f"AUC={row['roc_auc']:.3f}")
    cfg["label"] = label  # restore

mlp_df = pd.DataFrame(mlp_rows)
mlp_df.to_csv(os.path.join(OUT_DIR, "dl_mlp_extended.csv"), index=False)

# Bar chart
fig, ax = plt.subplots(figsize=(10, 5))
ratios = mlp_df["split_ratio"].unique()
x = np.arange(len(ratios))
width = 0.35
for i, label in enumerate(mlp_df["model"].unique()):
    sub = mlp_df[mlp_df["model"] == label]
    vals = [sub[sub["split_ratio"] == r]["balanced_accuracy"].values[0]
            if r in sub["split_ratio"].values else 0 for r in ratios]
    ax.bar(x + i * width, vals, width, label=label)
ax.axhline(0.814, linestyle="--", color="red", linewidth=1,
           label="Nested CV balanced acc. (reference)")
ax.set_xticks(x + width / 2)
ax.set_xticklabels(ratios)
ax.set_ylabel("Balanced Accuracy")
ax.set_title("Extended MLP — Hold-out Splits vs Nested CV Reference\n"
             "(Exploratory — not a validated result)")
ax.legend(fontsize=8)
ax.set_ylim(0, 1.05)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "dl_mlp_comparison.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("\n  → results/dl_extensions/dl_mlp_extended.csv")

# ----------------------------------------------------------------
# DL CONCEPT 2 — 1D-CNN architecture stub (future work)
# ----------------------------------------------------------------
print("\n--- DL Concept 2: 1D Convolutional Neural Network (FUTURE WORK) ---")
cnn_arch_text = """1D-CNN Architecture for Future Extension
==========================================

RATIONALE:
  The current 84 features are summary statistics (mean, SD, CV, slope, etc.)
  computed from per-frame trajectories. A 1D-CNN can learn temporal patterns
  directly from the raw trajectory sequences (shape: 8 reps × T_frames × 3 joints)
  without manual feature engineering, which is more scalable and potentially
  more expressive on a larger dataset.

PROPOSED ARCHITECTURE (PyTorch):
  Input  : (batch, channels=3, time_steps=~65)
           3 channels = x, y coordinates + resultant velocity per frame
           time_steps = padded/trimmed to max clip length (~65 frames)

  Block 1: Conv1D(in=3, out=32, kernel=5, padding=2) → BatchNorm → ReLU → MaxPool(2)
  Block 2: Conv1D(in=32, out=64, kernel=3, padding=1) → BatchNorm → ReLU → MaxPool(2)
  Block 3: Conv1D(in=64, out=128, kernel=3, padding=1) → BatchNorm → ReLU → AdaptiveAvgPool(1)
  Flatten : (batch, 128)
  FC      : Linear(128, 64) → Dropout(0.4) → ReLU → Linear(64, 1) → Sigmoid
  Loss    : BCEWithLogitsLoss (class-weighted for 30 DCP : 21 TD)
  Optimizer: Adam(lr=1e-3), weight_decay=1e-4
  Epochs  : 150, early stopping patience=20

MULTI-REP FUSION VARIANT:
  Run the same 1D-CNN on each of the 8 repetitions independently, then
  aggregate (mean pool or attention) before the classifier head. This
  captures within-subject variability across repetitions.

WHY THIS IS FUTURE WORK FOR n=51:
  - 51 subjects with an 80:20 split = 40 train / 11 test.
  - A CNN with ~50k parameters on 40 samples will massively overfit.
  - Minimum recommended sample size for a 1D-CNN in a clinical classification
    task: ~300–500 subjects (Raghu et al. 2019; Zhang et al. 2020).
  - Required before this can be used: (a) a site-balanced external cohort,
    (b) data augmentation (time warping, jitter, permutation), (c) pre-training
    on a larger motion-capture corpus.

SKETCH CODE (requires PyTorch — not installed in this environment):
-----------------------------------------------------------------
import torch
import torch.nn as nn

class ReachCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.Dropout(0.4), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):  # x: (batch, 3, T)
        return self.classifier(self.encoder(x)).squeeze(1)
-----------------------------------------------------------------

STATUS: Architecture documented. Training NOT attempted on n=51.
Next step: Collect ≥300 subjects with site metadata, then run with
5-fold subject-stratified CV and compare to the tabular pipeline.
"""
arch_path = os.path.join(OUT_DIR, "dl_architecture_cnn.txt")
with open(arch_path, "w", encoding="utf-8") as f:
    f.write(cnn_arch_text)
print(cnn_arch_text[:800])
print(f"\n  → {arch_path}")

print("\n" + "=" * 70)
print("STEP 46 COMPLETE")
print("  DL Concept 1: Extended MLP (trained, exploratory)")
print("  DL Concept 2: 1D-CNN (architecture documented, future work)")
print("=" * 70)
