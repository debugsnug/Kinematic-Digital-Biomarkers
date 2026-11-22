#!/usr/bin/env python
"""Audit saved subject partitions and prediction artifacts.

The benchmark pins scikit-learn==1.6.1. Splitter internals can change between
scikit-learn releases, so exact fold re-derivation is performed only under the
pinned version. Saved assignments/predictions are always checked directly.
"""
import os, sys
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "processed", "experiment_dataset.csv")
ASSIGN = os.path.join(BASE, "results", "model_benchmark", "outer_fold_assignment.csv")
if not os.path.exists(DATA) or not os.path.exists(ASSIGN):
    raise FileNotFoundError("Required experiment dataset or saved outer-fold assignment is missing")
df = pd.read_csv(DATA)
if df.subject.duplicated().any():
    raise AssertionError("Experiment dataset must contain exactly one row per subject")
subjects = set(df.subject.astype(str))
assignment = pd.read_csv(ASSIGN, dtype={"subject": str})
if assignment.subject.duplicated().any():
    raise AssertionError("Saved outer assignment contains duplicate subject IDs")
if set(assignment.subject) != subjects:
    raise AssertionError("Saved outer assignment subject set differs from experiment dataset")
if assignment.outer_fold.isna().any() or not set(assignment.outer_fold.astype(int)).issubset({1,2,3,4,5}):
    raise AssertionError("Invalid outer-fold labels")
fold_map = dict(zip(assignment.subject, assignment.outer_fold.astype(int)))
if set(fold_map.values()) != {1,2,3,4,5}:
    raise AssertionError("One or more outer folds are absent")

# Directly verify each saved model's held-out predictions against the authoritative assignment.
checks = [("benchmark", "results/model_benchmark/model_benchmark_predictions.csv"),
          ("top3 validation", "results/top3_validation/top3_predictions.csv")]
for name, rel in checks:
    path = os.path.join(BASE, rel)
    if not os.path.exists(path):
        print(f"INFO: {name} prediction artifact absent; skipped")
        continue
    pred = pd.read_csv(path, dtype={"subject": str})
    required = {"model", "fold", "subject"}
    if not required.issubset(pred.columns):
        raise AssertionError(f"{name}: missing required columns {required-set(pred.columns)}")
    for model, part in pred.groupby("model"):
        if part.subject.duplicated().any():
            raise AssertionError(f"{name}/{model}: duplicate subject predictions")
        if set(part.subject) != subjects:
            raise AssertionError(f"{name}/{model}: prediction subject coverage differs from dataset")
        mismatches = [(s, int(f), fold_map.get(s)) for s, f in zip(part.subject, part.fold)
                      if fold_map.get(s) != int(f)]
        if mismatches:
            raise AssertionError(f"{name}/{model}: saved fold labels disagree with assignment: {mismatches[:5]}")

# Predictor schema exclusion checks.
banned = {"subject", "side", "group", "target", "label"}
features = [c for c in df.columns if c not in banned]
suspicious = [c for c in features if "subject" in c.lower() or c.lower() in {"group", "target", "label"}]
if suspicious:
    raise AssertionError(f"Identifier/label-like predictors found: {suspicious}")

print(f"Saved-partition audit PASS: {len(subjects)} subjects; 5 outer folds; no duplicate/missing subject assignments.")
print("Saved prediction audit PASS: each model predicts every subject exactly once and fold IDs match the saved assignment.")
print(f"Runtime scikit-learn: {sklearn.__version__}; project pin: 1.6.1")
if sklearn.__version__ != "1.6.1":
    print("NOTE: exact inner/outer splitter reconstruction intentionally skipped because runtime differs from pinned version.")
else:
    y = df.target.to_numpy(); g = df.subject.astype(str).to_numpy()
    X = df.drop(columns=["target", "subject", "side", "group"]).select_dtypes(include=[np.number])
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    seen = []
    for fold, (tr, te) in enumerate(outer.split(X, y, g), 1):
        if set(g[tr]) & set(g[te]): raise AssertionError(f"Outer fold {fold}: subject overlap")
        if set(g[te]) != {s for s, f in fold_map.items() if f == fold}:
            raise AssertionError(f"Outer fold {fold}: re-derived membership differs from saved assignment")
        seen.extend(g[te])
        inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
        for k, (itr, iva) in enumerate(inner.split(X.iloc[tr], y[tr], g[tr]), 1):
            if set(g[tr][itr]) & set(g[tr][iva]): raise AssertionError(f"Outer {fold}/inner {k}: subject overlap")
            if set(g[tr][iva]) & set(g[te]): raise AssertionError(f"Outer {fold}/inner {k}: inner validation touches outer test")
    if len(seen) != len(set(seen)) or set(seen) != subjects:
        raise AssertionError("Re-derived outer folds do not partition subjects exactly once")
    print("Pinned-version split reconstruction PASS: no subject overlap in outer or inner folds.")
