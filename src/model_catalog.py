# ============================================================
# SHARED MODEL CATALOG (single source of truth)
# ============================================================
# Every script that trains one of the 16 classifiers (benchmark 23, validation 16, repeated CV 27/34,
# SHAP 20, LIME 31) imports its definition, hyperparameter grid, preprocessing and scoring from HERE,
# so "Random_Forest" or "Linear_SVM" means exactly one configuration everywhere.
#
# Decisions (documented in AUDIT_REPORT.md):
#   * Linear_SVM = sklearn SVC(kernel="linear") (libsvm), NOT LinearSVC. One implementation everywhere.
#   * class_weight="balanced" for every model that supports it (classes are 30 DCP : 21 TD). Models
#     without native support (KNN, LDA, GaussianNB, MLP, GradientBoosting, AdaBoost, XGBoost) are unweighted.
#   * Selection metric for hyper-parameters: accuracy of the inner subject-wise 4-fold CV (matching the
#     teacher's highest-accuracy selection instruction); ties -> first configuration in sklearn ParameterGrid order.
#   * Scores: models with predict_proba give a probability; SVC gives its raw decision_function value.
#     Decision values are used ONLY for rank metrics (ROC-AUC, PR-AUC). They are never squashed through a
#     sigmoid and never called probabilities; Brier score is reported only for models with predict_proba.
#   * Preprocessing is always: median imputer -> StandardScaler -> SelectKBest(mutual information, seeded),
#     fitted on training subjects only (inside a Pipeline, or equivalently pre-fitted per training fold).

import numpy as np
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
                              HistGradientBoostingClassifier, RandomForestClassifier)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 42
OUTER_SPLITS = 5
INNER_SPLITS = 4
K_VALUES = [20, 40]
BAL = "balanced"


def mi_score(X, y):
    """Mutual-information filter score, seeded (sklearn adds random noise to continuous features)."""
    return mutual_info_classif(X, y, random_state=SEED)


def get_candidate_models():
    """name -> (class, fixed kwargs, grid). Grid key 'k' is the SelectKBest size; other keys are model params."""
    models = {
        "Logistic_L1": (LogisticRegression,
                        {"penalty": "l1", "solver": "liblinear", "max_iter": 3000, "class_weight": BAL, "random_state": SEED},
                        {"k": K_VALUES, "C": [0.01, 0.1, 1.0, 10.0]}),
        "Logistic_L2": (LogisticRegression,
                        {"penalty": "l2", "solver": "liblinear", "max_iter": 3000, "class_weight": BAL, "random_state": SEED},
                        {"k": K_VALUES, "C": [0.01, 0.1, 1.0, 10.0]}),
        "Linear_SVM": (SVC,
                       {"kernel": "linear", "probability": False, "class_weight": BAL, "random_state": SEED},
                       {"k": K_VALUES, "C": [0.01, 0.1, 1.0, 10.0]}),
        "RBF_SVM": (SVC,
                    {"kernel": "rbf", "probability": False, "class_weight": BAL, "random_state": SEED},
                    {"k": K_VALUES, "C": [0.1, 1.0, 10.0], "gamma": ["scale", 0.001]}),
        "Polynomial_SVM": (SVC,
                           {"kernel": "poly", "probability": False, "class_weight": BAL, "random_state": SEED},
                           {"k": K_VALUES, "C": [0.1, 1.0], "degree": [2, 3]}),
        "KNN": (KNeighborsClassifier,
                {"n_jobs": 1},
                {"k": K_VALUES, "n_neighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"]}),
        "LDA": (LinearDiscriminantAnalysis,
                {},
                {"k": K_VALUES, "solver": ["lsqr"], "shrinkage": [None, 0.1, 0.5, "auto"]}),
        "Random_Forest": (RandomForestClassifier,
                          {"n_estimators": 300, "class_weight": BAL, "random_state": SEED, "n_jobs": 1},
                          {"k": K_VALUES, "max_depth": [None, 10], "min_samples_leaf": [1, 2]}),
        "Extra_Trees": (ExtraTreesClassifier,
                        {"n_estimators": 300, "class_weight": BAL, "random_state": SEED, "n_jobs": 1},
                        {"k": K_VALUES, "max_depth": [None, 10], "min_samples_leaf": [1, 2]}),
        "HistGradientBoosting": (HistGradientBoostingClassifier,
                                 {"class_weight": BAL, "random_state": SEED},
                                 {"k": K_VALUES, "learning_rate": [0.05, 0.1], "max_iter": [100], "min_samples_leaf": [3, 5]}),
        "Gradient_Boosting": (GradientBoostingClassifier,
                              {"random_state": SEED},
                              {"k": K_VALUES, "n_estimators": [50, 100], "learning_rate": [0.05, 0.1], "max_depth": [2, 3]}),
        "AdaBoost": (AdaBoostClassifier,
                     {"random_state": SEED},
                     {"k": K_VALUES, "n_estimators": [50, 100, 200], "learning_rate": [0.1, 0.5, 1.0]}),
        "Decision_Tree": (DecisionTreeClassifier,
                          {"class_weight": BAL, "random_state": SEED},
                          {"k": K_VALUES, "max_depth": [3, 5, None]}),
        "Gaussian_NB": (GaussianNB,
                        {},
                        {"k": K_VALUES, "var_smoothing": [1e-9, 1e-6, 1e-3, 1e-1]}),
        "MLP": (MLPClassifier,
                {"max_iter": 2000, "early_stopping": False, "random_state": SEED},
                {"k": K_VALUES, "alpha": [0.0001, 0.01, 0.1], "hidden_layer_sizes": [(16,), (32,)]}),
    }
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:  # never silently drop a model
        raise ImportError("xgboost is required (pinned in requirements.txt); exactly 16 models must be evaluated.") from exc
    models["XGBoost"] = (XGBClassifier,
                         {"n_estimators": 100, "eval_metric": "logloss",
                          "random_state": SEED, "n_jobs": 1},
                         {"k": K_VALUES, "max_depth": [2, 3], "learning_rate": [0.05, 0.1]})
    return models


def sk_grid(name):
    """Grid in sklearn Pipeline naming (selector__k, model__<param>)."""
    _, _, grid = get_candidate_models()[name]
    return {("selector__k" if key == "k" else f"model__{key}"): list(vals) for key, vals in grid.items()}


def make_pipeline(name):
    cls, fixed, _ = get_candidate_models()[name]
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("selector", SelectKBest(score_func=mi_score, k=K_VALUES[0])),
        ("model", cls(**fixed)),
    ])


def split_params(sk_params):
    """{'selector__k': 20, 'model__C': 1.0} -> (20, {'C': 1.0})."""
    k = sk_params["selector__k"]
    kw = {key[len("model__"):]: v for key, v in sk_params.items() if key.startswith("model__")}
    return k, kw


def make_cv(seed=SEED):
    outer = StratifiedGroupKFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=seed)
    inner = StratifiedGroupKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=seed)
    return outer, inner


def fit_nested(name, X, y, groups, inner_cv):
    """Hyper-parameter search by inner subject-wise CV on the TRAINING subjects only (X, y, groups = training part)."""
    gs = GridSearchCV(make_pipeline(name), sk_grid(name), scoring="accuracy", cv=inner_cv, n_jobs=1, refit=True)
    gs.fit(X, y, groups=groups)
    return gs


def score_of(estimator, X):
    """(scores, type): probability of class 1 if available, else the raw decision-function value."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1], "probability"
    if hasattr(estimator, "decision_function"):
        return np.asarray(estimator.decision_function(X), dtype=float), "decision_function"
    return estimator.predict(X).astype(float), "label"


def is_probabilistic(name):
    cls, fixed, _ = get_candidate_models()[name]
    return hasattr(cls, "predict_proba") and not (cls is SVC)
