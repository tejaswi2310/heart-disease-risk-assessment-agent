"""
Model registry and hyperparameter space definitions for Phase 1 ML benchmarking.
"""

from typing import Any, Dict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from src.config import RANDOM_STATE


def get_model_registry() -> Dict[str, Any]:
    """
    Returns dictionary of un-tuned baseline estimators with fixed random states.
    """
    return {
        "Logistic Regression": LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=1000,
            solver="lbfgs",
        ),
        "Random Forest": RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_estimators=100,
        ),
        "XGBoost": XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_estimators=100,
        ),
        "Support Vector Machine": SVC(
            random_state=RANDOM_STATE,
            probability=True,  # Mandatory for probability calibration and ROC-AUC
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5,
        ),
        "Multi-Layer Perceptron": MLPClassifier(
            random_state=RANDOM_STATE,
            max_iter=600,
            early_stopping=True,
            n_iter_no_change=20,
        ),
    }


def get_hyperparameter_grids() -> Dict[str, Dict[str, list]]:
    """
    Returns search spaces for Stratified 5-Fold Cross Validation tuning on training data.
    Grids are selected to explore regularized, non-overfitting parameters.
    """
    return {
        "Logistic Regression": {
            "C": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
            "penalty": ["l2"],
            "class_weight": [None, "balanced"],
        },
        "Random Forest": {
            "n_estimators": [50, 100],
            "max_depth": [3, 5, 8, None],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "class_weight": [None, "balanced"],
        },
        "XGBoost": {
            "n_estimators": [50, 100],
            "max_depth": [2, 3, 4],
            "learning_rate": [0.05, 0.1, 0.2],
            "subsample": [0.8, 1.0],
        },
        "Support Vector Machine": {
            "C": [0.1, 0.5, 1.0, 5.0],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", 0.1],
        },
        "K-Nearest Neighbors": {
            "n_neighbors": [3, 5, 7, 9, 11],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"],
        },
        "Multi-Layer Perceptron": {
            "hidden_layer_sizes": [(32,), (64,), (32, 16)],
            "alpha": [0.001, 0.01, 0.1],
            "learning_rate_init": [0.001, 0.01],
            "activation": ["relu"],
        },
    }

