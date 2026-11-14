"""
Model training, hyperparameter optimization, and cross-validation module.
Fits candidate models strictly on the training partition using Stratified 5-Fold CV.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

from src.config import RANDOM_STATE
from src.models.registry import get_hyperparameter_grids, get_model_registry


def train_and_tune_model(
    model_name: str,
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    tune: bool = True,
    cv_splits: int = 5,
    scoring: str = "roc_auc",
) -> Tuple[Any, Dict[str, Any]]:
    """
    Trains a model on X_train using Stratified K-Fold CV.
    If tune=True, performs grid search strictly within the training folds.

    Parameters:
        model_name: Name of the model from the registry.
        X_train: Preprocessed training features.
        y_train: Training target labels.
        tune: Whether to execute GridSearchCV. Default is True.
        cv_splits: Number of stratified folds. Default is 5.
        scoring: Metric to optimize during cross-validation. Default is 'roc_auc'.

    Returns:
        Tuple[Any, Dict[str, Any]]: (fitted best estimator, cross-validation metrics summary).
    """
    models = get_model_registry()
    grids = get_hyperparameter_grids()

    if model_name not in models:
        raise ValueError(f"Unknown model name '{model_name}'. Available: {list(models.keys())}")

    base_estimator = models[model_name]
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)

    X_arr = np.asarray(X_train)
    y_arr = np.asarray(y_train)

    if tune and model_name in grids:
        grid = grids[model_name]
        grid_search = GridSearchCV(
            estimator=base_estimator,
            param_grid=grid,
            scoring=scoring,
            cv=cv,
            n_jobs=1,
            refit=True,
            return_train_score=True,
        )
        grid_search.fit(X_arr, y_arr)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_cv_score = float(grid_search.best_score_)
    else:
        best_model = base_estimator
        best_model.fit(X_arr, y_arr)
        best_params = best_model.get_params()
        best_cv_score = 0.0

    # Execute comprehensive 5-fold CV with all diagnostic metrics on the tuned model
    cv_metrics = cross_validate(
        best_model,
        X_arr,
        y_arr,
        cv=cv,
        scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
        return_train_score=False,
    )

    cv_summary = {
        "best_params": best_params,
        "cv_roc_auc_mean": round(float(np.mean(cv_metrics["test_roc_auc"])), 4),
        "cv_roc_auc_std": round(float(np.std(cv_metrics["test_roc_auc"])), 4),
        "cv_accuracy_mean": round(float(np.mean(cv_metrics["test_accuracy"])), 4),
        "cv_accuracy_std": round(float(np.std(cv_metrics["test_accuracy"])), 4),
        "cv_recall_mean": round(float(np.mean(cv_metrics["test_recall"])), 4),
        "cv_recall_std": round(float(np.std(cv_metrics["test_recall"])), 4),
        "cv_precision_mean": round(float(np.mean(cv_metrics["test_precision"])), 4),
        "cv_precision_std": round(float(np.std(cv_metrics["test_precision"])), 4),
        "cv_f1_mean": round(float(np.mean(cv_metrics["test_f1"])), 4),
        "cv_f1_std": round(float(np.std(cv_metrics["test_f1"])), 4),
    }

    return best_model, cv_summary


def extract_feature_importance(
    model: Any, feature_names: List[str]
) -> Optional[Dict[str, float]]:
    """
    Extracts feature importance weights or coefficients if the model architecture supports it.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        return dict(
            sorted(
                zip(feature_names, [round(float(v), 4) for v in importances]),
                key=lambda x: x[1],
                reverse=True,
            )
        )
    elif hasattr(model, "coef_"):
        coefs = model.coef_[0]
        # Return signed coefficients as well as magnitude ranking
        return dict(
            sorted(
                zip(feature_names, [round(float(v), 4) for v in coefs]),
                key=lambda x: abs(x[1]),
                reverse=True,
            )
        )
    return None


def create_end_to_end_pipeline(
    preprocessor: Pipeline, classifier: Any
) -> Pipeline:
    """
    Combines the fitted preprocessor with the trained champion model into a single
    inference pipeline usable on raw, un-transformed patient records.
    """
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

