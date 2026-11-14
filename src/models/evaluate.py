"""
Comprehensive diagnostic and clinical evaluation module for machine learning models.
Calculates Accuracy, Precision, Recall (Sensitivity), Specificity, F1, ROC-AUC,
Confusion Matrix components, and assesses overfitting gaps.
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def get_prediction_probabilities(
    model: Any, X: Union[pd.DataFrame, np.ndarray]
) -> np.ndarray:
    """
    Safely retrieves probability scores for the positive class (target=1).
    Handles estimators with predict_proba or decision_function.
    """
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        if probs.ndim == 2:
            return probs[:, 1]
        return probs
    elif hasattr(model, "decision_function"):
        decisions = model.decision_function(X)
        # Convert decision scores into calibrated 0-1 probabilities via logistic sigmoid
        return expit(decisions)
    else:
        raise AttributeError(f"Model {type(model)} does not support predict_proba or decision_function.")


def evaluate_model_performance(
    model: Any,
    X_eval: Union[pd.DataFrame, np.ndarray],
    y_eval: Union[pd.Series, np.ndarray],
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Evaluates a trained classifier on an evaluation partition (e.g. test set).

    Returns a rich clinical dictionary with:
      - Diagnostic metrics: Accuracy, Precision, Recall, Specificity, F1, ROC-AUC
      - Confusion matrix counts: TP, TN, FP, FN
      - Calibration metric: Brier Score
      - ROC curve points: fpr, tpr, thresholds
    """
    y_true = np.asarray(y_eval)
    y_prob = get_prediction_probabilities(model, X_eval)
    y_pred = (y_prob >= threshold).astype(int)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Metrics
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    auc = float(roc_auc_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))

    # ROC curve points
    fpr, tpr, roc_thresh = roc_curve(y_true, y_prob)

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "specificity": round(spec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "brier_score": round(brier, 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "matrix": cm.tolist(),
        },
        "probabilities": y_prob,
        "predictions": y_pred,
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": roc_thresh.tolist(),
        },
    }


def compute_overfitting_analysis(
    train_eval: Dict[str, Any], test_eval: Dict[str, Any]
) -> Dict[str, float]:
    """
    Computes performance drop between training set and holdout test set to quantify overfitting.
    """
    acc_gap = round(train_eval["accuracy"] - test_eval["accuracy"], 4)
    rec_gap = round(train_eval["recall"] - test_eval["recall"], 4)
    auc_gap = round(train_eval["roc_auc"] - test_eval["roc_auc"], 4)

    return {
        "train_accuracy": train_eval["accuracy"],
        "test_accuracy": test_eval["accuracy"],
        "accuracy_gap": acc_gap,
        "train_recall": train_eval["recall"],
        "test_recall": test_eval["recall"],
        "recall_gap": rec_gap,
        "train_roc_auc": train_eval["roc_auc"],
        "test_roc_auc": test_eval["roc_auc"],
        "roc_auc_gap": auc_gap,
    }

