"""
Unit tests for Phase 1 Machine Learning models and End-to-End inference pipeline.
Run with: pytest tests/test_model_pipeline.py -v
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from src.config import CHAMPION_METADATA_PATH, CHAMPION_MODEL_PATH
from src.models.registry import get_model_registry


def test_model_registry_completeness():
    registry = get_model_registry()
    expected_models = {
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
        "Support Vector Machine",
        "K-Nearest Neighbors",
        "Multi-Layer Perceptron",
    }
    assert set(registry.keys()) == expected_models


def test_champion_model_file_exists():
    assert CHAMPION_MODEL_PATH.exists(), f"Champion model missing at {CHAMPION_MODEL_PATH}"
    assert CHAMPION_MODEL_PATH.stat().st_size > 0, "Champion model file is empty"


def test_end_to_end_pipeline_inference():
    pipeline = joblib.load(CHAMPION_MODEL_PATH)

    # Raw patient record without prior manual transformation
    raw_patient = pd.DataFrame([{
        "age": 54,
        "sex": 1,
        "cp": 2,
        "trestbps": 130,
        "chol": 240,
        "fbs": 0,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 1.0,
        "slope": 2,
        "ca": 0,
        "thal": 2,
    }])

    pred = pipeline.predict(raw_patient)
    assert pred.shape == (1,)
    assert int(pred[0]) in {0, 1}

    proba = pipeline.predict_proba(raw_patient)
    assert proba.shape == (1, 2)
    assert np.isclose(proba.sum(), 1.0)
    assert 0.0 <= proba[0, 1] <= 1.0


def test_end_to_end_pipeline_handles_surrogate_missing_codes():
    pipeline = joblib.load(CHAMPION_MODEL_PATH)

    # Raw patient with surrogate missing codes ca=4 and thal=0
    raw_patient = pd.DataFrame([{
        "age": 60,
        "sex": 0,
        "cp": 1,
        "trestbps": 140,
        "chol": 280,
        "fbs": 1,
        "restecg": 1,
        "thalach": 140,
        "exang": 1,
        "oldpeak": 2.0,
        "slope": 1,
        "ca": 4,      # Out-of-bounds surrogate
        "thal": 0,    # Out-of-bounds surrogate
    }])

    pred = pipeline.predict(raw_patient)
    proba = pipeline.predict_proba(raw_patient)

    assert pred.shape == (1,)
    assert not np.isnan(proba).any()
    assert np.isclose(proba.sum(), 1.0)


def test_champion_metadata():
    assert CHAMPION_METADATA_PATH.exists()
    import json
    with open(CHAMPION_METADATA_PATH, "r") as f:
        meta = json.load(f)

    assert "champion_model_name" in meta
    assert meta["champion_model_name"] == "Support Vector Machine"
    assert "metrics_holdout_test" in meta
    assert meta["metrics_holdout_test"]["recall"] >= 0.85
    assert meta["metrics_holdout_test"]["accuracy"] >= 0.80
