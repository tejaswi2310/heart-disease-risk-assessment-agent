"""
Automated Quality Assurance and Validation test suite for Phase 1.
Validates:
1. Leak-free preprocessing and scaler parameters.
2. End-to-end inference equivalence (raw patient data -> pipeline -> model -> prediction).
3. Patient record schema validation, boundary checks, and error handling.
4. Risk tier stratification and probability calibration.
5. Determinism and reproducibility.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from src.config import (
    ALL_FEATURE_COLUMNS,
    CHAMPION_MODEL_PATH,
    DATA_PROCESSED_DIR,
    RAW_DATA_PATH,
)
from src.models.predict import predict_patient_risk, validate_clinical_bounds


@pytest.fixture(scope="module")
def champion_pipeline():
    assert CHAMPION_MODEL_PATH.exists(), f"Champion model not found at {CHAMPION_MODEL_PATH}"
    return joblib.load(CHAMPION_MODEL_PATH)


@pytest.fixture(scope="module")
def valid_patient_dict():
    return {
        "age": 55,
        "sex": 1,
        "cp": 1,
        "trestbps": 130,
        "chol": 240,
        "fbs": 0,
        "restecg": 0,
        "thalach": 155,
        "exang": 0,
        "oldpeak": 0.8,
        "slope": 2,
        "ca": 0,
        "thal": 2,
    }


# =====================================================================
# 1. Data Leakage & Preprocessing Integrity Tests
# =====================================================================

def test_raw_dataset_immutable():
    """Verifies that the original raw dataset was never overwritten or modified."""
    df_raw = pd.read_csv(RAW_DATA_PATH)
    assert df_raw.shape == (1025, 14), "Raw dataset dimensions were altered!"


def test_scaler_fitted_strictly_on_training_data():
    """
    Verifies that the StandardScaler parameters inside the pipeline exactly match
    the training set statistics and differ from test set statistics.
    """
    X_train_raw = pd.read_csv(DATA_PROCESSED_DIR / "X_train_raw.csv")
    X_test_raw = pd.read_csv(DATA_PROCESSED_DIR / "X_test_raw.csv")
    pipeline = joblib.load(CHAMPION_MODEL_PATH)

    preprocessor = pipeline.named_steps["preprocessor"]
    ct = preprocessor.named_steps["features"]
    num_scaler = ct.named_transformers_["num"].named_steps["scaler"]

    # Numerical feature means in X_train
    num_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    train_means = X_train_raw[num_cols].mean().values
    test_means = X_test_raw[num_cols].mean().values

    # Check scaler mean matches training set exactly
    np.testing.assert_allclose(num_scaler.mean_, train_means, rtol=1e-5)

    # Check scaler mean does NOT equal test set means (zero leakage)
    assert not np.allclose(num_scaler.mean_, test_means, rtol=1e-3), "Scaler appears to be contaminated by test data!"


def test_train_test_partitions_disjoint():
    """Verifies that no identical records exist across train and test partitions."""
    X_train_raw = pd.read_csv(DATA_PROCESSED_DIR / "X_train_raw.csv")
    X_test_raw = pd.read_csv(DATA_PROCESSED_DIR / "X_test_raw.csv")

    # Combine partitions and ensure no duplicate overlap exists
    combined = pd.concat([X_train_raw, X_test_raw], axis=0)
    assert combined.duplicated().sum() == 0, "Data leakage detected: Duplicate patient found in train and test splits!"


# =====================================================================
# 2. End-to-End Inference & Preprocessing Consistency Tests
# =====================================================================

def test_end_to_end_inference_dict_input(champion_pipeline, valid_patient_dict):
    """
    Verifies raw patient dictionary -> pipeline -> prediction.
    Tests that passing a dictionary produces valid predicted class and probability.
    """
    result = predict_patient_risk(champion_pipeline, valid_patient_dict)

    assert result["predicted_class"] in [0, 1]
    assert 0.0 <= result["risk_probability"] <= 1.0
    assert result["risk_tier"] in ["LOW", "MODERATE", "HIGH", "CRITICAL"]
    assert "diagnosis_label" in result
    assert "clinical_action" in result
    assert isinstance(result["validation_warnings"], list)


def test_end_to_end_inference_dataframe_input(champion_pipeline, valid_patient_dict):
    """Verifies that passing a 1-row DataFrame produces identical results to dict input."""
    df_patient = pd.DataFrame([valid_patient_dict])
    res_dict = predict_patient_risk(champion_pipeline, valid_patient_dict)
    res_df = predict_patient_risk(champion_pipeline, df_patient)

    assert res_dict["predicted_class"] == res_df["predicted_class"]
    assert np.isclose(res_dict["risk_probability"], res_df["risk_probability"])


def test_end_to_end_inference_surrogate_imputation(champion_pipeline, valid_patient_dict):
    """
    Verifies that raw surrogate missing codes (ca=4 and thal=0) are correctly handled
    by the inference pipeline and flagged in validation warnings.
    """
    patient_with_surrogates = valid_patient_dict.copy()
    patient_with_surrogates["ca"] = 4
    patient_with_surrogates["thal"] = 0

    result = predict_patient_risk(champion_pipeline, patient_with_surrogates)

    assert result["predicted_class"] in [0, 1]
    assert not np.isnan(result["risk_probability"])
    # Validation warnings should flag both surrogate codes
    warning_text = " ".join(result["validation_warnings"])
    assert "ca=4" in warning_text
    assert "thal=0" in warning_text


def test_column_order_invariance(champion_pipeline, valid_patient_dict):
    """Verifies that reordering input dictionary keys does not alter predictions."""
    reversed_dict = {k: valid_patient_dict[k] for k in reversed(list(valid_patient_dict.keys()))}

    res_normal = predict_patient_risk(champion_pipeline, valid_patient_dict)
    res_reversed = predict_patient_risk(champion_pipeline, reversed_dict)

    assert res_normal["predicted_class"] == res_reversed["predicted_class"]
    assert np.isclose(res_normal["risk_probability"], res_reversed["risk_probability"])


# =====================================================================
# 3. Error Handling and Clinical Validation Tests
# =====================================================================

def test_missing_column_raises_informative_error(champion_pipeline, valid_patient_dict):
    """Verifies that omitting a required feature raises a descriptive ValueError."""
    incomplete_patient = valid_patient_dict.copy()
    del incomplete_patient["chol"]

    with pytest.raises(ValueError, match="missing required clinical fields.*chol"):
        predict_patient_risk(champion_pipeline, incomplete_patient)


def test_empty_record_raises_error(champion_pipeline):
    """Verifies that an empty dictionary raises a ValueError."""
    with pytest.raises(ValueError, match="Patient record is empty"):
        predict_patient_risk(champion_pipeline, {})


def test_invalid_age_raises_error(champion_pipeline, valid_patient_dict):
    """Verifies that impossible ages trigger validation errors."""
    bad_patient = valid_patient_dict.copy()
    bad_patient["age"] = -10
    with pytest.raises(ValueError, match="Invalid age"):
        predict_patient_risk(champion_pipeline, bad_patient)


def test_invalid_sex_raises_error(champion_pipeline, valid_patient_dict):
    """Verifies that invalid categorical codes trigger validation errors."""
    bad_patient = valid_patient_dict.copy()
    bad_patient["sex"] = 2  # Only 0 and 1 are valid
    with pytest.raises(ValueError, match="Invalid biological sex"):
        predict_patient_risk(champion_pipeline, bad_patient)


def test_hypertensive_crisis_warning_flagged(champion_pipeline, valid_patient_dict):
    """Verifies that blood pressure >= 180 triggers clinical hypertensive warning."""
    hypertensive_patient = valid_patient_dict.copy()
    hypertensive_patient["trestbps"] = 195

    result = predict_patient_risk(champion_pipeline, hypertensive_patient)
    warning_text = " ".join(result["validation_warnings"])
    assert "Hypertensive crisis" in warning_text


# =====================================================================
# 4. Reproducibility & Determinism Tests
# =====================================================================

def test_deterministic_predictions(champion_pipeline, valid_patient_dict):
    """Verifies that repeated invocations on identical data produce identical probability outputs."""
    probs = [
        predict_patient_risk(champion_pipeline, valid_patient_dict)["risk_probability"]
        for _ in range(5)
    ]
    assert len(set(probs)) == 1, "Non-deterministic predictions detected!"
