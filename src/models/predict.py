"""
Patient inference and clinical validation service for Heart Disease Risk Assessment.
Accepts raw patient records, validates clinical bounds, runs pipeline inference,
and returns structured diagnostic predictions, risk tiers, and validation messages.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import ALL_FEATURE_COLUMNS, FEATURE_METADATA, RISK_TIERS


def validate_clinical_bounds(
    patient_data: Union[Dict[str, Any], pd.DataFrame, pd.Series]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Validates that an incoming patient record conforms to the clinical schema and
    checks for physiological plausibility warnings.

    Parameters:
        patient_data: Single patient record as dict, Series, or 1-row DataFrame.

    Returns:
        Tuple[pd.DataFrame, List[str]]: (normalized 1-row DataFrame, list of clinical warning messages).

    Raises:
        ValueError: If mandatory columns are missing, data is empty, or values are outside hard limits.
    """
    if isinstance(patient_data, dict):
        df = pd.DataFrame([patient_data])
    elif isinstance(patient_data, pd.Series):
        df = patient_data.to_frame().T
    elif isinstance(patient_data, pd.DataFrame):
        df = patient_data.copy()
    else:
        raise TypeError(f"Unsupported patient record type: {type(patient_data)}. Expected dict, Series, or DataFrame.")

    if df.empty:
        raise ValueError("Patient record is empty.")

    # 1. Check for missing columns
    missing_cols = set(ALL_FEATURE_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Patient record is missing required clinical fields: {sorted(list(missing_cols))}. "
            f"Expected all of: {ALL_FEATURE_COLUMNS}"
        )

    # Reorder columns to standard canonical schema
    df_clean = df[ALL_FEATURE_COLUMNS].copy()

    warnings: List[str] = []
    row = df_clean.iloc[0]

    # 2. Hard limits & validation rules
    # Age
    if row["age"] <= 0 or row["age"] > 120:
        raise ValueError(f"Invalid age: {row['age']}. Must be between 1 and 120 years.")
    elif row["age"] < 18:
        warnings.append(f"Pediatric patient (age={row['age']}). Model was trained on adult cardiology cohort.")

    # Sex
    if row["sex"] not in [0, 1]:
        raise ValueError(f"Invalid biological sex: {row['sex']}. Expected 0 (Female) or 1 (Male).")

    # Chest Pain (cp)
    if row["cp"] not in [0, 1, 2, 3]:
        raise ValueError(f"Invalid chest pain type (cp): {row['cp']}. Expected values 0, 1, 2, or 3.")

    # Blood Pressure (trestbps)
    if row["trestbps"] <= 40 or row["trestbps"] > 300:
        raise ValueError(f"Physiologically implausible resting blood pressure: {row['trestbps']} mm Hg.")
    elif row["trestbps"] >= 180:
        warnings.append(f"Hypertensive crisis range (trestbps={row['trestbps']} mm Hg). Requires immediate emergency attention.")

    # Cholesterol (chol)
    if row["chol"] <= 40 or row["chol"] > 800:
        raise ValueError(f"Physiologically implausible serum cholesterol: {row['chol']} mg/dl.")
    elif row["chol"] > 400:
        warnings.append(f"Severe hypercholesterolemia (chol={row['chol']} mg/dl). Potential familial lipid disorder.")

    # Fasting Blood Sugar (fbs)
    if row["fbs"] not in [0, 1]:
        raise ValueError(f"Invalid fasting blood sugar flag (fbs): {row['fbs']}. Expected 0 or 1.")

    # Resting ECG (restecg)
    if row["restecg"] not in [0, 1, 2]:
        raise ValueError(f"Invalid resting ECG result: {row['restecg']}. Expected 0, 1, or 2.")

    # Max Heart Rate (thalach)
    if row["thalach"] <= 40 or row["thalach"] > 250:
        raise ValueError(f"Physiologically implausible maximum heart rate: {row['thalach']} bpm.")

    # Exercise Angina (exang)
    if row["exang"] not in [0, 1]:
        raise ValueError(f"Invalid exercise angina flag (exang): {row['exang']}. Expected 0 or 1.")

    # ST Depression (oldpeak)
    if row["oldpeak"] < 0 or row["oldpeak"] > 15:
        raise ValueError(f"Invalid ST depression (oldpeak): {row['oldpeak']} mm. Must be between 0.0 and 15.0 mm.")
    elif row["oldpeak"] >= 4.0:
        warnings.append(f"Severe ST segment depression (oldpeak={row['oldpeak']} mm). Indicates extensive exercise ischemia.")

    # Slope
    if row["slope"] not in [0, 1, 2]:
        raise ValueError(f"Invalid ST slope: {row['slope']}. Expected 0 (upsloping), 1 (flat), or 2 (downsloping).")

    # Fluoroscopy Vessels (ca)
    if row["ca"] not in [0, 1, 2, 3, 4]:
        raise ValueError(f"Invalid fluoroscopy vessel count (ca): {row['ca']}. Expected 0-3 (or 4 for unrecorded).")
    elif row["ca"] == 4:
        warnings.append("Vessel count 'ca=4' is an unrecorded surrogate code. Value will be statistically imputed.")

    # Thalassemia (thal)
    if row["thal"] not in [0, 1, 2, 3]:
        raise ValueError(f"Invalid thalassemia status (thal): {row['thal']}. Expected 1, 2, 3 (or 0 for unrecorded).")
    elif row["thal"] == 0:
        warnings.append("Thalassemia status 'thal=0' is an unrecorded surrogate code. Value will be statistically imputed.")

    return df_clean, warnings


def assign_risk_tier(probability: float) -> Tuple[str, str]:
    """
    Maps continuous probability into clinical triage risk tier and recommended action.
    """
    if probability <= RISK_TIERS["LOW"]["threshold"]:
        return "LOW", RISK_TIERS["LOW"]["action"]
    elif probability <= RISK_TIERS["MODERATE"]["threshold"]:
        return "MODERATE", RISK_TIERS["MODERATE"]["action"]
    elif probability <= RISK_TIERS["HIGH"]["threshold"]:
        return "HIGH", RISK_TIERS["HIGH"]["action"]
    else:
        return "CRITICAL", RISK_TIERS["CRITICAL"]["action"]


def predict_patient_risk(
    pipeline: Pipeline,
    patient_data: Union[Dict[str, Any], pd.DataFrame, pd.Series],
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Performs patient-level diagnostic inference using the end-to-end champion pipeline.

    Parameters:
        pipeline: Serialized end-to-end Pipeline (preprocessor + champion classifier).
        patient_data: Raw clinical record (dict, Series, or DataFrame).
        threshold: Decision threshold for binary classification. Default is 0.50.

    Returns:
        Dict[str, Any]: Diagnostic assessment with predicted class, risk probability,
                        triage tier, clinical action, and validation warnings.
    """
    # 1. Validate inputs and clinical plausibility
    df_clean, warnings = validate_clinical_bounds(patient_data)

    # 2. Execute end-to-end pipeline prediction
    # raw patient data -> preprocessing pipeline -> trained model -> prediction
    raw_prob = pipeline.predict_proba(df_clean)[0, 1]
    predicted_class = int(raw_prob >= threshold)
    risk_tier, action = assign_risk_tier(raw_prob)

    diagnosis_label = (
        "Presence of Significant Coronary Artery Disease"
        if predicted_class == 1
        else "Absence of Significant Coronary Artery Disease"
    )

    return {
        "predicted_class": predicted_class,
        "diagnosis_label": diagnosis_label,
        "risk_probability": round(float(raw_prob), 4),
        "risk_percentage": f"{raw_prob * 100:.1f}%",
        "risk_tier": risk_tier,
        "clinical_action": action,
        "decision_threshold_used": threshold,
        "validation_warnings": warnings,
        "patient_summary": df_clean.iloc[0].to_dict(),
    }

