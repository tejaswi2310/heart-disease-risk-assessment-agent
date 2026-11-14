"""
Patient input intake, schema validation, and physiological consistency checking module.
Enforces strict medical plausibility bounds and privacy-conscious identifier handling.
"""

import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.agent.config import PHYSIOLOGICAL_BOUNDS
from src.config import ALL_FEATURE_COLUMNS


class PatientValidationError(ValueError):
    """Raised when an incoming patient record fails schema, type, or clinical validation."""
    pass


def generate_anonymous_patient_id(record: Dict[str, Any]) -> str:
    """
    Generates a deterministic, privacy-conscious pseudonym identifier
    from patient features and timestamp if none was provided.
    """
    seed_str = f"{record.get('age')}_{record.get('sex')}_{record.get('trestbps')}_{time.time()}"
    hash_digest = hashlib.sha256(seed_str.encode()).hexdigest()[:8].upper()
    return f"PT-{hash_digest}"


def validate_patient_record(
    patient_data: Union[Dict[str, Any], pd.DataFrame, pd.Series],
    patient_id: Optional[str] = None,
) -> Tuple[str, Dict[str, Union[int, float]], List[str]]:
    """
    Validates an incoming patient record for completeness, correct datatypes,
    physiological plausibility, and internal consistency.

    Parameters:
        patient_data: Raw patient record.
        patient_id: Optional external patient identifier.

    Returns:
        Tuple[str, Dict[str, Union[int, float]], List[str]]:
            (validated patient_id, cleaned feature dictionary, list of clinical warnings)

    Raises:
        PatientValidationError: If fields are missing, invalid, or physically impossible.
    """
    # 1. Normalize intake into a dictionary
    if isinstance(patient_data, dict):
        raw_dict = patient_data.copy()
    elif isinstance(patient_data, pd.Series):
        raw_dict = patient_data.to_dict()
    elif isinstance(patient_data, pd.DataFrame):
        if patient_data.empty:
            raise PatientValidationError("Patient record is empty.")
        raw_dict = patient_data.iloc[0].to_dict()
    else:
        raise PatientValidationError(f"Unsupported record format: {type(patient_data)}. Expected dict or DataFrame.")

    if not raw_dict:
        raise PatientValidationError("Patient record dictionary is empty.")

    # 2. Extract or generate Patient Identifier
    extracted_id = patient_id or raw_dict.pop("patient_id", None) or generate_anonymous_patient_id(raw_dict)
    clean_patient_id = str(extracted_id).strip()

    # 3. Check for Missing Required Fields
    missing_fields = [f for f in ALL_FEATURE_COLUMNS if f not in raw_dict]
    if missing_fields:
        raise PatientValidationError(
            f"Patient record is missing required clinical field(s): {missing_fields}. "
            f"All 13 features must be provided: {ALL_FEATURE_COLUMNS}"
        )

    clean_features: Dict[str, Union[int, float]] = {}
    warnings: List[str] = []

    # 4. Type Casting and Hard Range Checks
    for feat in ALL_FEATURE_COLUMNS:
        val = raw_dict[feat]

        # Check for NaN / None
        if val is None or pd.isna(val):
            # ca and thal allow unrecorded surrogate representation
            if feat == "ca":
                val = 4
                warnings.append("Fluoroscopy vessel count 'ca' is missing. Imputed with surrogate code 4.")
            elif feat == "thal":
                val = 0
                warnings.append("Thalassemia status 'thal' is missing. Imputed with surrogate code 0.")
            else:
                raise PatientValidationError(f"Feature '{feat}' cannot be null or NaN.")

        # Cast to float or int
        try:
            numeric_val = float(val)
        except (ValueError, TypeError):
            raise PatientValidationError(f"Invalid non-numeric value for '{feat}': {val!r}")

        bounds = PHYSIOLOGICAL_BOUNDS[feat]

        # Check categorical discrete sets
        if "valid_values" in bounds:
            int_val = int(numeric_val)
            if numeric_val != int_val or int_val not in bounds["valid_values"]:
                valid_str = ", ".join(map(str, bounds["valid_values"]))
                raise PatientValidationError(
                    f"Invalid categorical value for '{feat}': {val}. Expected one of [{valid_str}]."
                )
            clean_features[feat] = int_val

            # Flag surrogate codes
            if feat == "ca" and int_val == 4:
                warnings.append("Fluoroscopy vessels 'ca=4' indicates unrecorded examination. Value will be statistically imputed.")
            elif feat == "thal" and int_val == 0:
                warnings.append("Thalassemia status 'thal=0' indicates unrecorded laboratory record. Value will be statistically imputed.")

        # Check continuous numerical bounds
        else:
            hard_min = bounds["hard_min"]
            hard_max = bounds["hard_max"]
            unit = bounds.get("unit", "")

            if numeric_val < hard_min or numeric_val > hard_max:
                raise PatientValidationError(
                    f"Physiologically implausible value for '{feat}': {numeric_val} {unit}. "
                    f"Acceptable clinical bounds: [{hard_min}, {hard_max}]."
                )

            # Plausibility warnings for clinical extremes
            if numeric_val < bounds["min"] or numeric_val > bounds["max"]:
                warnings.append(
                    f"Unusual extreme for '{feat}': {numeric_val} {unit} "
                    f"(Standard clinical reference range: {bounds['min']} - {bounds['max']} {unit})."
                )

            # Cast oldpeak to float, others to int or float as appropriate
            clean_features[feat] = float(numeric_val) if feat == "oldpeak" else int(numeric_val)

    # 5. Inconsistent Inputs & Physiological Cross-Checks
    age = clean_features["age"]
    thalach = clean_features["thalach"]
    trestbps = clean_features["trestbps"]
    oldpeak = clean_features["oldpeak"]
    chol = clean_features["chol"]

    # Maximum heart rate vs age-predicted maximum (Tanaka formula: 208 - 0.7 * age, or standard 220 - age)
    max_theoretical_hr = 220 - age
    if thalach > max_theoretical_hr + 20:
        warnings.append(
            f"Achieved heart rate ({thalach} bpm) substantially exceeds age-predicted physiological maximum "
            f"(220 - age = {max_theoretical_hr} bpm)."
        )

    # Hypertensive crisis check
    if trestbps >= 180:
        warnings.append(f"Resting blood pressure ({trestbps} mm Hg) is in hypertensive crisis range (>=180 mm Hg).")

    # Severe hypercholesterolemia
    if chol >= 400:
        warnings.append(f"Severe hypercholesterolemia detected ({chol} mg/dl). Potential familial dyslipidemia.")

    # Severe ST depression
    if oldpeak >= 3.0:
        warnings.append(f"Pronounced exercise-induced ST depression ({oldpeak} mm) indicates extensive myocardial ischemia.")

    return clean_patient_id, clean_features, warnings

