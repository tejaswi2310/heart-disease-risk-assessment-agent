"""
Agent configuration, clinical thresholds, risk categories, and legal disclaimers.
Extends the central project configuration.
"""

from pathlib import Path
from typing import Dict, Any

from src.config import (
    CHAMPION_METADATA_PATH,
    CHAMPION_MODEL_PATH,
    MONITORING_DIR,
    MONITORING_LOG_PATH,
    PATIENT_REPORTS_DIR,
    PROJECT_ROOT,
    RISK_TIERS,
)

# Agent Version
AGENT_VERSION = "1.0.0"
AGENT_NAME = "Heart Disease Risk Assessment Agent"

# Clinical Physiological Bounds for Patient Input Validation
PHYSIOLOGICAL_BOUNDS: Dict[str, Dict[str, Any]] = {
    "age": {"min": 18, "max": 105, "unit": "years", "hard_min": 1, "hard_max": 120},
    "sex": {"valid_values": [0, 1], "labels": {0: "Female", 1: "Male"}},
    "cp": {
        "valid_values": [0, 1, 2, 3],
        "labels": {
            0: "Typical Angina",
            1: "Atypical Angina",
            2: "Non-anginal Chest Pain",
            3: "Asymptomatic",
        },
    },
    "trestbps": {"min": 80, "max": 240, "unit": "mm Hg", "hard_min": 50, "hard_max": 300},
    "chol": {"min": 100, "max": 600, "unit": "mg/dl", "hard_min": 60, "hard_max": 800},
    "fbs": {"valid_values": [0, 1], "labels": {0: "<= 120 mg/dl", 1: "> 120 mg/dl"}},
    "restecg": {
        "valid_values": [0, 1, 2],
        "labels": {
            0: "Normal",
            1: "ST-T Wave Abnormality",
            2: "Left Ventricular Hypertrophy",
        },
    },
    "thalach": {"min": 50, "max": 220, "unit": "bpm", "hard_min": 40, "hard_max": 250},
    "exang": {"valid_values": [0, 1], "labels": {0: "No", 1: "Yes"}},
    "oldpeak": {"min": 0.0, "max": 8.0, "unit": "mm", "hard_min": 0.0, "hard_max": 15.0},
    "slope": {
        "valid_values": [0, 1, 2],
        "labels": {0: "Upsloping", 1: "Flat", 2: "Downsloping"},
    },
    "ca": {
        "valid_values": [0, 1, 2, 3, 4],
        "labels": {0: "0 vessels", 1: "1 vessel", 2: "2 vessels", 3: "3 vessels", 4: "Unrecorded"},
    },
    "thal": {
        "valid_values": [0, 1, 2, 3],
        "labels": {0: "Unrecorded", 1: "Normal", 2: "Fixed Defect", 3: "Reversible Defect"},
    },
}

# Longitudinal Status Trend Labels
TREND_STATUS = {
    "FIRST_ASSESSMENT": "Baseline Assessment (No prior records)",
    "STABLE": "Stable (Risk change within ±5%)",
    "IMPROVED": "Improved (Cardiovascular risk decreased)",
    "DETERIORATED": "Deteriorated (Cardiovascular risk increased)",
    "CRITICAL_ALERT": "Critical Alert (Risk escalated to High/Critical tier)",
}

# Official Clinical Decision-Support Disclaimer
CLINICAL_DISCLAIMER = (
    "DISCLAIMER: This assessment is an automated machine-learning-based risk stratification and clinical "
    "decision-support tool. It does NOT constitute a medical diagnosis of coronary artery disease, acute coronary "
    "syndrome, or myocardial infarction. All predictions, risk probabilities, and recommendations must be reviewed, "
    "verified, and interpreted by a licensed cardiologist or qualified medical professional in conjunction with "
    "formal clinical evaluation, laboratory biomarkers, and diagnostic imaging."
)

