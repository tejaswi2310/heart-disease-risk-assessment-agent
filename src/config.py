"""
Heart Disease Risk Assessment Agent - Central Configuration
Defines project paths, feature metadata, seeds, model benchmarks, and Phase 2 agent parameters.
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
MONITORING_DIR = DATA_DIR / "monitoring"
REPORTS_DIR = PROJECT_ROOT / "reports"
PATIENT_REPORTS_DIR = REPORTS_DIR / "patient_reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

# Phase 2 Monitoring Log Path
MONITORING_LOG_PATH = MONITORING_DIR / "patient_monitoring_log.jsonl"

# Dataset Files
RAW_DATA_PATH = DATA_RAW_DIR / "heart.csv"
PROCESSED_DATA_PATH = DATA_PROCESSED_DIR / "heart_cleaned.csv"
DEDUP_DATA_PATH = DATA_PROCESSED_DIR / "heart_deduplicated.csv"

# Preprocessed Split Artifact Paths
X_TRAIN_RAW_PATH = DATA_PROCESSED_DIR / "X_train_raw.csv"
X_TEST_RAW_PATH = DATA_PROCESSED_DIR / "X_test_raw.csv"
Y_TRAIN_PATH = DATA_PROCESSED_DIR / "y_train.csv"
Y_TEST_PATH = DATA_PROCESSED_DIR / "y_test.csv"
X_TRAIN_PROCESSED_PATH = DATA_PROCESSED_DIR / "X_train_processed.csv"
X_TEST_PROCESSED_PATH = DATA_PROCESSED_DIR / "X_test_processed.csv"

# Fitted Preprocessor Pipeline Path
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
CHAMPION_MODEL_PATH = MODELS_DIR / "champion_model.joblib"
CHAMPION_METADATA_PATH = MODELS_DIR / "champion_metadata.json"
MODEL_RESULTS_CSV_PATH = REPORTS_DIR / "model_comparison_results.csv"
MODEL_RESULTS_JSON_PATH = REPORTS_DIR / "model_comparison_results.json"

# Global Reproducibility Seed & Splitting Configuration
RANDOM_STATE = 42
SPLIT_CONFIG = {
    "test_size": 0.20,
    "random_state": RANDOM_STATE,
    "stratify": True,
}

# Feature Definitions
TARGET_COLUMN = "target"

NUMERICAL_FEATURES = [
    "age",       # Age in years
    "trestbps",  # Resting blood pressure (mm Hg)
    "chol",      # Serum cholesterol (mg/dl)
    "thalach",   # Maximum heart rate achieved (bpm)
    "oldpeak",   # ST depression induced by exercise relative to rest
]

BINARY_CATEGORICAL_FEATURES = [
    "sex",    # 0 = female, 1 = male
    "fbs",    # Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
    "exang",  # Exercise induced angina (1 = yes, 0 = no)
]

MULTI_CATEGORICAL_FEATURES = [
    "cp",       # Chest pain type (0: typical angina, 1: atypical angina, 2: non-anginal, 3: asymptomatic)
    "restecg",  # Resting ECG (0: normal, 1: ST-T wave abnormality, 2: left ventricular hypertrophy)
    "slope",    # Slope of peak exercise ST segment (0: upsloping, 1: flat, 2: downsloping)
    "ca",       # Major vessels colored by fluoroscopy (0-3 valid, 4 = surrogate missing)
    "thal",     # Thalassemia (1: normal, 2: fixed defect, 3: reversible defect, 0 = surrogate missing)
]

ALL_FEATURE_COLUMNS = NUMERICAL_FEATURES + BINARY_CATEGORICAL_FEATURES + MULTI_CATEGORICAL_FEATURES

# Domain Metadata & Validation Bounds
FEATURE_METADATA = {
    "age": {"desc": "Age in years", "min": 18, "max": 100, "unit": "years"},
    "sex": {"desc": "Biological sex (1=Male, 0=Female)", "valid_values": [0, 1]},
    "cp": {"desc": "Chest pain type", "valid_values": [0, 1, 2, 3]},
    "trestbps": {"desc": "Resting blood pressure", "min": 60, "max": 240, "unit": "mm Hg"},
    "chol": {"desc": "Serum cholesterol", "min": 80, "max": 600, "unit": "mg/dl"},
    "fbs": {"desc": "Fasting blood sugar > 120 mg/dl", "valid_values": [0, 1]},
    "restecg": {"desc": "Resting electrocardiographic results", "valid_values": [0, 1, 2]},
    "thalach": {"desc": "Maximum heart rate achieved", "min": 50, "max": 230, "unit": "bpm"},
    "exang": {"desc": "Exercise induced angina", "valid_values": [0, 1]},
    "oldpeak": {"desc": "ST depression induced by exercise relative to rest", "min": 0.0, "max": 10.0, "unit": "mm"},
    "slope": {"desc": "Slope of peak exercise ST segment", "valid_values": [0, 1, 2]},
    "ca": {"desc": "Major vessels colored by fluoroscopy", "valid_values": [0, 1, 2, 3], "missing_code": 4},
    "thal": {"desc": "Thalassemia blood disorder", "valid_values": [1, 2, 3], "missing_code": 0},
    "target": {"desc": "Heart disease diagnosis (0=No disease, 1=Disease present)", "valid_values": [0, 1]},
}

# Phase 1: Machine Learning Benchmark Models
BENCHMARK_MODELS = [
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
    "Support Vector Machine (SVM)",
    "K-Nearest Neighbors (KNN)",
    "Multi-Layer Perceptron (MLP)",
]

# Phase 1: Evaluation Metrics
EVALUATION_METRICS = [
    "Accuracy",
    "Precision",
    "Recall (Sensitivity)",
    "F1-score",
    "ROC-AUC",
    "Confusion Matrix",
]

# Phase 2: Agent Health Risk Tiers
RISK_TIERS = {
    "LOW": {"threshold": 0.30, "action": "Routine annual health screening & healthy lifestyle maintenance"},
    "MODERATE": {"threshold": 0.60, "action": "Cardiological consultation, lifestyle modification, and lipid panel follow-up"},
    "HIGH": {"threshold": 0.85, "action": "Urgent specialist referral, diagnostic stress imaging, and immediate medical evaluation"},
    "CRITICAL": {"threshold": 1.00, "action": "Emergency medical consultation & intensive cardiovascular monitoring"},
}

