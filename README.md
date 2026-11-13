# Heart Disease Risk Assessment Agent

[![Python Version](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.4%2B-red.svg)](https://xgboost.ai/)
[![Dataset](https://img.shields.io/badge/Kaggle-johnsmith88%2Fheart--disease-blueviolet.svg)](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
[![Tests](https://img.shields.io/badge/pytest-passing-brightgreen.svg)](tests/)

An end-to-end clinical machine learning and intelligent healthcare monitoring system designed for early cardiovascular risk stratification, automated diagnostic evaluation, and longitudinal patient tracking.

---

## 📌 Project Overview

Cardiovascular disease remains the leading cause of global mortality. The **Heart Disease Risk Assessment Agent** is an academic and clinical AI project organized into two coordinated phases:

* **Phase 1 — Machine Learning Model Benchmarking:**  
  Rigorous data cleaning, leak-free preprocessing, and comparative benchmarking of 6+ supervised classification algorithms (Logistic Regression, Random Forest, XGBoost, SVM, KNN, MLP, and optional ANN) evaluated across six standard clinical diagnostic metrics.
* **Phase 2 — AI Healthcare Monitoring Agent:**  
  An autonomous monitoring agent that ingests electronic patient health records, runs predictive inference via the optimal Phase 1 model, synthesizes patient-specific health-risk triage reports, and maintains a longitudinal monitoring log for early intervention.

---

## 🔍 Dataset Findings & Pre-Modeling Audit

The project uses the benchmark **Kaggle Heart Disease Dataset** (`johnsmith88/heart-disease-dataset`), derived from the Cleveland Clinic database:

| Metric | Dataset Observation | Clinical / Modeling Consequence |
| :--- | :--- | :--- |
| **Shape** | 1,025 rows, 14 columns | 13 physiological/clinical predictors + 1 binary target |
| **Duplicate Rows** | **723 rows (70.54%)** are exact duplicates | **Critical finding:** The dataset consists of only 302 unique patients repeated 3–4 times. Random splitting without deduplication causes massive test-set leakage. |
| **Target Variable** | `target` (0 = No Disease, 1 = Disease Present) | Balanced: 51.32% positive (raw), 54.30% positive (deduplicated). |
| **Missing Values** | 0 explicit `NaN` values | **Surrogate missing codes identified:** `ca == 4` (18 rows, 4 unique) and `thal == 0` (7 rows, 2 unique) represent missing clinical values. |
| **Feature Types** | 5 continuous numerical, 3 binary, 5 multi-class categorical | Requires unified pipeline scaling and one-hot encoding for linear/distance models. |

> Detailed findings, statistical distributions, and correlation matrices are documented in [DATASET_ANALYSIS.md](DATASET_ANALYSIS.md).

---

## 🏗️ Project Architecture

```
HeartAttack/
├── data/
│   ├── raw/
│   │   └── heart.csv                # Raw Kaggle dataset (1,025 rows, 14 columns)
│   └── processed/                   # Processed, cleaned, and split datasets (Phase 1)
├── src/
│   ├── __init__.py
│   ├── config.py                    # Global configurations, paths, bounds, and hyperparameters
│   ├── data/                        # Data loaders, transformers, and cleaning pipelines
│   │   └── __init__.py
│   ├── models/                      # ML benchmark implementations and evaluation tools
│   │   └── __init__.py
│   └── agent/                       # Phase 2 AI Health Monitoring Agent
│       └── __init__.py
├── scripts/
│   ├── inspect_dataset.py           # Automated schema & data-quality audit script
│   └── generate_eda_plots.py        # Generates exploratory visualizations
├── notebooks/                       # Exploratory analysis notebooks (optional)
├── reports/
│   ├── dataset_inspection_summary.json
│   └── figures/                     # Generated charts (distributions, correlations)
├── tests/
│   ├── __init__.py
│   └── test_dataset_integrity.py    # Automated schema and integrity test suite
├── .gitignore
├── requirements.txt
├── DATASET_ANALYSIS.md              # In-depth dataset inspection report
└── README.md                        # Project documentation
```

---

## 📐 System Design & Roadmap

```mermaid
flowchart TD
    subgraph Data Layer
        A[Raw Kaggle Dataset: 1,025 Rows] --> B[Inspection & Quality Audit]
        B --> C[Deduplication: 302 Unique Patients]
        B --> D[Missing Code Replacement: ca=4, thal=0]
    end

    subgraph Phase 1: Machine Learning Engine
        C & D --> E[Leak-Free Preprocessing Pipeline\n- ColumnTransformer\n- RobustScaler / StandardScaler\n- OneHotEncoder]
        E --> F[Stratified K-Fold Cross Validation]
        F --> G1[Logistic Regression]
        F --> G2[Random Forest]
        F --> G3[XGBoost Classifier]
        F --> G4[Support Vector Machine]
        F --> G5[K-Nearest Neighbors]
        F --> G6[Multi-Layer Perceptron / ANN]
        G1 & G2 & G3 & G4 & G5 & G6 --> H[Model Evaluation Matrix\nAccuracy | Precision | Recall | F1 | ROC-AUC | Confusion Matrix]
        H --> I[Select Optimal Clinical Model]
    end

    subgraph Phase 2: AI Healthcare Monitoring Agent
        J[Patient Health Record / Vitals Input] --> K[Agent Ingestion & Validation Engine]
        I --> L[Agent Inference Engine]
        K --> L
        L --> M[Health Risk Classifier & Probability Stratifier]
        M --> N[Clinical Report Generator\n- Risk Tier: Low / Moderate / High / Critical\n- Key Risk Drivers & Lifestyle Guidance]
        M --> O[Longitudinal Patient Monitoring Log]
    end
```

---

## 🔬 Phase 1: Machine Learning Benchmark Plan

### Algorithms Evaluated
1. **Logistic Regression:** Linear baseline with L2 regularization.
2. **Random Forest Classifier:** Non-linear ensemble with feature importance attribution.
3. **XGBoost:** Gradient boosted decision trees optimized for tabular predictive power.
4. **Support Vector Machine (SVM):** Non-linear RBF kernel classification.
5. **K-Nearest Neighbors (KNN):** Distance-based non-parametric classifier.
6. **Multi-Layer Perceptron (MLP):** Feedforward neural network with ReLU activations.
7. *(Optional Extension)* **Artificial Neural Network (ANN):** Deep learning architecture with dropout regularization.

### Evaluation Metrics
* **Accuracy:** Overall diagnostic correctness.
* **Precision:** Minimizing false alarms in clinical referrals.
* **Recall (Sensitivity):** Maximizing detection of positive heart disease cases (crucial for medical screening).
* **F1-Score:** Harmonic balance of precision and recall.
* **ROC-AUC:** Threshold-independent discriminatory power.
* **Confusion Matrix:** Granular count of True Positives, False Positives, True Negatives, and False Negatives.

---

## 🤖 Phase 2: AI Healthcare Monitoring Agent Plan

The Phase 2 agent will provide an automated, auditable, and interpretable clinical assistant:

1. **Patient Record Ingestion & Validation:** Validates incoming physiological parameters (blood pressure, cholesterol, ECG, ST depression) against physiological plausibility bounds.
2. **Risk Prediction Engine:** Executes inference using the champion model saved from Phase 1, generating calibrated posterior probabilities.
3. **Health-Risk Triage Report Generator:** Translates numeric probabilities into structured triage levels:
   - **Low Risk ($\le 30\%$):** Routine annual wellness screening.
   - **Moderate Risk ($31\% - 60\%$):** Lifestyle interventions and scheduled follow-up.
   - **High Risk ($61\% - 85\%$):** Urgent cardiologist referral and diagnostic imaging.
   - **Critical Risk ($> 85\%$):** Immediate medical intervention and continuous telemetry.
4. **Longitudinal Patient Monitoring Logs:** Appends patient observations to an audit log (`logs/patient_monitoring_log.jsonl`), tracking vitals trajectories and alerting clinicians to escalating risk trends over time.

---

## 🚀 Quickstart & Reproducibility

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-username/Heart-Disease-Risk-Assessment-Agent.git
cd Heart-Disease-Risk-Assessment-Agent

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Dataset Inspection
```bash
python scripts/inspect_dataset.py
```

### 3. Generate Exploratory Visualizations
```bash
python scripts/generate_eda_plots.py
```
Generated plots will be saved in `reports/figures/`.

### 4. Run Test Suite
```bash
pytest tests/ -v
```

---

## 📜 Academic Integrity & Standards
- **No Synthetic Fabrication:** Real raw records are verified without data invention.
- **Leak-Free Methodology:** Preprocessing transformers are fit exclusively on training folds during cross-validation.
- **Reproducibility:** Seeded random states (`RANDOM_STATE = 42`) across all components.
