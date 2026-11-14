# Heart Disease Risk Assessment Agent

[![Python Version](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.4%2B-red.svg)](https://xgboost.ai/)
[![Dataset](https://img.shields.io/badge/Kaggle-johnsmith88%2Fheart--disease-blueviolet.svg)](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
[![Tests](https://img.shields.io/badge/pytest-91%20passed-brightgreen.svg)](tests/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.60%2B-FF4B4B.svg)](https://streamlit.io/)

An end-to-end clinical machine learning and intelligent healthcare monitoring system designed for early cardiovascular risk stratification, automated diagnostic evaluation, and longitudinal patient tracking — with a full Streamlit web UI.

---

## 📌 Project Overview

Cardiovascular disease remains the leading cause of global mortality. The **Heart Disease Risk Assessment Agent** is an academic and clinical AI project organized into two coordinated phases:

* **Phase 1 — Machine Learning Model Benchmarking:**  
  Rigorous data cleaning, leak-free preprocessing, and comparative benchmarking of 6+ supervised classification algorithms (Logistic Regression, Random Forest, XGBoost, SVM, KNN, MLP, and optional ANN) evaluated across six standard clinical diagnostic metrics.
* **Phase 2 — AI Healthcare Monitoring Agent:**  
  An autonomous monitoring agent that ingests electronic patient health records, runs predictive inference via the optimal Phase 1 model, synthesizes patient-specific health-risk triage reports, and maintains a longitudinal monitoring log for early intervention.
* **Phase 3 — Streamlit Web UI:**  
  A clean, professional, healthcare-grade decision-support interface connecting directly to the Phase 2 agent. Provides patient intake forms, real-time risk assessment, longitudinal monitoring charts, model governance dashboards, and one-click report export (Markdown + JSON).

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

## 🔬 Phase 1: Machine Learning Benchmark Results

The Phase 1 comparative benchmark evaluated all 6 supervised algorithms using **Stratified 5-Fold Cross-Validation** on the training partition ($N = 241$), followed by holdout evaluation on the untouched test partition ($N = 61$):

| Model | CV ROC-AUC | Test Acc | Test Prec | Test Rec (Sensitivity) | Test Spec | Test F1 | Test ROC-AUC | False Negatives (FN) | Overfitting Gap ($\Delta_{\text{Acc}}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Support Vector Machine (SVM)\*** | **0.918 ± 0.041** | **0.853** | **0.853** | **0.879** | **0.821** | **0.866** | **0.894** | **4** | **+0.031** |
| **Logistic Regression** | 0.925 ± 0.035 | 0.820 | 0.824 | 0.849 | 0.786 | 0.836 | 0.897 | 5 | +0.060 |
| **Random Forest** | 0.903 ± 0.043 | 0.820 | 0.824 | 0.849 | 0.786 | 0.836 | 0.898 | 5 | +0.160 |
| **K-Nearest Neighbors (KNN)** | 0.898 ± 0.036 | 0.836 | 0.871 | 0.818 | 0.857 | 0.844 | 0.911 | 6 | +0.164 |
| **XGBoost** | 0.902 ± 0.033 | 0.803 | 0.800 | 0.849 | 0.750 | 0.824 | 0.858 | 5 | +0.151 |
| **Multi-Layer Perceptron (MLP)** | 0.894 ± 0.039 | 0.803 | 0.839 | 0.788 | 0.821 | 0.812 | 0.883 | 7 | +0.110 |

*\* Selected Clinical Champion Model: Serialized into `models/champion_model.joblib`.*

> For in-depth diagnostic curves, confusion matrices, and feature importance analyses, see [MODEL_EVALUATION.md](MODEL_EVALUATION.md).

### Algorithms Evaluated & Champion Rationale
1. **Support Vector Machine (SVM) — 🏆 Champion:** Achieved the highest holdout recall (87.88%), lowest fatal false negatives (4 missed cases), and strongest generalization (minimal 3.1% overfitting gap).
2. **Logistic Regression:** Solid linear baseline with calibrated probabilities and low overfitting (+0.060).
3. **Random Forest & XGBoost:** Strong ensemble learners that exhibited significant overfitting (+0.160 and +0.151 gaps) on the small clinical cohort ($N = 241$).
4. **K-Nearest Neighbors:** Strong discriminative AUC (0.911), but lower sensitivity (81.82%).
5. **Multi-Layer Perceptron:** Lowest sensitivity (78.79%) and highest false negatives (7 missed cases), confirming that small tabular clinical cohorts benefit more from regularized max-margin kernels than deep feedforward networks.

---

## 🤖 Phase 2: AI Healthcare Monitoring Agent

The Phase 2 agent provides an autonomous clinical decision-support and longitudinal patient monitoring workflow, reusing the Phase 1 champion pipeline (`models/champion_model.joblib`) without retraining:

```
PATIENT RECORD → INPUT VALIDATION → PATIENT DATA ANALYSIS → PREPROCESSING PIPELINE → TRAINED ML MODEL → RISK PREDICTION → RISK INTERPRETATION → HEALTH-RISK REPORT → MONITORING LOG
```

### Key Agent Capabilities
1. **Clinical Intake & Validation (`src/agent/validation.py`):** Enforces completeness across all 13 predictors; checks physiological bounds; detects crisis vitals; rejects invalid or corrupted records with informative errors.
2. **Pluggable Prediction (`src/agent/predictor.py`):** Abstract `BaseRiskPredictor` interface allows the ML model to be replaced in the future without altering agent workflow logic.
3. **Risk Interpretation & Attribution (`src/agent/interpretation.py`):** Categorizes risk into 4 tiers (Low, Moderate, High, Critical); analyzes positive risk drivers (e.g. ST depression, fluoroscopy vessels) and protective markers; formulates evidence-based AHA/ACC recommendations.
4. **Structured Health-Risk Reports (`src/agent/report_generator.py`):** Generates structured JSON reports for EHR systems and publication-grade Markdown consultation documents with mandatory medical disclaimers.
5. **Longitudinal Patient Monitoring (`src/agent/monitoring.py`):** Appends to persistent JSONL log (`data/monitoring/patient_monitoring_log.jsonl`); distinguishes between new assessments, previous assessments, and longitudinal monitoring trajectories.

> For complete architectural diagrams and component specifications, see [AI_AGENT_ARCHITECTURE.md](AI_AGENT_ARCHITECTURE.md).

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

### 2. Run Dataset Inspection & Preprocessing Pipeline
```bash
# Audit raw dataset schema and quality
python scripts/inspect_dataset.py

# Execute leak-free preprocessing and train/test partitioning
python scripts/run_preprocessing.py
```
> For complete preprocessing design details, see [PREPROCESSING.md](PREPROCESSING.md).

### 3. Run Machine Learning Benchmarking (Phase 1)
```bash
python scripts/train_and_compare_models.py
```
> Evaluates all 6 models, generates 5 clinical figures in `reports/figures/`, and serializes the champion pipeline to `models/champion_model.joblib`. See [MODEL_EVALUATION.md](MODEL_EVALUATION.md).

### 4. Run AI Monitoring Agent Demonstration (Phase 2)
```bash
python scripts/run_agent_demo.py
```
> Demonstrates new patient intake, 3-month follow-up longitudinal trend analysis, previous assessment retrieval, and invalid data rejection.

### 5. Launch the Streamlit Web UI (Phase 3)
```bash
streamlit run app.py
```
> Opens the healthcare decision-support dashboard in your browser. Navigate to **Patient Assessment** in the left sidebar to submit a clinical record and receive an AI-generated cardiovascular risk assessment.

### 6. Run Complete Test Suite
```bash
pytest tests/ -v
```
> Executes all 91 unit tests covering data integrity, preprocessing invariance, model pipeline, QA audits, agent workflows, and UI module conformance.

---

## 📜 Academic Integrity & Standards
- **No Synthetic Fabrication:** Real raw records are verified without data invention.
- **Leak-Free Methodology:** Preprocessing transformers are fit exclusively on training folds during cross-validation.
- **Reproducibility:** Seeded random states (`RANDOM_STATE = 42`) across all components.

