# Phase 1: Quality Assurance & Validation Audit Report

**Project:** Heart Disease Risk Assessment Agent  
**Audit Scope:** Data Pipeline, Cross-Validation, Preprocessing Invariance, Model Evaluation, and Serialization  
**Validation Suite:** 33 Automated Unit Tests (100% Passing)  
**Audit Date:** 2026-11-14  
**Audit Status:** APPROVED (Production-Ready for Phase 2)

---

## 1. Executive Summary

A comprehensive quality assurance (QA) and verification pass was conducted on the Phase 1 Machine Learning pipeline. The objective of this audit was to mathematically and empirically verify that the implementation is free from data leakage, train-test contamination, preprocessing discrepancies, and serialization bugs, while ensuring robust clinical validation and error handling for out-of-sample inference.

---

## 2. 15-Point QA Audit Checklist

| Audit Item | Status | Verification Evidence / Finding | Resolution / Verification |
| :--- | :---: | :--- | :--- |
| **1. Data Leakage** | **PASSED** | Checked `X_train_raw.csv` and `X_test_raw.csv`. Combined duplicate row count is exactly `0`. Deduplicated 302-patient cohort eliminated the 723 clone rows. | Zero leakage confirmed. Verified by `test_train_test_partitions_disjoint`. |
| **2. Incorrect Train/Test Splitting** | **PASSED** | 80/20 partitioning with `stratify=y` and `random_state=42`. Training set positive ratio: 54.36%; Test set positive ratio: 54.10%. | Target balance preserved across partitions within 0.26% margin. Verified by `test_stratified_split_balance`. |
| **3. Preprocessor Fitted on Test Data** | **PASSED** | Audited `StandardScaler.mean_` inside `models/champion_model.joblib`. Exactly matches training means: `[54.195, 131.315, 247.631, 150.826, 0.995]`. Test means are offset (`[+0.124, +0.085, -0.104, -0.277, +0.206]`). | Confirmed scaler was fit ONLY on training data. Verified by `test_scaler_fitted_strictly_on_training_data`. |
| **4. Incorrect Target Handling** | **PASSED** | Binary diagnosis $\{0, 1\}$ mapped with $1 = \text{Disease Present}$, $0 = \text{Healthy}$. Confusion matrix labels oriented as `[0, 1]` ($TN, FP, FN, TP$). | Verified clinical metrics correctly assign sensitivity to disease detection. |
| **5. Incorrect Categorical Encoding** | **PASSED** | Multi-class nominal variables (`cp`, `restecg`, `thal`) encoded using `OneHotEncoder(drop='first', handle_unknown='ignore')`. | Dummy variable trap eliminated; unseen categories gracefully handled. |
| **6. Incorrect Scaling** | **PASSED** | Continuous features (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`) and ordinal (`slope`, `ca`) scaled to unit variance. Binary flags (`sex`, `fbs`, `exang`) unscaled. | Preserves binary interpretability while preventing distance distortion in SVM/KNN. |
| **7. Incorrect ROC-AUC Calculation** | **PASSED** | Calculated via true continuous positive class probabilities ($P(y=1)$) from `predict_proba` and calibrated Platt logistic scores. | Verified by `evaluate_model_performance`. |
| **8. Test-Set Contamination in Tuning** | **PASSED** | Audited `train_and_tune_model()`. `GridSearchCV` receives exclusively `X_train_processed` and `y_train`. Holdout test partition evaluated strictly once. | Zero test-set leakage during hyperparameter optimization. |
| **9. Overfitting** | **PASSED** | Overfitting gap calculated across all 6 models. Tree models overfit (+16.0% for RF, +15.1% for XGBoost). Champion SVM showed minimal gap (+3.13%). | Overfitting penalized in champion selection formula. Verified by `compute_overfitting_analysis`. |
| **10. Reproducibility Issues** | **PASSED** | All random seeds pinned to `RANDOM_STATE = 42`. Repeated inference yields identical probability scores across multiple calls. | Verified by `test_deterministic_predictions`. |
| **11. Incorrect Metric Calculations** | **PASSED** | Audited Sensitivity ($TP/(TP+FN)$), Specificity ($TN/(TN+FP)$), Precision ($TP/(TP+FP)$), F1, Brier score. | All formulas adhere to clinical diagnostic standards. |
| **12. Inconsistent Inference Preprocessing** | **PASSED** | Raw patient input passes through identical `SurrogateMissingHandler`, `SimpleImputer`, `OneHotEncoder`, and `StandardScaler` via the unified `Pipeline`. | Preprocessing equivalence guaranteed. |
| **13. Missing Error Handling** | **RESOLVED** | Identified that passing raw scalar dictionary directly crashed pandas index creation. | Upgraded `SurrogateMissingHandler.transform()` and built `src/models/predict.py`. |
| **14. Hard-Coded Dataset Assumptions** | **RESOLVED** | Added dictionary key order invariance and physiological range validators for all 13 features. | Verified by `test_column_order_invariance`. |
| **15. Incorrect Model Serialization** | **PASSED** | Champion model and full preprocessing pipeline serialized into single file `models/champion_model.joblib`. | Loads and predicts without auxiliary dependencies. |

---

## 3. Problems Found & Fixes Implemented

### Problem 1: Multiprocessing Process Stall on Windows Sandbox
* **Discovery:** During model training, `GridSearchCV` with `n_jobs=-1` hung on Windows due to sandbox process isolation blocking worker IPC communication.
* **Fix:** Adjusted `n_jobs=1` and set `os.environ["LOKY_MAX_CPU_COUNT"] = "1"` at script initialization. Because the training cohort is compact ($N=241$), sequential training runs in ~5 seconds with zero process overhead.

### Problem 2: Raw Dictionary Input Handling in Scikit-Learn Pipeline
* **Discovery:** When a raw dictionary `{"age": 55, ...}` was fed to `pipeline.predict()`, `pd.DataFrame(dict)` raised a `ValueError: If using all scalar values, you must pass an index`.
* **Fix:** Updated `SurrogateMissingHandler.transform()` to inspect input types:
  ```python
  if isinstance(X, pd.DataFrame):
      X_df = X.copy()
  elif isinstance(X, dict):
      X_df = pd.DataFrame([X])
  elif isinstance(X, pd.Series):
      X_df = X.to_frame().T
  elif isinstance(X, list) and len(X) > 0 and isinstance(X[0], dict):
      X_df = pd.DataFrame(X)
  ```
  The pipeline now natively accepts dictionaries, lists of dictionaries, Series, or DataFrames.

### Problem 3: Silent Acceptance of Physiologically Impossible Inputs
* **Discovery:** Raw pipeline would blindly process physically impossible vitals (e.g., negative blood pressure or impossible ages) without warning clinicians.
* **Fix:** Implemented `src/models/predict.py` with `validate_clinical_bounds()` and `predict_patient_risk()`. It enforces hard physiological constraints (raising clear `ValueError` exceptions for impossible entries) and generates warning alerts for crisis ranges (e.g., hypertensive crisis $\ge 180\text{ mm Hg}$).

---

## 4. End-to-End Inference Verification

The verification verifies that:
$$\text{Raw Patient Data} \longrightarrow \text{Preprocessing Pipeline} \longrightarrow \text{Trained Model} \longrightarrow \text{Prediction}$$
uses the exact same logic and transformation statistics as training:

```python
from src.models.predict import predict_patient_risk
import joblib

# Load serialized champion pipeline
pipeline = joblib.load("models/champion_model.joblib")

# Unseen raw clinical record with surrogate missing codes (ca=4, thal=0)
patient = {
    "age": 62, "sex": 1, "cp": 2, "trestbps": 185, "chol": 260,
    "fbs": 0, "restecg": 1, "thalach": 145, "exang": 1,
    "oldpeak": 2.2, "slope": 1, "ca": 4, "thal": 0
}

result = predict_patient_risk(pipeline, patient)
```

**Verification Output:**
```json
{
  "predicted_class": 1,
  "diagnosis_label": "Presence of Significant Coronary Artery Disease",
  "risk_probability": 0.9422,
  "risk_percentage": "94.2%",
  "risk_tier": "CRITICAL",
  "clinical_action": "Emergency medical consultation & intensive cardiovascular monitoring",
  "decision_threshold_used": 0.5,
  "validation_warnings": [
    "Hypertensive crisis range (trestbps=185 mm Hg). Requires immediate emergency attention.",
    "Vessel count 'ca=4' is an unrecorded surrogate code. Value will be statistically imputed.",
    "Thalassemia status 'thal=0' is an unrecorded surrogate code. Value will be statistically imputed."
  ]
}
```

---

## 5. Automated Test Suite Summary

The automated test suite in `tests/` contains **33 comprehensive tests across 4 test modules**:

```text
tests/test_dataset_integrity.py  .......             [ 7 passed]
tests/test_preprocessing.py      ........            [ 8 passed]
tests/test_model_pipeline.py     .....               [ 5 passed]
tests/test_qa_validation.py      .............       [13 passed]
======================= 33 passed in 3.62s =======================
```

---

## 6. Remaining Clinical & Algorithmic Limitations

1. **Cohort Sample Size:** The underlying deduplicated cohort comprises 302 unique patients. While representative of benchmark Cleveland clinic cohorts, external validation on larger modern cohorts (e.g., UK Biobank or NHANES) is recommended before acute hospital deployment.
2. **Binary Cutoff at 50% Stenosis:** The ground truth label corresponds to $\ge 50\%$ vessel diameter narrowing via angiography. It does not grade microvascular ischemia or non-obstructive coronary plaques.
3. **Platt Probability Calibration on Small Samples:** For SVM, probability calibration via Platt scaling provides smooth risk estimates, but should be interpreted alongside clinical judgment and diagnostic imaging.
