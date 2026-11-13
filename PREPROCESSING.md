# Phase 1: Clinical Data Preprocessing Architecture & Verification

**Project:** Heart Disease Risk Assessment Agent  
**Module:** `src/data/preprocessing.py` & `src/data/dataset.py`  
**Pipeline Serialized Artifact:** `models/preprocessor.joblib`  
**Dataset Partitions:** `data/processed/`  
**Status:** Completed & Tested (15 Unit Tests Passing)

---

## 1. Overview & Preprocessing Philosophy

In medical machine learning, preprocessing is not merely a formatting step—it dictates diagnostic validity, clinical generalizability, and model fairness. An improperly constructed pipeline can artificially inflate validation metrics through data leakage, resulting in algorithms that fail catastrophically in clinical deployment.

This document details the leak-free, reproducible preprocessing pipeline implemented for the **Heart Disease Risk Assessment Agent**.

### Core Guarantees
1. **Zero Data Leakage:** Preprocessing transformations (imputer statistics, standardizer means/variances, one-hot category dictionaries) are **fitted exclusively on training data** (`X_train`) and applied downstream to test data (`X_test`) and future patient records.
2. **Deduplication:** Dropping 723 exact clone rows from the raw Kaggle file eliminates train-test contamination, benchmarking models on the true underlying cohort of 302 unique patients.
3. **Domain-Specific Surrogate Code Handling:** Out-of-bounds categorical codes (`ca = 4` and `thal = 0`) are explicitly converted to `NaN` and imputed with mode statistics rather than being mistakenly treated as valid categories.
4. **Scikit-Learn Standard Pipeline Integration:** All feature transformations are compiled into a unified `ColumnTransformer` inside a `Pipeline`, enabling single-call serialization (`joblib.dump`) and seamless out-of-sample inference for Phase 2.

---

## 2. Preprocessing Architecture Pipeline

```mermaid
flowchart TD
    A[Raw Patient Record / CSV] --> B[Deduplication Gatekeeper\nRetains 302 Unique Patients]
    B --> C[Stratified 80/20 Split\nSeed: 42 | Preserves 54.3% Positive Ratio]
    
    subgraph Training Data Partition Only
        C -->|X_train: 241 cases| D[Pipeline.fit]
    end

    subgraph Preprocessing Pipeline Structure
        D --> E[Step 1: SurrogateMissingHandler\n- Maps ca=4 to NaN\n- Maps thal=0 to NaN]
        E --> F[Step 2: ColumnTransformer]
        
        F --> G1[Numerical Sub-Pipeline\nage, trestbps, chol, thalach, oldpeak\n- Median Imputer\n- StandardScaler: z = x-μ / σ]
        
        F --> G2[Nominal Categorical Sub-Pipeline\ncp, restecg, thal\n- Mode Imputer\n- OneHotEncoder drop='first']
        
        F --> G3[Ordinal Sub-Pipeline\nslope, ca\n- Mode Imputer\n- StandardScaler]
        
        F --> G4[Binary Sub-Pipeline\nsex, fbs, exang\n- Mode Imputer\n- Passthrough: preserves 0/1]
    end

    subgraph Out-of-Sample Inference Gate
        C -->|X_test: 61 cases| H[Pipeline.transform\nNO REFITTING]
        I[Unseen Phase 2 Patient Record] --> H
        F --> H
        H --> J[Transformed 17-Dimensional Feature Vector\nZero NaNs | Standardized Scales]
    end
```

---

## 3. Data Transformations: Step-by-Step Rationale

### 3.1. Deduplication (Addressing 70.54% Duplicate Rows)
* **Finding:** The raw Kaggle dataset contains 1,025 rows, of which 723 are exact duplicates. Every patient is duplicated 3, 4, or 8 times.
* **Risk:** A standard train/test split without deduplication places cloned copies of the same patient into both the training and test sets. The model memorizes training patients and scores an unrealistic ~98–100% on the test set, creating false confidence.
* **Resolution:** The data loader deduplicates the dataset to the 302 unique clinical cases before partitioning. The deduplicated dataset is stored at `data/processed/heart_deduplicated.csv`.

### 3.2. Stratified Partitioning (Train/Test Split)
* **Configuration:** 80% Training ($N = 241$), 20% Testing ($N = 61$).
* **Random State:** Fixed at `42` (`RANDOM_STATE` in `src/config.py`).
* **Stratification:** The target has a 54.30% positive class prevalence. Stratification preserves this exact proportion in both subsets:
  * **Training Set:** 131 positive cases out of 241 (54.36%).
  * **Test Set:** 33 positive cases out of 61 (54.10%).

### 3.3. Surrogate Missing Code Cleaning (`SurrogateMissingHandler`)
* **Medical Context:**
  * `ca`: Counts major coronary vessels (0, 1, 2, 3) highlighted by fluoroscopy. Value `4` is an out-of-range code used in the original Cleveland database for unrecorded examinations.
  * `thal`: Evaluates thalassemia defect status (1 = normal, 2 = fixed defect, 3 = reversible defect). Value `0` represents missing/unrecorded laboratory records.
* **Transformation:** Custom Scikit-Learn transformer converts `ca == 4` and `thal == 0` to `np.nan`.
* **Imputation Strategy:**
  * Missing `ca` values are imputed with the training set **mode**.
  * Missing `thal` values are imputed with the training set **mode**.

### 3.4. Continuous Feature Scaling (Numerical Pipeline)
* **Features:** `age`, `trestbps` (resting blood pressure), `chol` (serum cholesterol), `thalach` (maximum heart rate), `oldpeak` (ST depression).
* **Technique:**
  1. `SimpleImputer(strategy='median')`: Resilient to extreme physiological outliers (e.g., `chol = 564 mg/dl`).
  2. `StandardScaler()`: Standardizes features to zero mean ($\mu = 0$) and unit variance ($\sigma = 1$):
     $$z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
* **Justification:** Gradient-based algorithms (Logistic Regression, Multi-Layer Perceptron) and distance-based algorithms (K-Nearest Neighbors, Support Vector Machines) are heavily distorted when features operate on wildly different magnitudes (e.g., cholesterol in hundreds vs. ST depression in decimals).

### 3.5. Nominal Categorical Feature Encoding (Nominal Pipeline)
* **Features:** `cp` (chest pain: 0, 1, 2, 3), `restecg` (0, 1, 2), `thal` (1, 2, 3).
* **Technique:**
  1. `SimpleImputer(strategy='most_frequent')`.
  2. `OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')`.
* **Justification:**
  * Treating nominal categories as single integer scalars assumes an artificial ordinal hierarchy (e.g., chest pain type 3 is "three times" type 1), introducing false linear slopes.
  * `drop='first'` prevents multicollinearity (dummy variable trap) in linear and logistic models.
  * `handle_unknown='ignore'` guarantees that unexpected categories during Phase 2 live inference do not crash the pipeline, but are safely zero-encoded.

### 3.6. Ordinal & Discrete Features
* **Features:** `slope` (ST slope: 0, 1, 2), `ca` (vessels: 0, 1, 2, 3).
* **Technique:** Mode Imputation + `StandardScaler()`.
* **Justification:** Ordinal features have intrinsic directional meaning, but must be scaled to unit variance to prevent them from dominating distance computations in KNN and SVM.

### 3.7. Binary Indicator Features
* **Features:** `sex` (0/1), `fbs` (0/1), `exang` (0/1).
* **Technique:** Mode Imputation + `passthrough` (no scaling).
* **Justification:** Binary flags are already bounded within $[0, 1]$. Leaving them unscaled preserves biological interpretability and feature importance weights.

---

## 4. Transformed Feature Space

The original 13 features expand into **17 standardized clinical features**:

| Transformed Feature Name | Source Column | Feature Type | Pipeline Processing |
| :--- | :--- | :--- | :--- |
| `num__age` | `age` | Continuous | Median Impute + StandardScaler |
| `num__trestbps` | `trestbps` | Continuous | Median Impute + StandardScaler |
| `num__chol` | `chol` | Continuous | Median Impute + StandardScaler |
| `num__thalach` | `thalach` | Continuous | Median Impute + StandardScaler |
| `num__oldpeak` | `oldpeak` | Continuous | Median Impute + StandardScaler |
| `nom__cp_1.0` | `cp` | Nominal Dummy | Chest pain type 1 vs 0 |
| `nom__cp_2.0` | `cp` | Nominal Dummy | Chest pain type 2 vs 0 |
| `nom__cp_3.0` | `cp` | Nominal Dummy | Chest pain type 3 vs 0 |
| `nom__restecg_1.0` | `restecg` | Nominal Dummy | Resting ECG 1 vs 0 |
| `nom__restecg_2.0` | `restecg` | Nominal Dummy | Resting ECG 2 vs 0 |
| `nom__thal_2.0` | `thal` | Nominal Dummy | Thalassemia 2 vs 1 |
| `nom__thal_3.0` | `thal` | Nominal Dummy | Thalassemia 3 vs 1 |
| `ord__slope` | `slope` | Ordinal | Mode Impute + StandardScaler |
| `ord__ca` | `ca` | Discrete | Surrogate Cleaner + Impute + StandardScaler |
| `bin__sex` | `sex` | Binary | Mode Impute + Passthrough |
| `bin__fbs` | `fbs` | Binary | Mode Impute + Passthrough |
| `bin__exang` | `exang` | Binary | Mode Impute + Passthrough |

---

## 5. Preprocessing Verification & Validation Checks

Execution of `scripts/run_preprocessing.py` and automated tests in `tests/test_preprocessing.py` validated the following criteria:

```text
======================================================================
DATA PREPROCESSING VERIFICATION SUMMARY
======================================================================
[Check 1] Training Partition Shape:         (241, 17) -> Verified
[Check 2] Test Partition Shape:             (61, 17)  -> Verified
[Check 3] NaN Count in Processed Train:     0         -> Verified
[Check 4] NaN Count in Processed Test:      0         -> Verified
[Check 5] Train Target Positive Ratio:      54.36%    -> Verified (Stratified)
[Check 6] Test Target Positive Ratio:       54.10%    -> Verified (Stratified)
[Check 7] Preprocessor Fitted On Test?:     NO (Fit ONLY on X_train)
[Check 8] Pipeline Serialization:           models/preprocessor.joblib (Loaded & Tested)
[Check 9] Unseen Patient Record Inference:  1x17 vector generated cleanly
[Check 10] Pytest Verification:             15/15 unit tests passing
======================================================================
```

---

## 6. Artifact Inventory

1. `data/processed/heart_deduplicated.csv`: Cleaned 302-row cohort.
2. `data/processed/X_train_raw.csv` & `data/processed/X_test_raw.csv`: Raw feature partitions.
3. `data/processed/y_train.csv` & `data/processed/y_test.csv`: Target label vectors.
4. `data/processed/X_train_processed.csv`: Standardized 17-dimensional training matrix ($241 \times 17$).
5. `data/processed/X_test_processed.csv`: Standardized 17-dimensional test matrix ($61 \times 17$).
6. `models/preprocessor.joblib`: Serialized Scikit-Learn preprocessing pipeline for Phase 1 model training and Phase 2 agent inference.
