# Comprehensive Dataset Inspection & Clinical Analysis Report

**Project:** Heart Disease Risk Assessment Agent  
**Dataset Source:** [Kaggle Heart Disease Dataset (`johnsmith88/heart-disease-dataset`)](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)  
**Derived From:** Cleveland Clinic Foundation Heart Disease Database (UCI Machine Learning Repository)  
**Inspection Date:** 2026-11-13  
**Status:** Completed (Phase 1 Ingestion & Pre-modeling Audit)

---

## 1. Executive Summary

This report documents the rigorous pre-modeling data inspection conducted on the Kaggle Heart Disease dataset. The objective of this audit is to identify the real underlying data schema, assess distribution properties, detect data quality anomalies, unmask potential data leakage risks, and formulate an empirically grounded preprocessing strategy for Phase 1 (ML Model Benchmarking) and Phase 2 (AI Health Monitoring Agent).

### Key Takeaways
1. **Dimensions:** The dataset contains **1,025 rows** and **14 columns** (13 candidate predictors, 1 binary target).
2. **Critical Discovery — Massive Row Duplication (70.54%):** Out of 1,025 rows, **723 are exact duplicates** across all 14 features. The dataset comprises only **302 unique patient instances** sampled with replacement (187 patients repeated 3 times, 114 repeated 4 times, 1 repeated 8 times).
3. **Data Leakage Risk:** A naive random `train_test_split` on the raw 1,025 rows will contaminate test folds with training clones, leading to spurious accuracy (~98–100%) that collapses on unseen clinical data.
4. **Missing Values & Domain Anomalies:** While standard pandas checks show `0` explicit `NaN` values, domain validation reveals **surrogate missing codes**: `ca == 4` (18 raw rows, 4 unique) and `thal == 0` (7 raw rows, 2 unique), representing unrecorded fluoroscopy and thalassemia findings from the original UCI repository.
5. **Target Balance:** The target is well-balanced (**51.32% positive** in raw, **54.30% positive** in deduplicated data), eliminating the need for aggressive synthetic oversampling (e.g., SMOTE) while requiring Stratified K-Fold cross-validation.

---

## 2. Dataset Schema & Feature Taxonomy

| Column Name | Physical / Clinical Meaning | Data Type | Feature Class | Range / Valid Set | Units |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`age`** | Age of the patient | `int64` | Continuous Numerical | [29, 77] | Years |
| **`sex`** | Biological sex | `int64` | Binary Categorical | {0: Female, 1: Male} | Flag |
| **`cp`** | Chest pain type experienced | `int64` | Nominal Multi-class | {0, 1, 2, 3} | Category code |
| **`trestbps`** | Resting blood pressure on hospital admission | `int64` | Continuous Numerical | [94, 200] | mm Hg |
| **`chol`** | Serum cholesterol level | `int64` | Continuous Numerical | [126, 564] | mg/dl |
| **`fbs`** | Fasting blood sugar > 120 mg/dl | `int64` | Binary Categorical | {0: False, 1: True} | Flag |
| **`restecg`** | Resting electrocardiographic findings | `int64` | Nominal Multi-class | {0, 1, 2} | Category code |
| **`thalach`** | Maximum heart rate achieved during exercise | `int64` | Continuous Numerical | [71, 202] | bpm |
| **`exang`** | Exercise-induced angina | `int64` | Binary Categorical | {0: No, 1: Yes} | Flag |
| **`oldpeak`** | ST depression induced by exercise relative to rest | `float64` | Continuous Numerical | [0.0, 6.2] | mm |
| **`slope`** | Slope of peak exercise ST segment | `int64` | Ordinal Categorical | {0, 1, 2} | Category code |
| **`ca`** | Number of major vessels colored by fluoroscopy | `int64` | Discrete Ordinal | {0, 1, 2, 3, 4\*} | Vessel count |
| **`thal`** | Thalassemia blood defect status | `int64` | Nominal Multi-class | {0\*, 1, 2, 3} | Category code |
| **`target`** | Presence of angiographic heart disease (>50% diameter narrowing) | `int64` | Binary Target | {0: Absence, 1: Presence} | Diagnosis |

*\* Asterisks denote out-of-domain surrogate missing values.*

---

## 3. Detailed Inspection Findings

### 3.1. Row Duplication and Provenance Analysis

```
Total Rows: 1,025
Unique Patient Records: 302
Exact Duplicated Rows: 723 (70.54%)
```

- **Duplication Multiplicity Breakdown:**
  - 187 unique patients appear exactly **3 times** (561 rows).
  - 114 unique patients appear exactly **4 times** (456 rows).
  - 1 unique patient appears exactly **8 times** (8 rows).
  - Total: $187 \times 3 + 114 \times 4 + 1 \times 8 = 1,025\text{ rows}$.

```
+-------------------------------------------------------------+
| Raw Kaggle Dataset (1,025 rows)                            |
|                                                             |
|  [302 Original Patient Cases]                              |
|         + [723 Exact Replicate Copies]                      |
|                                                             |
|  WARNING: Random train_test_split causes test leakage!      |
|  RECOMMENDATION: Deduplicate or enforce patient grouping.  |
+-------------------------------------------------------------+
```

### 3.2. Target Column Distribution

The target variable represents whether angiographic coronary artery disease is diagnosed ($\ge 50\%$ vessel diameter narrowing):
- **Value 0:** No heart disease detected.
- **Value 1:** Clinically significant heart disease present.

| Distribution Metric | Value 0 (No Disease) | Value 1 (Disease Present) | Total |
| :--- | :--- | :--- | :--- |
| **Raw Dataset Count** | 499 (48.68%) | 526 (51.32%) | 1,025 |
| **Deduplicated Count** | 138 (45.70%) | 164 (54.30%) | 302 |

**Target Verdict:** Both raw and deduplicated cohorts show an evenly balanced distribution ($~54\% : 46\%$). No severe class imbalance re-weighting is needed, though Stratified cross-validation is required.

### 3.3. Numerical Features Descriptive Statistics

| Feature | Mean | Std Dev | Min | 25% | Median | 75% | Max | IQR | Skewness | IQR Outliers | $Z > 3$ Outliers |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`age`** | 54.43 | 9.07 | 29.0 | 48.0 | 56.0 | 61.0 | 77.0 | 13.0 | -0.25 | 0 | 0 |
| **`trestbps`** | 131.61 | 17.52 | 94.0 | 120.0 | 130.0 | 140.0 | 200.0 | 20.0 | 0.74 | 30 | 7 |
| **`chol`** | 246.00 | 51.59 | 126.0 | 211.0 | 240.0 | 275.0 | 564.0 | 64.0 | 1.07 | 16 | 13 |
| **`thalach`** | 149.11 | 23.01 | 71.0 | 132.0 | 152.0 | 166.0 | 202.0 | 34.0 | -0.51 | 4 | 4 |
| **`oldpeak`** | 1.07 | 1.18 | 0.0 | 0.0 | 0.8 | 1.8 | 6.2 | 1.8 | 1.27 | 7 | 7 |

#### Clinical Interpretation of Outliers:
1. **`chol = 564 mg/dl`:** A physiological extreme observed in severe familial hypercholesterolemia. Although clinically valid, in linear or distance-based models (KNN, SVM, Logistic Regression, MLP), such extreme magnitudes disproportionately skew loss gradients.
2. **`trestbps = 200 mm Hg`:** Represents stage 4 / malignant hypertensive crisis. Fully plausible in cardiology admissions.
3. **`oldpeak = 6.2 mm`:** Severe exercise-induced myocardial ischemia (heavy right skew, median 0.8 mm).

### 3.4. Categorical & Discrete Distributions

#### 1. `sex` (Biological Sex)
- `1` (Male): 713 rows (69.56% raw) / 206 unique (68.21%)
- `0` (Female): 312 rows (30.44% raw) / 96 unique (31.79%)

#### 2. `cp` (Chest Pain Type)
- `0` (Typical Angina): 497 rows (48.49%)
- `1` (Atypical Angina): 167 rows (16.29%)
- `2` (Non-anginal Pain): 284 rows (27.71%)
- `3` (Asymptomatic): 77 rows (7.51%)
*Notice: `cp` displays a strong positive correlation ($r = +0.4349$) with the presence of heart disease.*

#### 3. `fbs` (Fasting Blood Sugar > 120 mg/dl)
- `0` ($\le 120\text{ mg/dl}$): 872 rows (85.07%)
- `1` ($> 120\text{ mg/dl}$): 153 rows (14.93%)

#### 4. `restecg` (Resting Electrocardiogram)
- `0` (Normal): 497 rows (48.49%)
- `1` (ST-T wave abnormality): 513 rows (50.05%)
- `2` (Left ventricular hypertrophy): 15 rows (1.46%)

#### 5. `exang` (Exercise-Induced Angina)
- `0` (No): 680 rows (66.34%)
- `1` (Yes): 345 rows (33.66%)
*Notice: Strong inverse correlation ($r = -0.4380$) with target.*

#### 6. `slope` (Peak Exercise ST Slope)
- `0` (Upsloping): 74 rows (7.22%)
- `1` (Flat): 482 rows (47.02%)
- `2` (Downsloping): 469 rows (45.76%)

#### 7. `ca` (Number of Major Vessels Colored by Fluoroscopy)
- `0`: 578 rows (56.39%)
- `1`: 226 rows (22.05%)
- `2`: 134 rows (13.07%)
- `3`: 69 rows (6.73%)
- **`4` (Surrogate Missing): 18 rows (1.76% raw, 4 unique cases)**

#### 8. `thal` (Thalassemia Status)
- **`0` (Surrogate Missing): 7 rows (0.68% raw, 2 unique cases)**
- `1` (Normal): 64 rows (6.24%)
- `2` (Fixed defect): 544 rows (53.07%)
- `3` (Reversible defect): 410 rows (40.00%)

---

## 4. Feature Correlation with Target

| Feature | Pearson Correlation ($r$) | Association with Heart Disease |
| :--- | :--- | :--- |
| **`oldpeak`** | **-0.4384** | Higher ST depression is inversely correlated in this encoding |
| **`exang`** | **-0.4380** | Exercise angina strongly correlates with severe pathology |
| **`cp`** | **+0.4349** | Certain chest pain presentations strongly elevate disease probability |
| **`thalach`** | **+0.4229** | Higher max achievable heart rate observed in non-diseased baseline |
| **`ca`** | **-0.3821** | Vessel fluoroscopy opacification count |
| **`slope`** | **+0.3455** | ST segment profile slope |
| **`thal`** | **-0.3378** | Thalassemia pathology code |
| **`sex`** | **-0.2809** | Gender-specific prevalence differences |
| **`age`** | **-0.2293** | Mild demographic trend |
| **`trestbps`**| **-0.1388** | Mild admission baseline correlation |
| **`restecg`** | **+0.1345** | Electrocardiographic abnormality signal |
| **`chol`** | **-0.0999** | Weak linear correlation |
| **`fbs`** | **-0.0411** | Lowest direct linear correlation |

---

## 5. Data Quality Issues & Preprocessing Requirements

| Issue Identified | Evidence in Dataset | Risk / Consequence | Required Preprocessing Solution |
| :--- | :--- | :--- | :--- |
| **Massive Row Duplication** | 723 duplicated rows (70.54% of dataset); only 302 unique patients. | Splitting without deduplication contaminates test sets, giving fake ~98% accuracy. | **Mandatory Deduplication:** Deduplicate dataset to 302 unique patients prior to train/test partitioning, or benchmark both pipelines to document leakage impact. |
| **Surrogate Missing Codes** | `ca == 4` (18 occurrences / 4 unique patients) and `thal == 0` (7 occurrences / 2 unique patients). | `4` in `ca` and `0` in `thal` are biologically nonexistent and distort model weights. | Convert `ca=4` and `thal=0` to `NaN`, then impute using median/mode or a `SimpleImputer` / `KNNImputer`. |
| **Heterogeneous Feature Scales** | `chol` reaches 564, while `oldpeak` is between 0 and 6.2, and `sex` is 0/1. | Distance and gradient-based models (KNN, SVM, Logistic Regression, MLP, ANN) will be dominated by `chol` and `trestbps`. | Apply **`StandardScaler`** or **`RobustScaler`** to numerical continuous columns, fitted strictly on training folds. |
| **Nominal Multi-Class Features** | `cp` (0–3), `restecg` (0–2), `thal` (1–3) stored as arbitrary integers. | Linear models and neural networks will interpret categories as ordinal distances ($3 > 2 > 1$). | Apply **One-Hot Encoding** (`OneHotEncoder(drop='first', handle_unknown='ignore')`) to non-ordinal categories. |
| **Extreme Outliers** | `chol = 564 mg/dl`, `trestbps = 200 mm Hg`, `oldpeak = 6.2 mm`. | Outliers distort linear decision boundaries and gradient descent updates. | Employ **`RobustScaler`** (IQR-based scaling) or assess impact on Tree models (Random Forest, XGBoost) which are naturally invariant to monotonic outliers. |
| **Validation Leakage Prevention** | Any transformer fitted on the entire dataset leaks test distribution into training. | Optimistic bias during validation. | Implement rigorous **`sklearn.pipeline.Pipeline`** paired with **`StratifiedKFold` (5-fold or 10-fold)**. |

---

## 6. Generated Visualizations

The following exploratory visualizations were generated and persisted in `reports/figures/`:
1. `target_distribution.png`: Direct comparison between raw ($N=1,025$) and deduplicated ($N=302$) class balances.
2. `duplicate_analysis.png`: Bar chart demonstrating exact patient record repetition counts (3x, 4x, 8x).
3. `correlation_with_target.png`: Horizontal bar chart of Pearson correlations between predictors and diagnosis.
4. `numerical_distributions.png`: Density distributions and histograms of continuous features stratified by target status.

---

## 7. Next Steps for Phase 1

1. **Pipeline Construction:** Build reproducible `scikit-learn` `ColumnTransformer` pipelines combining median/mode imputation, one-hot encoding, and robust scaling.
2. **Model Training & Comparison:**
   - Logistic Regression
   - Random Forest
   - XGBoost
   - Support Vector Machine (SVM)
   - K-Nearest Neighbors (KNN)
   - Multi-Layer Perceptron (MLP)
   - Artificial Neural Network (ANN)
3. **Comprehensive Metric Reporting:** Calculate Accuracy, Precision, Recall, F1-score, ROC-AUC, and Confusion Matrices across both standard cross-validation and holdout validation sets.
