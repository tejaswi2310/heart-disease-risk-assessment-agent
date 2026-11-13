"""
Orchestration script for Phase 1 Data Preprocessing.
Executes leak-free preprocessing, train/test partitioning, pipeline fitting, and transformation verification.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.config import (
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    RAW_DATA_PATH,
    SPLIT_CONFIG,
    TARGET_COLUMN,
)
from src.data.dataset import (
    deduplicate_dataset,
    load_raw_data,
    save_split_artifacts,
    separate_features_and_target,
    split_data,
)
from src.data.preprocessing import (
    create_preprocessor,
    get_feature_names,
    load_preprocessor,
    save_preprocessor,
    transform_patient_record,
)


def run_pipeline():
    print("=" * 70)
    print("HEART DISEASE RISK ASSESSMENT - PHASE 1: DATA PREPROCESSING PIPELINE")
    print("=" * 70)

    # 1. Load Raw Dataset
    print(f"\n[Step 1] Ingesting raw dataset from {RAW_DATA_PATH}...")
    df_raw = load_raw_data(RAW_DATA_PATH)
    print(f"  -> Raw dataset loaded: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns.")

    # 2. Deduplicate to eliminate train-test leakage
    print("\n[Step 2] Deduplicating patient cohort to prevent test set contamination...")
    df_dedup, dropped_dups = deduplicate_dataset(df_raw)
    print(f"  -> Dropped {dropped_dups} duplicate rows ({dropped_dups / len(df_raw) * 100:.2f}%).")
    print(f"  -> Clean clinical cohort retained: {len(df_dedup)} unique patient cases.")

    # Save deduplicated data
    dedup_path = DATA_PROCESSED_DIR / "heart_deduplicated.csv"
    dedup_path.parent.mkdir(parents=True, exist_ok=True)
    df_dedup.to_csv(dedup_path, index=False)
    print(f"  -> Saved deduplicated cohort to: {dedup_path}")

    # 3. Separate Features and Target
    print("\n[Step 3] Separating features X and target y...")
    X, y = separate_features_and_target(df_dedup, target_col=TARGET_COLUMN)
    print(f"  -> Features matrix X: {X.shape[0]} samples, {X.shape[1]} features.")
    print(f"  -> Target vector y: {y.shape[0]} labels. Class 1: {(y==1).sum()} ({(y==1).mean()*100:.2f}%), Class 0: {(y==0).sum()} ({(y==0).mean()*100:.2f}%).")

    # 4. Stratified Train / Test Split
    test_size = SPLIT_CONFIG.get("test_size", 0.20)
    print(f"\n[Step 4] Performing Stratified Train/Test Split ({int((1-test_size)*100)}% Train / {int(test_size*100)}% Test)...")
    (X_train, y_train), (X_test, y_test) = split_data(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=True,
    )
    print(f"  -> Training Set: {X_train.shape[0]} samples (Positives: {(y_train==1).sum()}, {(y_train==1).mean()*100:.2f}%)")
    print(f"  -> Test Set:     {X_test.shape[0]} samples (Positives: {(y_test==1).sum()}, {(y_test==1).mean()*100:.2f}%)")

    # 5. Build Reusable Preprocessing Pipeline
    print("\n[Step 5] Building scikit-learn ColumnTransformer pipeline...")
    preprocessor = create_preprocessor(scaling="standard", handle_surrogates=True)
    print("  -> Pipeline architecture:")
    print("     • Step 1: SurrogateMissingHandler (maps ca=4 and thal=0 to NaN)")
    print("     • Step 2: ColumnTransformer:")
    print("         - Continuous ['age','trestbps','chol','thalach','oldpeak']: Median Impute + StandardScaler")
    print("         - Nominal Multi-Class ['cp','restecg','thal']: Mode Impute + OneHotEncoder(drop='first')")
    print("         - Ordinal ['slope','ca']: Mode Impute + StandardScaler")
    print("         - Binary ['sex','fbs','exang']: Mode Impute + Passthrough")

    # 6. Fit Pipeline ONLY on Training Data (Zero Leakage)
    print("\n[Step 6] FITTING PREPROCESSOR ONLY ON X_TRAIN (Zero Leakage)...")
    preprocessor.fit(X_train)
    print("  -> Preprocessor fitted successfully on training data only.")

    # 7. Transform Training and Test Data
    print("\n[Step 7] Transforming X_train and unseen X_test...")
    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    feature_names = get_feature_names(preprocessor)
    print(f"  -> Transformed feature count: {len(feature_names)}")
    print(f"  -> Output feature names: {feature_names}")

    df_X_train_trans = pd.DataFrame(X_train_trans, columns=feature_names, index=X_train.index)
    df_X_test_trans = pd.DataFrame(X_test_trans, columns=feature_names, index=X_test.index)

    # 8. Verification Checks
    print("\n[Step 8] Running data transformation verification checks:")
    train_nans = int(df_X_train_trans.isnull().sum().sum())
    test_nans = int(df_X_test_trans.isnull().sum().sum())
    print(f"  [Check] NaN count in X_train_processed: {train_nans} (Expected: 0)")
    print(f"  [Check] NaN count in X_test_processed:  {test_nans} (Expected: 0)")
    assert train_nans == 0, "Error: NaNs found in processed training data!"
    assert test_nans == 0, "Error: NaNs found in processed test data!"

    train_shape_match = (df_X_train_trans.shape[0] == len(X_train)) and (df_X_train_trans.shape[1] == len(feature_names))
    test_shape_match = (df_X_test_trans.shape[0] == len(X_test)) and (df_X_test_trans.shape[1] == len(feature_names))
    print(f"  [Check] X_train shape: {df_X_train_trans.shape} - Match: {train_shape_match}")
    print(f"  [Check] X_test shape:  {df_X_test_trans.shape} - Match: {test_shape_match}")
    assert train_shape_match and test_shape_match, "Error: Dimension mismatch in transformed matrices!"

    # 9. Save Artifacts
    print("\n[Step 9] Persisting split artifacts and serialized preprocessor...")
    save_split_artifacts(X_train, X_test, y_train, y_test)
    df_X_train_trans.to_csv(DATA_PROCESSED_DIR / "X_train_processed.csv", index=False)
    df_X_test_trans.to_csv(DATA_PROCESSED_DIR / "X_test_processed.csv", index=False)

    saved_pipeline_path = save_preprocessor(preprocessor)
    print(f"  -> Saved X_train_raw, X_test_raw, y_train, y_test to {DATA_PROCESSED_DIR}")
    print(f"  -> Saved X_train_processed.csv and X_test_processed.csv")
    print(f"  -> Serialized fitted pipeline to: {saved_pipeline_path}")

    # 10. Simulate Out-of-Sample Patient Inference (Phase 2 Readiness)
    print("\n[Step 10] Simulating Phase 2 unseen patient record inference...")
    loaded_pipe = load_preprocessor(saved_pipeline_path)
    sample_patient = {
        "age": 62,
        "sex": 1,
        "cp": 2,
        "trestbps": 140,
        "chol": 260,
        "fbs": 0,
        "restecg": 1,
        "thalach": 145,
        "exang": 1,
        "oldpeak": 2.2,
        "slope": 1,
        "ca": 4,     # Surrogate code: will be imputed by pipeline
        "thal": 2,
    }
    sample_transformed = transform_patient_record(loaded_pipe, sample_patient)
    print(f"  -> Raw Patient Input: age={sample_patient['age']}, chol={sample_patient['chol']}, ca={sample_patient['ca']} (surrogate code)")
    print(f"  -> Preprocessed 1x{sample_transformed.shape[1]} vector successfully produced without refitting.")
    print(f"  -> Sample preprocessed output values (first 5 features): {sample_transformed[0, :5].round(3)}")

    print("\n" + "=" * 70)
    print("PHASE 1 DATA PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
