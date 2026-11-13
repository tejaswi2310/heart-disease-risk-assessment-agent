"""
Unit tests for data preprocessing module, train/test splitting, and pipeline execution.
Run with: pytest tests/test_preprocessing.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.config import (
    ALL_FEATURE_COLUMNS,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TARGET_COLUMN,
)
from src.data.dataset import (
    deduplicate_dataset,
    load_raw_data,
    separate_features_and_target,
    split_data,
)
from src.data.preprocessing import (
    SurrogateMissingHandler,
    create_preprocessor,
    get_feature_names,
    load_preprocessor,
    save_preprocessor,
    transform_patient_record,
)


@pytest.fixture(scope="module")
def raw_df():
    return load_raw_data(RAW_DATA_PATH)


@pytest.fixture(scope="module")
def clean_splits(raw_df):
    df_dedup, _ = deduplicate_dataset(raw_df)
    X, y = separate_features_and_target(df_dedup)
    (X_train, y_train), (X_test, y_test) = split_data(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=True
    )
    return X_train, X_test, y_train, y_test


def test_load_raw_data_valid(raw_df):
    assert isinstance(raw_df, pd.DataFrame)
    assert raw_df.shape == (1025, 14)
    assert TARGET_COLUMN in raw_df.columns


def test_deduplication(raw_df):
    df_dedup, dropped = deduplicate_dataset(raw_df)
    assert len(df_dedup) == 302
    assert dropped == 723
    assert df_dedup.duplicated().sum() == 0


def test_separate_features_and_target(raw_df):
    df_dedup, _ = deduplicate_dataset(raw_df)
    X, y = separate_features_and_target(df_dedup)
    assert TARGET_COLUMN not in X.columns
    assert len(X.columns) == 13
    assert len(y) == len(X)
    assert set(y.unique()) == {0, 1}


def test_stratified_split_balance(clean_splits):
    X_train, X_test, y_train, y_test = clean_splits
    assert len(X_train) == 241
    assert len(X_test) == 61

    # Check stratification preservation within ~1% tolerance
    train_pos_rate = y_train.mean()
    test_pos_rate = y_test.mean()
    assert abs(train_pos_rate - test_pos_rate) < 0.02


def test_surrogate_missing_handler():
    handler = SurrogateMissingHandler()
    df_dummy = pd.DataFrame({
        "age": [50, 60],
        "ca": [0, 4],     # 4 is surrogate missing
        "thal": [0, 2],   # 0 is surrogate missing
    })
    df_transformed = handler.fit_transform(df_dummy)
    assert np.isnan(df_transformed.loc[1, "ca"])
    assert np.isnan(df_transformed.loc[0, "thal"])
    assert df_transformed.loc[0, "ca"] == 0
    assert df_transformed.loc[1, "thal"] == 2


def test_pipeline_fit_train_transform_test(clean_splits):
    X_train, X_test, _, _ = clean_splits

    pipeline = create_preprocessor(scaling="standard", handle_surrogates=True)
    # Fit ONLY on training data
    pipeline.fit(X_train)

    # Transform training and test independently
    X_train_trans = pipeline.transform(X_train)
    X_test_trans = pipeline.transform(X_test)

    assert X_train_trans.shape == (241, 17)
    assert X_test_trans.shape == (61, 17)

    # Verify zero NaNs in transformed output
    assert not np.isnan(X_train_trans).any()
    assert not np.isnan(X_test_trans).any()


def test_feature_names_extracted(clean_splits):
    X_train, _, _, _ = clean_splits
    pipeline = create_preprocessor(scaling="standard")
    pipeline.fit(X_train)
    names = get_feature_names(pipeline)
    assert len(names) == 17
    assert any("num__age" in n for n in names)
    assert any("nom__cp" in n for n in names)


def test_single_patient_record_inference(clean_splits, tmp_path):
    X_train, _, _, _ = clean_splits
    pipeline = create_preprocessor()
    pipeline.fit(X_train)

    # Single patient dictionary (unseen test instance)
    patient = {
        "age": 58,
        "sex": 0,
        "cp": 1,
        "trestbps": 120,
        "chol": 210,
        "fbs": 0,
        "restecg": 0,
        "thalach": 160,
        "exang": 0,
        "oldpeak": 0.5,
        "slope": 2,
        "ca": 4,   # Should be safely handled by imputer
        "thal": 1,
    }

    vec = transform_patient_record(pipeline, patient)
    assert vec.shape == (1, 17)
    assert not np.isnan(vec).any()

    # Save and reload pipeline
    saved_path = tmp_path / "test_pipeline.joblib"
    save_preprocessor(pipeline, saved_path)
    loaded = load_preprocessor(saved_path)
    vec_reloaded = transform_patient_record(loaded, patient)
    np.testing.assert_allclose(vec, vec_reloaded)
