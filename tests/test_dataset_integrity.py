"""
Unit tests for raw dataset integrity and schema verification.
Run with: pytest tests/test_dataset_integrity.py
"""

from pathlib import Path
import pandas as pd
import pytest

from src.config import RAW_DATA_PATH, ALL_FEATURE_COLUMNS, TARGET_COLUMN


@pytest.fixture(scope="module")
def raw_dataset():
    assert RAW_DATA_PATH.exists(), f"Raw dataset not found at {RAW_DATA_PATH}"
    return pd.read_csv(RAW_DATA_PATH)


def test_raw_data_file_exists():
    assert RAW_DATA_PATH.exists(), f"Raw data file missing at {RAW_DATA_PATH}"
    assert RAW_DATA_PATH.stat().st_size > 0, "Raw dataset file is empty"


def test_dataset_dimensions(raw_dataset):
    assert raw_dataset.shape == (1025, 14), f"Unexpected dataset shape: {raw_dataset.shape}"


def test_expected_columns(raw_dataset):
    expected_cols = set(ALL_FEATURE_COLUMNS + [TARGET_COLUMN])
    actual_cols = set(raw_dataset.columns)
    assert actual_cols == expected_cols, f"Mismatch in columns: {actual_cols ^ expected_cols}"


def test_no_explicit_nan_values(raw_dataset):
    null_count = raw_dataset.isnull().sum().sum()
    assert null_count == 0, f"Found {null_count} unexpected NaN values in raw data"


def test_duplicate_row_structure(raw_dataset):
    num_duplicates = raw_dataset.duplicated().sum()
    unique_count = len(raw_dataset.drop_duplicates())
    assert num_duplicates == 723, f"Expected 723 duplicate rows, got {num_duplicates}"
    assert unique_count == 302, f"Expected 302 unique patient records, got {unique_count}"


def test_target_distribution(raw_dataset):
    target_values = set(raw_dataset[TARGET_COLUMN].unique())
    assert target_values == {0, 1}, f"Unexpected target classes: {target_values}"
    counts = raw_dataset[TARGET_COLUMN].value_counts().to_dict()
    assert counts[1] == 526, f"Expected 526 class 1 instances, got {counts.get(1)}"
    assert counts[0] == 499, f"Expected 499 class 0 instances, got {counts.get(0)}"


def test_surrogate_missing_codes_identified(raw_dataset):
    # ca == 4 is out-of-bounds surrogate code for missing fluoroscopy result
    ca_out_of_bounds = (raw_dataset["ca"] == 4).sum()
    assert ca_out_of_bounds == 18, f"Expected 18 rows with ca==4, got {ca_out_of_bounds}"

    # thal == 0 is out-of-bounds surrogate code for unrecorded thalassemia
    thal_out_of_bounds = (raw_dataset["thal"] == 0).sum()
    assert thal_out_of_bounds == 7, f"Expected 7 rows with thal==0, got {thal_out_of_bounds}"
