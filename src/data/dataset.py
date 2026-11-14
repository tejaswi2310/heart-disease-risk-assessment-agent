"""
Dataset loading, validation, deduplication, and partitioning module.
Provides leak-free, stratified dataset operations for the Heart Disease Risk Assessment project.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    RAW_DATA_PATH,
    ALL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    RANDOM_STATE,
    DATA_PROCESSED_DIR,
)


def load_raw_data(filepath: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Loads and validates the raw Heart Disease dataset from disk without altering it.

    Parameters:
        filepath: Path to the raw CSV file. Defaults to RAW_DATA_PATH from config.

    Returns:
        pd.DataFrame: A copy of the raw dataset.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If required columns are missing or the file is empty.
    """
    path = Path(filepath) if filepath else RAW_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset file not found at: {path.resolve()}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"Dataset at {path.resolve()} is empty.")

    expected_cols = set(ALL_FEATURE_COLUMNS + [TARGET_COLUMN])
    actual_cols = set(df.columns)
    missing_cols = expected_cols - actual_cols
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    return df.copy()


def deduplicate_dataset(df: pd.DataFrame, keep: str = "first") -> Tuple[pd.DataFrame, int]:
    """
    Deduplicates the dataset to extract the true underlying patient cohort (302 instances),
    preventing train-test data contamination.

    Parameters:
        df: Input DataFrame.
        keep: Duplication retention policy ('first', 'last', False). Default is 'first'.

    Returns:
        Tuple[pd.DataFrame, int]: (deduplicated DataFrame, count of duplicate rows dropped).
    """
    initial_rows = len(df)
    df_dedup = df.drop_duplicates(keep=keep).copy()
    dropped_count = initial_rows - len(df_dedup)
    return df_dedup, dropped_count


def separate_features_and_target(
    df: pd.DataFrame, target_col: str = TARGET_COLUMN
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Cleanly separates features X and target vector y.

    Parameters:
        df: Input DataFrame containing features and target.
        target_col: Name of the target column.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features DataFrame X and Target Series y.

    Raises:
        KeyError: If target_col is not present in df.
        ValueError: If target values are not binary {0, 1}.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

    y = df[target_col].copy()
    X = df.drop(columns=[target_col]).copy()

    # Verify target domain
    unique_targets = set(y.unique())
    if not unique_targets.issubset({0, 1}):
        raise ValueError(f"Unexpected target values: {unique_targets}. Expected binary {0, 1}.")

    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    val_size: Optional[float] = None,
    random_state: int = RANDOM_STATE,
    stratify: bool = True,
) -> Union[
    Tuple[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]],
    Tuple[
        Tuple[pd.DataFrame, pd.Series],
        Tuple[pd.DataFrame, pd.Series],
        Tuple[pd.DataFrame, pd.Series],
    ],
]:
    """
    Partitions features and target into stratified, leak-free splits with fixed random seed.

    Parameters:
        X: Features DataFrame.
        y: Target Series.
        test_size: Proportion of dataset allocated to test set. Default is 0.20 (20%).
        val_size: Optional proportion of remaining training data allocated to validation set.
        random_state: Fixed seed for exact reproducibility.
        stratify: Whether to preserve class proportions across partitions. Default is True.

    Returns:
        If val_size is None:
            ((X_train, y_train), (X_test, y_test))
        If val_size is provided:
            ((X_train, y_train), (X_val, y_val), (X_test, y_test))
    """
    stratify_target = y if stratify else None

    # Initial split: Separate test set from training/dev set
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )

    if val_size is not None:
        stratify_val = y_train_full if stratify else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full,
            y_train_full,
            test_size=val_size,
            random_state=random_state,
            stratify=stratify_val,
        )
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    return (X_train_full, y_train_full), (X_test, y_test)


def save_split_artifacts(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: Optional[Union[str, Path]] = None,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
) -> None:
    """
    Persists data partitions to CSV files in the processed data directory.
    """
    out_dir = Path(output_dir) if output_dir else DATA_PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(out_dir / "X_train_raw.csv", index=False)
    X_test.to_csv(out_dir / "X_test_raw.csv", index=False)
    y_train.to_frame(name=TARGET_COLUMN).to_csv(out_dir / "y_train.csv", index=False)
    y_test.to_frame(name=TARGET_COLUMN).to_csv(out_dir / "y_test.csv", index=False)

    if X_val is not None and y_val is not None:
        X_val.to_csv(out_dir / "X_val_raw.csv", index=False)
        y_val.to_frame(name=TARGET_COLUMN).to_csv(out_dir / "y_val.csv", index=False)

