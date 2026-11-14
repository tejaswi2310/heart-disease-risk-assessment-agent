"""
Reusable and leak-free preprocessing pipeline module for Heart Disease Risk Assessment.
Handles surrogate missing codes, categorical one-hot encoding, and numerical scaling.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

from src.config import (
    ALL_FEATURE_COLUMNS,
    BINARY_CATEGORICAL_FEATURES,
    FEATURE_METADATA,
    MULTI_CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    PREPROCESSOR_PATH,
)


class SurrogateMissingHandler(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn transformer to convert domain-specific surrogate missing codes
    to true NaN values prior to statistical imputation.

    Anomalies Handled:
        - `ca == 4`: Out-of-range vessel count (surrogate missing code in Cleveland dataset).
        - `thal == 0`: Out-of-range thalassemia code (surrogate missing code in Cleveland dataset).
    """

    def __init__(self, ca_col: str = "ca", thal_col: str = "thal"):
        self.ca_col = ca_col
        self.thal_col = thal_col
        self.feature_names_in_: Optional[List[str]] = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        elif hasattr(X, "shape"):
            self.feature_names_in_ = [f"feat_{i}" for i in range(X.shape[1])]
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray, dict, pd.Series, list]) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        elif isinstance(X, dict):
            X_df = pd.DataFrame([X])
        elif isinstance(X, pd.Series):
            X_df = X.to_frame().T
        elif isinstance(X, list) and len(X) > 0 and isinstance(X[0], dict):
            X_df = pd.DataFrame(X)
        else:
            cols = self.feature_names_in_ if self.feature_names_in_ else ALL_FEATURE_COLUMNS
            X_df = pd.DataFrame(X, columns=cols)

        # Validate that expected columns are present
        expected_cols = set(self.feature_names_in_ if self.feature_names_in_ else ALL_FEATURE_COLUMNS)
        missing_cols = expected_cols - set(X_df.columns)
        if missing_cols:
            raise ValueError(f"Input patient record is missing required feature(s): {sorted(list(missing_cols))}")

        # Replace surrogate code 4 in ca with NaN
        if self.ca_col in X_df.columns:
            X_df[self.ca_col] = X_df[self.ca_col].replace(4, np.nan)

        # Replace surrogate code 0 in thal with NaN
        if self.thal_col in X_df.columns:
            X_df[self.thal_col] = X_df[self.thal_col].replace(0, np.nan)

        return X_df

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.array(input_features)
        if self.feature_names_in_ is not None:
            return np.array(self.feature_names_in_)
        return np.array(ALL_FEATURE_COLUMNS)


def create_preprocessor(
    scaling: str = "standard", handle_surrogates: bool = True
) -> Pipeline:
    """
    Builds a unified scikit-learn preprocessing Pipeline.

    Architecture:
        1. Surrogate Code Cleaner: Converts `ca=4` and `thal=0` to NaN.
        2. ColumnTransformer:
           - Numerical Features: Median Imputation + StandardScaler (or RobustScaler).
           - Nominal Categorical Features: Most Frequent Imputation + OneHotEncoder(drop='first').
           - Ordinal Features ('slope', 'ca'): Most Frequent Imputation + StandardScaler.
           - Binary Features ('sex', 'fbs', 'exang'): Most Frequent Imputation + Passthrough.

    Parameters:
        scaling: 'standard' for StandardScaler, 'robust' for RobustScaler.
        handle_surrogates: Whether to include SurrogateMissingHandler as the first step.

    Returns:
        Pipeline: Unfitted Scikit-Learn Pipeline ready for training.
    """
    # 1. Continuous Numerical Pipeline
    scaler = RobustScaler() if scaling.lower() == "robust" else StandardScaler()
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", scaler),
        ]
    )

    # 2. Nominal Categorical Pipeline (One-Hot Encoded)
    nominal_cols = ["cp", "restecg", "thal"]
    cat_nominal_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    drop="first",
                    sparse_output=False,
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    # 3. Ordinal / Discrete Categorical Pipeline (Scaled)
    ordinal_cols = ["slope", "ca"]
    ordinal_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("scaler", StandardScaler()),
        ]
    )

    # 4. Binary Indicator Pipeline (Passthrough)
    binary_cols = BINARY_CATEGORICAL_FEATURES  # ['sex', 'fbs', 'exang']
    binary_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("passthrough", "passthrough"),
        ]
    )

    # Assemble ColumnTransformer
    preprocessor_transformer = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("nom", cat_nominal_pipeline, nominal_cols),
            ("ord", ordinal_pipeline, ordinal_cols),
            ("bin", binary_pipeline, binary_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    if handle_surrogates:
        pipeline = Pipeline(
            steps=[
                ("surrogate_cleaner", SurrogateMissingHandler()),
                ("features", preprocessor_transformer),
            ]
        )
    else:
        pipeline = Pipeline(steps=[("features", preprocessor_transformer)])

    return pipeline


def get_feature_names(fitted_pipeline: Pipeline) -> List[str]:
    """
    Extracts output feature names from the fitted preprocessing pipeline.
    """
    if "features" in fitted_pipeline.named_steps:
        col_transformer = fitted_pipeline.named_steps["features"]
    else:
        col_transformer = fitted_pipeline

    feature_names = col_transformer.get_feature_names_out()
    return list(feature_names)


def save_preprocessor(
    pipeline: Pipeline, filepath: Optional[Union[str, Path]] = None
) -> Path:
    """
    Serializes and saves the fitted preprocessor pipeline to disk using joblib.
    """
    path = Path(filepath) if filepath else PREPROCESSOR_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def load_preprocessor(filepath: Optional[Union[str, Path]] = None) -> Pipeline:
    """
    Loads a previously serialized preprocessor pipeline from disk.
    """
    path = Path(filepath) if filepath else PREPROCESSOR_PATH
    if not path.exists():
        raise FileNotFoundError(f"Preprocessor file not found at: {path.resolve()}")
    pipeline = joblib.load(path)
    return pipeline


def validate_patient_record(patient_data: Union[Dict, pd.DataFrame, pd.Series]) -> pd.DataFrame:
    """
    Validates that an incoming patient record contains all necessary clinical features
    and maps it into a compliant 1-row DataFrame.
    """
    if isinstance(patient_data, dict):
        df_patient = pd.DataFrame([patient_data])
    elif isinstance(patient_data, pd.Series):
        df_patient = patient_data.to_frame().T
    elif isinstance(patient_data, pd.DataFrame):
        df_patient = patient_data.copy()
    else:
        raise TypeError(f"Unsupported patient data type: {type(patient_data)}")

    missing_cols = set(ALL_FEATURE_COLUMNS) - set(df_patient.columns)
    if missing_cols:
        raise ValueError(f"Patient record is missing required clinical fields: {missing_cols}")

    return df_patient[ALL_FEATURE_COLUMNS]


def transform_patient_record(
    fitted_preprocessor: Pipeline, patient_data: Union[Dict, pd.DataFrame, pd.Series]
) -> np.ndarray:
    """
    Transforms a single patient record or batch of unseen records using a fitted pipeline
    WITHOUT refitting.
    """
    df_patient = validate_patient_record(patient_data)
    transformed_record = fitted_preprocessor.transform(df_patient)
    return transformed_record

