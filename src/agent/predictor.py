"""
Pluggable Machine Learning prediction module for the AI Risk Assessment Agent.
Decouples model execution from the agent logic, allowing model interchangeability.
"""

from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
import joblib
import numpy as np
import pandas as pd

from src.config import ALL_FEATURE_COLUMNS, CHAMPION_METADATA_PATH, CHAMPION_MODEL_PATH


class BaseRiskPredictor(ABC):
    """Abstract interface defining the contract for clinical risk predictors."""

    @abstractmethod
    def predict_risk(
        self, patient_features: Dict[str, Union[int, float]], threshold: float = 0.50
    ) -> Dict[str, Any]:
        """
        Executes prediction on raw clinical features.
        Returns dictionary containing predicted_class, risk_probability, model_name, and threshold.
        """
        pass


class Phase1MLPredictor(BaseRiskPredictor):
    """
    Predictor implementation that interfaces with the trained and serialized Phase 1 champion pipeline.
    Does NOT retrain or alter the model.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        metadata_path: Optional[Union[str, Path]] = None,
    ):
        self.model_path = Path(model_path) if model_path else CHAMPION_MODEL_PATH
        self.metadata_path = Path(metadata_path) if metadata_path else CHAMPION_METADATA_PATH

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained Phase 1 pipeline not found at {self.model_path.resolve()}. "
                "Ensure Phase 1 model training has completed."
            )

        # Load end-to-end serialized pipeline
        self.pipeline = joblib.load(self.model_path)

        # Load metadata if present
        self.metadata = {}
        if self.metadata_path.exists():
            with open(self.metadata_path, "r") as f:
                self.metadata = json.load(f)

        self.model_name = self.metadata.get("champion_model_name", "Trained ML Model")

    def predict_risk(
        self, patient_features: Dict[str, Union[int, float]], threshold: float = 0.50
    ) -> Dict[str, Any]:
        """
        Executes end-to-end inference:
        patient_features -> Preprocessor (imputation + scaling + OHE) -> Classifier -> Risk Probability
        """
        # Convert dictionary to DataFrame with canonical column ordering
        df_patient = pd.DataFrame([patient_features])[ALL_FEATURE_COLUMNS]

        # Extract posterior probability for positive class (disease present)
        if hasattr(self.pipeline, "predict_proba"):
            probs = self.pipeline.predict_proba(df_patient)
            risk_prob = float(probs[0, 1])
        elif hasattr(self.pipeline, "decision_function"):
            from scipy.special import expit
            decision = self.pipeline.decision_function(df_patient)
            risk_prob = float(expit(decision)[0])
        else:
            raise AttributeError("Loaded pipeline does not support probability prediction.")

        predicted_class = int(risk_prob >= threshold)

        return {
            "predicted_class": predicted_class,
            "risk_probability": round(risk_prob, 4),
            "decision_threshold": threshold,
            "model_name": self.model_name,
            "model_artifact": str(self.model_path.name),
        }

