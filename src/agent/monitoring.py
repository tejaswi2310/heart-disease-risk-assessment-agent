"""
Longitudinal patient monitoring and persistent audit logging service.
Tracks patient cardiovascular trajectories, assesses historical risk shifts,
and generates clinical trend alerts over time.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

from src.agent.config import MONITORING_LOG_PATH, TREND_STATUS


class PatientMonitoringService:
    """
    Manages persistent longitudinal logging and historical trajectory retrieval.
    Stores immutable JSON Lines records at data/monitoring/patient_monitoring_log.jsonl.
    """

    def __init__(self, log_path: Optional[Union[str, Path]] = None):
        self.log_path = Path(log_path) if log_path else MONITORING_LOG_PATH
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_assessment(
        self,
        patient_id: str,
        features: Dict[str, Any],
        predicted_class: int,
        risk_probability: float,
        risk_category: str,
        validation_warnings: List[str],
        timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Appends an assessment to the persistent monitoring log, computing longitudinal
        trends relative to prior evaluations for the same patient.
        """
        now_iso = timestamp or datetime.now(timezone.utc).isoformat()
        assessment_id = f"ASM-{uuid.uuid4().hex[:8].upper()}"

        # Retrieve prior history for this patient
        prior_history = self.get_patient_history(patient_id)

        # Compute longitudinal comparison
        trend_analysis = self._compute_trend(prior_history, features, risk_probability)

        log_record = {
            "assessment_id": assessment_id,
            "patient_id": patient_id,
            "timestamp": now_iso,
            "prediction": predicted_class,
            "risk_probability": round(float(risk_probability), 4),
            "risk_category": risk_category,
            "trend_status": trend_analysis["status"],
            "trend_summary": trend_analysis["summary"],
            "risk_delta": trend_analysis["risk_delta"],
            "vital_deltas": trend_analysis["vital_deltas"],
            "validation_warnings": validation_warnings,
            "features_snapshot": features,
            "prior_assessment_count": len(prior_history),
        }

        # Append record to JSONL log
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_record) + "\n")

        return log_record

    def get_patient_history(self, patient_id: str) -> List[Dict[str, Any]]:
        """
        Retrieves all past assessments for a specific patient, sorted chronologically.
        """
        if not self.log_path.exists():
            return []

        patient_records: List[Dict[str, Any]] = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if record.get("patient_id") == patient_id:
                        patient_records.append(record)
                except json.JSONDecodeError:
                    continue

        # Sort chronologically by timestamp
        patient_records.sort(key=lambda x: x.get("timestamp", ""))
        return patient_records

    def get_all_records(self) -> List[Dict[str, Any]]:
        """Retrieves all log records across all patients."""
        if not self.log_path.exists():
            return []

        all_records: List[Dict[str, Any]] = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        all_records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return all_records

    def _compute_trend(
        self,
        prior_history: List[Dict[str, Any]],
        current_features: Dict[str, Any],
        current_risk: float,
    ) -> Dict[str, Any]:
        """
        Calculates shift in risk probability and clinical vitals compared to the
        immediate previous assessment.
        """
        if not prior_history:
            return {
                "status": "FIRST_ASSESSMENT",
                "summary": TREND_STATUS["FIRST_ASSESSMENT"],
                "risk_delta": 0.0,
                "vital_deltas": {},
            }

        last_record = prior_history[-1]
        prior_risk = last_record.get("risk_probability", current_risk)
        risk_delta = round(current_risk - prior_risk, 4)

        prior_features = last_record.get("features_snapshot", {})
        vital_deltas = {
            "trestbps_delta": current_features.get("trestbps", 0) - prior_features.get("trestbps", 0),
            "chol_delta": current_features.get("chol", 0) - prior_features.get("chol", 0),
            "thalach_delta": current_features.get("thalach", 0) - prior_features.get("thalach", 0),
            "oldpeak_delta": round(current_features.get("oldpeak", 0.0) - prior_features.get("oldpeak", 0.0), 2),
        }

        # Status classification
        if current_risk >= 0.85 and prior_risk < 0.85:
            status = "CRITICAL_ALERT"
            summary = f"Critical Alert: Estimated risk escalated to {current_risk*100:.1f}% (+{risk_delta*100:.1f}%)."
        elif risk_delta > 0.05:
            status = "DETERIORATED"
            summary = f"Risk Deteriorated: Cardiovascular risk increased by {risk_delta*100:.1f}%."
        elif risk_delta < -0.05:
            status = "IMPROVED"
            summary = f"Risk Improved: Cardiovascular risk decreased by {abs(risk_delta)*100:.1f}%."
        else:
            status = "STABLE"
            summary = f"Risk Stable: Negligible shift ({risk_delta*100:+.1f}%)."

        return {
            "status": status,
            "summary": summary,
            "risk_delta": risk_delta,
            "vital_deltas": vital_deltas,
        }

