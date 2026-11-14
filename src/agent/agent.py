"""
Heart Disease Risk Assessment AI Agent.
Coordinates patient intake, clinical validation, ML model inference, risk interpretation,
structured report generation, and longitudinal audit logging.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.agent.config import PATIENT_REPORTS_DIR
from src.agent.interpretation import generate_risk_interpretation
from src.agent.monitoring import PatientMonitoringService
from src.agent.predictor import BaseRiskPredictor, Phase1MLPredictor
from src.agent.report_generator import HealthRiskReportGenerator
from src.agent.validation import PatientValidationError, validate_patient_record


class HeartDiseaseRiskAssessmentAgent:
    """
    Intelligent Healthcare Monitoring and Risk Assessment Agent.
    
    Modular Workflow Architecture:
        PATIENT RECORD
        ↓
        INPUT VALIDATION (validate_patient_record)
        ↓
        PATIENT DATA ANALYSIS (consistency checks, surrogate imputation flags)
        ↓
        PREPROCESSING PIPELINE (Phase 1 ColumnTransformer inside pipeline)
        ↓
        TRAINED ML MODEL (Phase 1 Champion Classifier)
        ↓
        RISK PREDICTION (posterior probabilities & binary classification)
        ↓
        RISK INTERPRETATION (triage tiering, clinical attribution, recommendations)
        ↓
        HEALTH-RISK REPORT (structured JSON & formatted Markdown reports)
        ↓
        MONITORING LOG (persistent JSONL longitudinal tracking)
    """

    def __init__(
        self,
        predictor: Optional[BaseRiskPredictor] = None,
        monitoring_service: Optional[PatientMonitoringService] = None,
    ):
        """
        Initializes the agent with pluggable predictor and monitoring components.
        Allows model replacement without modifying the agent workflow logic.
        """
        self.predictor = predictor or Phase1MLPredictor()
        self.monitoring_service = monitoring_service or PatientMonitoringService()
        self.report_generator = HealthRiskReportGenerator()

    def assess_patient(
        self,
        patient_record: Union[Dict[str, Any], Any],
        patient_id: Optional[str] = None,
        threshold: float = 0.50,
        save_report: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes the primary clinical risk assessment workflow on an intake patient record.

        Parameters:
            patient_record: Patient features corresponding to the 13 clinical predictors.
            patient_id: Optional external patient ID. If omitted, a privacy-safe ID is generated.
            threshold: Decision boundary for binary classification. Default is 0.50.
            save_report: Whether to persist the generated markdown report to disk.

        Returns:
            Dict[str, Any]: Comprehensive assessment result with reports, predictions, and trend metrics.

        Raises:
            PatientValidationError: If the patient record violates schema or clinical bounds.
        """
        # Step 1: Input Validation & Physiological Consistency Checking
        clean_patient_id, clean_features, validation_warnings = validate_patient_record(
            patient_record, patient_id=patient_id
        )

        # Step 2: Query Prior History to determine Assessment Mode
        prior_history = self.monitoring_service.get_patient_history(clean_patient_id)
        assessment_mode = "NEW_ASSESSMENT" if len(prior_history) == 0 else "FOLLOWUP_ASSESSMENT"

        # Step 3: Risk Prediction via Phase 1 Pipeline
        prediction_result = self.predictor.predict_risk(clean_features, threshold=threshold)

        # Step 4: Risk Interpretation & Attribution
        interpretation = generate_risk_interpretation(prediction_result, clean_features)

        # Step 5: Persistent Longitudinal Audit Logging
        now_iso = datetime.now(timezone.utc).isoformat()
        log_record = self.monitoring_service.log_assessment(
            patient_id=clean_patient_id,
            features=clean_features,
            predicted_class=prediction_result["predicted_class"],
            risk_probability=prediction_result["risk_probability"],
            risk_category=interpretation["risk_tier"],
            validation_warnings=validation_warnings,
            timestamp=now_iso,
        )

        # Step 6: Generate Structured Reports (JSON and Markdown)
        json_report = self.report_generator.generate_json_report(
            patient_id=clean_patient_id,
            assessment_id=log_record["assessment_id"],
            timestamp=now_iso,
            features=clean_features,
            prediction_result=prediction_result,
            interpretation=interpretation,
            monitoring_log=log_record,
            validation_warnings=validation_warnings,
        )
        markdown_report = self.report_generator.generate_markdown_report(json_report)

        # Step 7: Optional Disk Persistence of Clinical Report
        saved_report_path = None
        if save_report:
            saved_report_path = self._save_markdown_report(clean_patient_id, log_record["assessment_id"], markdown_report)

        return {
            "assessment_id": log_record["assessment_id"],
            "patient_id": clean_patient_id,
            "assessment_type": assessment_mode,
            "timestamp": now_iso,
            "predicted_class": prediction_result["predicted_class"],
            "risk_probability": prediction_result["risk_probability"],
            "risk_category": interpretation["risk_tier"],
            "trend_status": log_record["trend_status"],
            "trend_summary": log_record["trend_summary"],
            "validation_warnings": validation_warnings,
            "report_path": str(saved_report_path) if saved_report_path else None,
            "json_report": json_report,
            "markdown_report": markdown_report,
        }

    def get_previous_assessment(
        self, patient_id: str, assessment_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieves a specific previous assessment or the most recent evaluation on file.
        Distinguishes single historical inspection from full longitudinal history.
        """
        history = self.monitoring_service.get_patient_history(patient_id)
        if not history:
            return None

        if assessment_id:
            for record in history:
                if record.get("assessment_id") == assessment_id:
                    return record
            return None

        # Return latest prior assessment
        return history[-1]

    def get_monitoring_history(self, patient_id: str) -> Dict[str, Any]:
        """
        Retrieves the complete longitudinal monitoring progression and trajectory
        for a specific patient over time.
        """
        history = self.monitoring_service.get_patient_history(patient_id)
        if not history:
            return {
                "patient_id": patient_id,
                "status": "NO_RECORDS_FOUND",
                "assessment_count": 0,
                "records": [],
            }

        first_eval = history[0]
        latest_eval = history[-1]
        baseline_risk = first_eval.get("risk_probability", 0.0)
        current_risk = latest_eval.get("risk_probability", 0.0)
        overall_delta = round(current_risk - baseline_risk, 4)

        trajectory = [
            {
                "assessment_id": r.get("assessment_id"),
                "timestamp": r.get("timestamp"),
                "risk_probability": r.get("risk_probability"),
                "risk_category": r.get("risk_category"),
                "trend_status": r.get("trend_status"),
                "trestbps": r.get("features_snapshot", {}).get("trestbps"),
                "oldpeak": r.get("features_snapshot", {}).get("oldpeak"),
            }
            for r in history
        ]

        return {
            "patient_id": patient_id,
            "status": "MONITORING_ACTIVE",
            "assessment_count": len(history),
            "first_assessment_date": first_eval.get("timestamp"),
            "latest_assessment_date": latest_eval.get("timestamp"),
            "baseline_risk": baseline_risk,
            "latest_risk": current_risk,
            "overall_risk_shift": overall_delta,
            "current_trend_status": latest_eval.get("trend_status"),
            "trajectory": trajectory,
        }

    def _save_markdown_report(
        self, patient_id: str, assessment_id: str, markdown_content: str
    ) -> Path:
        """Saves generated markdown consultation report to reports/patient_reports/."""
        PATIENT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"{patient_id}_{assessment_id}.md"
        out_path = PATIENT_REPORTS_DIR / filename
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        return out_path

