"""
Unit tests for Phase 2: Heart Disease Risk Assessment AI Agent.
Tests:
1. Patient intake & input validation (rejection of missing, impossible, and invalid data).
2. Integration with Phase 1 saved champion pipeline (no retraining).
3. Risk interpretation, attribution, and tiering.
4. Structured report generation (JSON and Markdown formats).
5. Persistent longitudinal logging and history retrieval.
6. Differentiation between new assessment, previous assessment, and monitoring history.
7. Pluggable predictor dependency injection.
"""

from pathlib import Path
import pytest

from src.agent import (
    BaseRiskPredictor,
    HeartDiseaseRiskAssessmentAgent,
    PatientMonitoringService,
    PatientValidationError,
    validate_patient_record,
)


@pytest.fixture
def temp_monitoring_service(tmp_path):
    log_file = tmp_path / "test_monitoring.jsonl"
    return PatientMonitoringService(log_path=log_file)


@pytest.fixture
def agent(temp_monitoring_service):
    return HeartDiseaseRiskAssessmentAgent(monitoring_service=temp_monitoring_service)


@pytest.fixture
def sample_patient_record():
    return {
        "age": 58,
        "sex": 1,
        "cp": 2,
        "trestbps": 135,
        "chol": 245,
        "fbs": 0,
        "restecg": 1,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 1.2,
        "slope": 2,
        "ca": 0,
        "thal": 2,
    }


# =====================================================================
# 1. Input Validation Tests
# =====================================================================

def test_valid_patient_intake(sample_patient_record):
    patient_id, clean_feats, warnings = validate_patient_record(sample_patient_record)
    assert patient_id.startswith("PT-")
    assert len(clean_feats) == 13
    assert isinstance(warnings, list)


def test_missing_required_field_rejected(sample_patient_record):
    bad_record = sample_patient_record.copy()
    del bad_record["thalach"]
    with pytest.raises(PatientValidationError, match="missing required clinical field.*thalach"):
        validate_patient_record(bad_record)


def test_invalid_categorical_code_rejected(sample_patient_record):
    bad_record = sample_patient_record.copy()
    bad_record["sex"] = 5
    with pytest.raises(PatientValidationError, match="Invalid categorical value for 'sex'"):
        validate_patient_record(bad_record)


def test_impossible_blood_pressure_rejected(sample_patient_record):
    bad_record = sample_patient_record.copy()
    bad_record["trestbps"] = 350
    with pytest.raises(PatientValidationError, match="Physiologically implausible value for 'trestbps'"):
        validate_patient_record(bad_record)


def test_empty_record_rejected():
    with pytest.raises(PatientValidationError, match="record dictionary is empty"):
        validate_patient_record({})


# =====================================================================
# 2. Agent Assessment & Risk Interpretation Tests
# =====================================================================

def test_assess_patient_full_workflow(agent, sample_patient_record):
    result = agent.assess_patient(sample_patient_record, patient_id="PT-TEST-001", save_report=False)

    assert result["patient_id"] == "PT-TEST-001"
    assert result["assessment_type"] == "NEW_ASSESSMENT"
    assert result["predicted_class"] in [0, 1]
    assert 0.0 <= result["risk_probability"] <= 1.0
    assert result["risk_category"] in ["LOW", "MODERATE", "HIGH", "CRITICAL"]
    assert "json_report" in result
    assert "markdown_report" in result


def test_report_contains_all_required_sections(agent, sample_patient_record):
    result = agent.assess_patient(sample_patient_record, patient_id="PT-TEST-REPORT", save_report=False)
    report = result["json_report"]

    assert report["report_metadata"]["patient_id"] == "PT-TEST-REPORT"
    assert "diagnostic_assessment" in report
    assert "clinical_attribution" in report
    assert "recommendations" in report
    assert "disclaimer" in report
    assert "DISCLAIMER" in report["disclaimer"]
    assert "does NOT constitute a medical diagnosis" in report["disclaimer"]


# =====================================================================
# 3. Longitudinal Monitoring & History Differentiation Tests
# =====================================================================

def test_distinguishes_new_vs_followup_assessment(agent, sample_patient_record):
    pid = "PT-LONGITUDINAL-01"

    # 1st Assessment: New
    res1 = agent.assess_patient(sample_patient_record, patient_id=pid, save_report=False)
    assert res1["assessment_type"] == "NEW_ASSESSMENT"
    assert res1["trend_status"] == "FIRST_ASSESSMENT"

    # 2nd Assessment: Follow-up
    res2 = agent.assess_patient(sample_patient_record, patient_id=pid, save_report=False)
    assert res2["assessment_type"] == "FOLLOWUP_ASSESSMENT"
    assert res2["trend_status"] == "STABLE"


def test_previous_assessment_retrieval(agent, sample_patient_record):
    pid = "PT-RETRIEVE-01"
    res = agent.assess_patient(sample_patient_record, patient_id=pid, save_report=False)

    prev = agent.get_previous_assessment(pid)
    assert prev is not None
    assert prev["assessment_id"] == res["assessment_id"]
    assert prev["risk_probability"] == res["risk_probability"]


def test_monitoring_history_trajectory(agent, sample_patient_record):
    pid = "PT-TRAJECTORY-01"

    # Baseline
    agent.assess_patient(sample_patient_record, patient_id=pid, save_report=False)

    # 2nd visit with higher blood pressure
    visit2 = sample_patient_record.copy()
    visit2["trestbps"] = 175
    agent.assess_patient(visit2, patient_id=pid, save_report=False)

    history = agent.get_monitoring_history(pid)
    assert history["status"] == "MONITORING_ACTIVE"
    assert history["assessment_count"] == 2
    assert len(history["trajectory"]) == 2
    assert history["trajectory"][0]["trestbps"] == 135
    assert history["trajectory"][1]["trestbps"] == 175


# =====================================================================
# 4. Pluggable Predictor Interface Tests
# =====================================================================

class MockPredictor(BaseRiskPredictor):
    """Mock predictor demonstrating decoupled model replacement."""
    def predict_risk(self, patient_features, threshold=0.50):
        return {
            "predicted_class": 1,
            "risk_probability": 0.88,
            "decision_threshold": threshold,
            "model_name": "Mock Test Predictor",
            "model_artifact": "mock.joblib",
        }


def test_pluggable_predictor_injection(temp_monitoring_service, sample_patient_record):
    mock = MockPredictor()
    custom_agent = HeartDiseaseRiskAssessmentAgent(
        predictor=mock, monitoring_service=temp_monitoring_service
    )
    result = custom_agent.assess_patient(sample_patient_record, patient_id="PT-MOCK-01", save_report=False)

    assert result["risk_probability"] == 0.88
    assert result["risk_category"] == "CRITICAL"
    assert result["json_report"]["diagnostic_assessment"]["model_used"] == "Mock Test Predictor"
