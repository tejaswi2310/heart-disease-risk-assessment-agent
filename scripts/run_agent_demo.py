"""
Demonstration and operational execution script for the Heart Disease Risk Assessment AI Agent.
Simulates baseline clinical assessment, longitudinal patient follow-up, and error rejection.
"""

import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import HeartDiseaseRiskAssessmentAgent, PatientValidationError


def run_agent_demonstration():
    print("=" * 80)
    print("HEART DISEASE RISK ASSESSMENT AI AGENT - CLINICAL WORKFLOW DEMONSTRATION")
    print("=" * 80)

    # Initialize agent (reusing Phase 1 champion pipeline)
    agent = HeartDiseaseRiskAssessmentAgent()
    test_patient_id = "PT-CARDIO-DEMO-01"

    # -----------------------------------------------------------------
    # SCENARIO 1: New Patient Intake (Baseline Clinical Assessment)
    # -----------------------------------------------------------------
    print("\n" + "-" * 80)
    print("SCENARIO 1: Intake & Assessment of New Patient (Baseline Evaluation)")
    print("-" * 80)

    baseline_record = {
        "age": 63,
        "sex": 1,          # Male
        "cp": 3,           # Asymptomatic chest pain (occult ischemia presentation)
        "trestbps": 165,   # Stage 2 Hypertension
        "chol": 295,       # Hypercholesterolemia
        "fbs": 1,          # Fasting blood sugar > 120
        "restecg": 1,      # ST-T wave abnormality
        "thalach": 128,    # Impaired exercise capacity
        "exang": 1,        # Exercise-induced angina
        "oldpeak": 2.8,    # Severe ST depression
        "slope": 1,        # Flat ST slope
        "ca": 2,           # 2 major coronary vessels with fluoroscopic stenosis
        "thal": 2,         # Fixed perfusion defect
    }

    print(f"Submitting baseline record for Patient ID: {test_patient_id}")
    print(f"Vitals: Age=63, BP=165 mmHg, Chol=295 mg/dl, ST Depression=2.8 mm, Vessels=2")

    result_baseline = agent.assess_patient(
        baseline_record, patient_id=test_patient_id, save_report=True
    )

    print("\n[Agent Workflow Completed]:")
    print(f"  -> Assessment Type:       {result_baseline['assessment_type']}")
    print(f"  -> Assessment ID:         {result_baseline['assessment_id']}")
    print(f"  -> Model Predicted Class: {result_baseline['predicted_class']} (Disease Present)")
    print(f"  -> Estimated Risk:        {result_baseline['risk_probability'] * 100:.1f}%")
    print(f"  -> Risk Classification:   {result_baseline['risk_category']}")
    print(f"  -> Trend Status:          {result_baseline['trend_status']}")
    print(f"  -> Report Saved To:       {result_baseline['report_path']}")

    # Print first few lines of generated Markdown consultation report
    print("\n--- [Excerpt from Generated Clinical Markdown Report] ---")
    print("\n".join(result_baseline["markdown_report"].splitlines()[:25]))
    print("...")

    # -----------------------------------------------------------------
    # SCENARIO 2: Follow-Up Assessment after 3 Months of Therapy
    # -----------------------------------------------------------------
    print("\n" + "-" * 80)
    print("SCENARIO 2: Longitudinal Follow-up (3-Month Re-evaluation of Same Patient)")
    print("-" * 80)

    followup_record = {
        "age": 63,
        "sex": 1,
        "cp": 1,           # Chest pain improved to atypical angina
        "trestbps": 124,   # Normotensive under anti-hypertensive therapy (down from 165)
        "chol": 195,       # Cholesterol normalized under statin therapy (down from 295)
        "fbs": 0,          # Glycemic control restored (down from 1)
        "restecg": 0,      # Resting ECG normalized
        "thalach": 158,    # Improved exercise tolerance (up from 128)
        "exang": 0,        # No exertional angina (down from 1)
        "oldpeak": 0.6,    # Minimal ST depression (down from 2.8)
        "slope": 2,        # Upsloping ST segment
        "ca": 2,           # Stable anatomical vessels
        "thal": 1,         # Reversible defect resolved
    }

    print(f"Submitting 3-month follow-up record for Patient ID: {test_patient_id}")
    print(f"Vitals Post-Treatment: BP=124 mmHg (-41), Chol=195 mg/dl (-100), ST Depression=0.6 mm (-2.2 mm)")

    result_followup = agent.assess_patient(
        followup_record, patient_id=test_patient_id, save_report=True
    )

    print("\n[Agent Follow-Up Workflow Completed]:")
    print(f"  -> Assessment Type:       {result_followup['assessment_type']}")
    print(f"  -> Assessment ID:         {result_followup['assessment_id']}")
    print(f"  -> Model Predicted Class: {result_followup['predicted_class']}")
    print(f"  -> New Estimated Risk:    {result_followup['risk_probability'] * 100:.1f}% (Baseline was {result_baseline['risk_probability']*100:.1f}%)")
    print(f"  -> Risk Classification:   {result_followup['risk_category']}")
    print(f"  -> Longitudinal Trend:    {result_followup['trend_status']}")
    print(f"  -> Trend Summary:         {result_followup['trend_summary']}")

    # -----------------------------------------------------------------
    # SCENARIO 3: Retrieving Patient History vs. Monitoring Summary
    # -----------------------------------------------------------------
    print("\n" + "-" * 80)
    print("SCENARIO 3: Distinct Querying of Previous Assessment vs. Monitoring Progression")
    print("-" * 80)

    # 3A: Retrieve specific previous assessment
    prev_assessment = agent.get_previous_assessment(test_patient_id)
    print(f"[Query A: Previous Assessment on File]")
    print(f"  -> Retrieved Assessment ID: {prev_assessment['assessment_id']}")
    print(f"  -> Recorded Risk:          {prev_assessment['risk_probability']*100:.1f}% ({prev_assessment['risk_category']})")
    print(f"  -> Timestamp:              {prev_assessment['timestamp']}")

    # 3B: Retrieve full longitudinal monitoring trajectory
    history_summary = agent.get_monitoring_history(test_patient_id)
    print(f"\n[Query B: Full Longitudinal Progression]")
    print(f"  -> Patient:            {history_summary['patient_id']}")
    print(f"  -> Total Encounters:   {history_summary['assessment_count']}")
    print(f"  -> Baseline Risk:      {history_summary['baseline_risk']*100:.1f}%")
    print(f"  -> Latest Risk:        {history_summary['latest_risk']*100:.1f}%")
    print(f"  -> Net Risk Shift:     {history_summary['overall_risk_shift']*100:+.1f}%")
    print(f"  -> Trajectory Progression:")
    for i, step in enumerate(history_summary["trajectory"]):
        print(f"     Encounter {i+1}: Risk={step['risk_probability']*100:.1f}% [{step['risk_category']}], BP={step['trestbps']} mmHg, ST Dep={step['oldpeak']} mm ({step['trend_status']})")

    # -----------------------------------------------------------------
    # SCENARIO 4: Input Validation & Error Handling (Rejection of Invalid Data)
    # -----------------------------------------------------------------
    print("\n" + "-" * 80)
    print("SCENARIO 4: Input Validation Audit (Rejection of Invalid & Inconsistent Records)")
    print("-" * 80)

    # Test Case 4A: Missing required field
    print("[Test 4A] Submitting record missing mandatory field 'trestbps'...")
    invalid_record_missing = baseline_record.copy()
    del invalid_record_missing["trestbps"]
    try:
        agent.assess_patient(invalid_record_missing)
        print("  ❌ ERROR: Agent failed to reject missing field!")
    except PatientValidationError as e:
        print(f"  ✅ SUCCESS: Agent caught missing field -> {e}")

    # Test Case 4B: Impossible physiological value
    print("\n[Test 4B] Submitting record with impossible blood pressure (trestbps = -50 mm Hg)...")
    invalid_record_bp = baseline_record.copy()
    invalid_record_bp["trestbps"] = -50
    try:
        agent.assess_patient(invalid_record_bp)
        print("  ❌ ERROR: Agent failed to reject impossible blood pressure!")
    except PatientValidationError as e:
        print(f"  ✅ SUCCESS: Agent caught impossible blood pressure -> {e}")

    # Test Case 4C: Invalid categorical code
    print("\n[Test 4C] Submitting record with invalid chest pain code (cp = 99)...")
    invalid_record_cp = baseline_record.copy()
    invalid_record_cp["cp"] = 99
    try:
        agent.assess_patient(invalid_record_cp)
        print("  ❌ ERROR: Agent failed to reject invalid categorical code!")
    except PatientValidationError as e:
        print(f"  ✅ SUCCESS: Agent caught invalid categorical code -> {e}")

    print("\n" + "=" * 80)
    print("ALL AI AGENT DEMONSTRATION WORKFLOWS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_agent_demonstration()
