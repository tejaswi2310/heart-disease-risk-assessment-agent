"""
Risk interpretation, feature attribution, and evidence-based recommendation module.
Translates raw model output into explainable, clinically grounded risk assessments.
"""

from typing import Any, Dict, List, Tuple
from src.agent.config import CLINICAL_DISCLAIMER, PHYSIOLOGICAL_BOUNDS, RISK_TIERS


def map_risk_category(risk_prob: float) -> Tuple[str, str, str]:
    """
    Maps continuous risk probability to standardized clinical tier,
    descriptive label, and recommended clinical management triage action.
    """
    if risk_prob <= RISK_TIERS["LOW"]["threshold"]:
        return (
            "LOW",
            "Low Risk of Obstructive Coronary Artery Disease",
            RISK_TIERS["LOW"]["action"],
        )
    elif risk_prob <= RISK_TIERS["MODERATE"]["threshold"]:
        return (
            "MODERATE",
            "Moderate Risk of Obstructive Coronary Artery Disease",
            RISK_TIERS["MODERATE"]["action"],
        )
    elif risk_prob <= RISK_TIERS["HIGH"]["threshold"]:
        return (
            "HIGH",
            "High Risk of Obstructive Coronary Artery Disease",
            RISK_TIERS["HIGH"]["action"],
        )
    else:
        return (
            "CRITICAL",
            "Critical Cardiovascular Risk / Acute Probability",
            RISK_TIERS["CRITICAL"]["action"],
        )


def analyze_contributing_factors(
    features: Dict[str, Any]
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Identifies patient-specific clinical factors that drive estimated risk upward
    or act as protective/favorable cardiovascular indicators.
    """
    elevating_factors: List[Dict[str, str]] = []
    protective_factors: List[Dict[str, str]] = []

    # 1. ST Depression (oldpeak)
    oldpeak = features.get("oldpeak", 0.0)
    if oldpeak >= 2.0:
        elevating_factors.append({
            "factor": "Pronounced Exercise ST Depression",
            "value": f"{oldpeak} mm",
            "clinical_impact": "Direct marker of subendocardial myocardial ischemia under exercise stress.",
        })
    elif oldpeak >= 1.0:
        elevating_factors.append({
            "factor": "Mild-to-Moderate ST Depression",
            "value": f"{oldpeak} mm",
            "clinical_impact": "Borderline exercise-induced electrocardiographic ischemia.",
        })
    elif oldpeak == 0.0:
        protective_factors.append({
            "factor": "Normal ST Segment Baseline",
            "value": "0.0 mm",
            "clinical_impact": "No exercise-induced ST depression observed during stress test.",
        })

    # 2. Fluoroscopy Major Vessels (ca)
    ca = features.get("ca", 0)
    if ca in [1, 2, 3]:
        elevating_factors.append({
            "factor": "Fluoroscopic Coronary Calcification / Stenosis",
            "value": f"{ca} major vessel(s) opacified",
            "clinical_impact": f"Direct anatomical visualization of obstructive coronary lesions in {ca} vessels.",
        })
    elif ca == 0:
        protective_factors.append({
            "factor": "Clear Fluoroscopy Vessels",
            "value": "0 vessels obstructed",
            "clinical_impact": "Absence of major fluoroscopic fluorophore staining in main vessels.",
        })

    # 3. Chest Pain Type (cp)
    cp = features.get("cp", 0)
    cp_labels = PHYSIOLOGICAL_BOUNDS["cp"]["labels"]
    if cp in [2, 3]:
        elevating_factors.append({
            "factor": f"Atypical Presentation ({cp_labels.get(cp)})",
            "value": f"Type {cp}",
            "clinical_impact": "In secondary referral populations, atypical/asymptomatic presentations correlate with occult CAD.",
        })

    # 4. Exercise Angina (exang)
    exang = features.get("exang", 0)
    if exang == 1:
        elevating_factors.append({
            "factor": "Exercise-Induced Angina",
            "value": "Positive (Yes)",
            "clinical_impact": "Reproducible chest pain triggered by exertional cardiac workload.",
        })
    else:
        protective_factors.append({
            "factor": "No Exertional Angina",
            "value": "Negative (No)",
            "clinical_impact": "Absence of angina during maximum exercise testing.",
        })

    # 5. Thalassemia Defect (thal)
    thal = features.get("thal", 1)
    if thal == 2:
        elevating_factors.append({
            "factor": "Thalassemia Fixed Perfusion Defect",
            "value": "Fixed Defect (Code 2)",
            "clinical_impact": "Often associated with prior irreversible myocardial scar or localized infarction.",
        })
    elif thal == 1:
        protective_factors.append({
            "factor": "Normal Myocardial Perfusion (Thal)",
            "value": "Normal (Code 1)",
            "clinical_impact": "Uniform nuclear scintigraphy myocardial uptake.",
        })

    # 6. Blood Pressure (trestbps)
    trestbps = features.get("trestbps", 120)
    if trestbps >= 160:
        elevating_factors.append({
            "factor": "Stage 2 Severe Hypertension",
            "value": f"{trestbps} mm Hg",
            "clinical_impact": "Marked chronic hemodynamic afterload and arterial shear stress.",
        })
    elif trestbps >= 140:
        elevating_factors.append({
            "factor": "Stage 1 Hypertension",
            "value": f"{trestbps} mm Hg",
            "clinical_impact": "Elevated resting systolic pressure increases cardiovascular workload.",
        })
    elif trestbps < 120:
        protective_factors.append({
            "factor": "Optimal Resting Blood Pressure",
            "value": f"{trestbps} mm Hg",
            "clinical_impact": "Systolic pressure within normotensive target range.",
        })

    # 7. Serum Cholesterol (chol)
    chol = features.get("chol", 200)
    if chol >= 280:
        elevating_factors.append({
            "factor": "Hypercholesterolemia",
            "value": f"{chol} mg/dl",
            "clinical_impact": "Elevated circulating lipids accelerate atherosclerotic plaque progression.",
        })
    elif chol < 200:
        protective_factors.append({
            "factor": "Desirable Cholesterol Range",
            "value": f"{chol} mg/dl",
            "clinical_impact": "Serum cholesterol meets AHA-recommended cardiovascular targets.",
        })

    # 8. Maximum Heart Rate (thalach)
    thalach = features.get("thalach", 150)
    if thalach >= 160:
        protective_factors.append({
            "factor": "High Exercise Capacity (Max HR)",
            "value": f"{thalach} bpm",
            "clinical_impact": "Good chronotropic competence and cardiovascular reserve.",
        })
    elif thalach < 120:
        elevating_factors.append({
            "factor": "Chronotropic Incompetence / Low Max HR",
            "value": f"{thalach} bpm",
            "clinical_impact": "Inability to achieve target exercise heart rate indicates impaired functional capacity.",
        })

    return elevating_factors, protective_factors


def formulate_clinical_recommendations(
    risk_tier: str,
    features: Dict[str, Any],
    elevating_factors: List[Dict[str, str]],
) -> List[str]:
    """
    Formulates evidence-based lifestyle and clinical consultation recommendations
    aligned with established AHA/ACC preventive cardiology guidelines.
    """
    recs: List[str] = []

    # Tier-based core management
    if risk_tier == "CRITICAL":
        recs.append("URGENT MEDICAL EVALUATION: Schedule an immediate consultation with an attending cardiologist.")
        recs.append("DIAGNOSTIC WORKUP: Recommend emergency or urgent coronary angiography / cardiac catheterization.")
        recs.append("CONTINUOUS MONITORING: Avoid strenuous physical exertion until full cardiovascular clearance is granted.")
    elif risk_tier == "HIGH":
        recs.append("SPECIALIST REFERRAL: Comprehensive cardiological examination within 1 to 2 weeks.")
        recs.append("NON-INVASIVE IMAGING: Consider stress echocardiography or coronary CT angiography (CCTA).")
        recs.append("MEDICATION REVIEW: Evaluate guideline-directed medical therapy (statins, ACE-inhibitors, anti-platelets).")
    elif risk_tier == "MODERATE":
        recs.append("CLINICAL FOLLOW-UP: Routine follow-up visit with primary care physician within 30 days.")
        recs.append("LIPID & GLYCEMIC AUDIT: Repeat fasting lipid profile (LDL-C, HDL-C, Triglycerides) and HbA1c.")
        recs.append("LIFESTYLE INTERVENTION: Initiate structured aerobic exercise (150 min/week) and Mediterranean diet.")
    else:  # LOW
        recs.append("ROUTINE SCREENING: Continue standard annual wellness checks and cardiovascular monitoring.")
        recs.append("HEALTH MAINTENANCE: Maintain balanced low-sodium diet, regular physical activity, and tobacco avoidance.")

    # Feature-specific targeted guidance
    trestbps = features.get("trestbps", 120)
    if trestbps >= 140:
        recs.append(f"HYPERTENSION MANAGEMENT: Implement home blood pressure logging; target systolic BP < 130 mm Hg.")

    chol = features.get("chol", 200)
    if chol >= 240:
        recs.append(f"LIPID OPTIMIZATION: Dietary reduction of saturated/trans fats; discuss lipid-lowering therapy.")

    fbs = features.get("fbs", 0)
    if fbs == 1:
        recs.append("GLYCEMIC CONTROL: Fasting glucose > 120 mg/dl; evaluate for type 2 diabetes or insulin resistance.")

    return recs


def generate_risk_interpretation(
    prediction_result: Dict[str, Any], features: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Combines raw prediction, feature attribution, and guidelines into a complete
    clinical decision-support risk interpretation.
    """
    risk_prob = prediction_result["risk_probability"]
    risk_tier, tier_label, triage_action = map_risk_category(risk_prob)
    elevating_factors, protective_factors = analyze_contributing_factors(features)
    recommendations = formulate_clinical_recommendations(risk_tier, features, elevating_factors)

    return {
        "risk_tier": risk_tier,
        "risk_tier_label": tier_label,
        "triage_action": triage_action,
        "elevating_risk_factors": elevating_factors,
        "protective_factors": protective_factors,
        "clinical_recommendations": recommendations,
        "disclaimer": CLINICAL_DISCLAIMER,
    }
