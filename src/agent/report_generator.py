"""
Clinical Health-Risk Report Generator module.
Formats comprehensive, structured diagnostic assessments into structured JSON and
human-readable Markdown clinical consultation reports.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from src.agent.config import AGENT_NAME, AGENT_VERSION, CLINICAL_DISCLAIMER, PHYSIOLOGICAL_BOUNDS


class HealthRiskReportGenerator:
    """Generates structured JSON and publication-quality Markdown clinical reports."""

    @staticmethod
    def generate_json_report(
        patient_id: str,
        assessment_id: str,
        timestamp: str,
        features: Dict[str, Any],
        prediction_result: Dict[str, Any],
        interpretation: Dict[str, Any],
        monitoring_log: Dict[str, Any],
        validation_warnings: List[str],
    ) -> Dict[str, Any]:
        """
        Assembles complete structured dictionary representation of the assessment report.
        """
        return {
            "report_metadata": {
                "agent_name": AGENT_NAME,
                "agent_version": AGENT_VERSION,
                "assessment_id": assessment_id,
                "patient_id": patient_id,
                "timestamp": timestamp,
            },
            "clinical_input_summary": features,
            "diagnostic_assessment": {
                "predicted_class": prediction_result["predicted_class"],
                "prediction_label": (
                    "Presence of Significant Coronary Artery Disease"
                    if prediction_result["predicted_class"] == 1
                    else "Absence of Significant Coronary Artery Disease"
                ),
                "risk_probability": prediction_result["risk_probability"],
                "risk_percentage": f"{prediction_result['risk_probability'] * 100:.1f}%",
                "risk_category": interpretation["risk_tier"],
                "risk_category_label": interpretation["risk_tier_label"],
                "decision_threshold": prediction_result["decision_threshold"],
                "model_used": prediction_result["model_name"],
            },
            "longitudinal_monitoring": {
                "trend_status": monitoring_log.get("trend_status", "FIRST_ASSESSMENT"),
                "trend_summary": monitoring_log.get("trend_summary", "Baseline assessment"),
                "risk_delta": monitoring_log.get("risk_delta", 0.0),
                "vital_deltas": monitoring_log.get("vital_deltas", {}),
                "prior_assessment_count": monitoring_log.get("prior_assessment_count", 0),
            },
            "clinical_attribution": {
                "elevating_risk_factors": interpretation["elevating_risk_factors"],
                "protective_factors": interpretation["protective_factors"],
            },
            "recommendations": interpretation["clinical_recommendations"],
            "validation_warnings": validation_warnings,
            "disclaimer": CLINICAL_DISCLAIMER,
        }

    @staticmethod
    def generate_markdown_report(report_dict: Dict[str, Any]) -> str:
        """
        Renders the structured report dictionary into a formatted Markdown clinical report.
        """
        meta = report_dict["report_metadata"]
        diag = report_dict["diagnostic_assessment"]
        longi = report_dict["longitudinal_monitoring"]
        inputs = report_dict["clinical_input_summary"]
        attr = report_dict["clinical_attribution"]
        recs = report_dict["recommendations"]
        warnings = report_dict.get("validation_warnings", [])
        disclaimer = report_dict["disclaimer"]

        # Risk Tier formatting color / icon
        tier = diag["risk_category"]
        tier_icons = {
            "LOW": "🟢 LOW RISK",
            "MODERATE": "🟡 MODERATE RISK",
            "HIGH": "🟠 HIGH RISK",
            "CRITICAL": "🔴 CRITICAL RISK",
        }
        tier_badge = tier_icons.get(tier, tier)

        sex_label = "Male" if inputs.get("sex") == 1 else "Female"
        cp_labels = PHYSIOLOGICAL_BOUNDS["cp"]["labels"]
        cp_str = f"{inputs.get('cp')} ({cp_labels.get(inputs.get('cp'), 'Unknown')})"

        # Format Elevating Factors
        if attr["elevating_risk_factors"]:
            elev_items = "\n".join(
                [f"- **{f['factor']}** (`{f['value']}`): {f['clinical_impact']}" for f in attr["elevating_risk_factors"]]
            )
        else:
            elev_items = "- *No major acute risk elevating drivers identified.*"

        # Format Protective Factors
        if attr["protective_factors"]:
            prot_items = "\n".join(
                [f"- **{f['factor']}** (`{f['value']}`): {f['clinical_impact']}" for f in attr["protective_factors"]]
            )
        else:
            prot_items = "- *No major protective factors noted.*"

        # Format Recommendations
        rec_items = "\n".join([f"{i+1}. {r}" for i, r in enumerate(recs)])

        # Format Warnings
        if warnings:
            warning_block = "\n> [!WARNING] **Clinical Input Alerts:**\n" + "\n".join([f"> - {w}" for w in warnings])
        else:
            warning_block = ""

        # Format Longitudinal Trend
        prior_count = longi.get("prior_assessment_count", 0)
        if prior_count > 0:
            v_deltas = longi.get("vital_deltas", {})
            trend_str = (
                f"- **Status:** {longi['trend_summary']}\n"
                f"- **Prior Encounters on Record:** {prior_count}\n"
                f"- **Shift in Risk Probability:** `{longi['risk_delta']*100:+.1f}%`\n"
                f"- **Vitals Shift:** BP Δ `{v_deltas.get('trestbps_delta', 0):+} mm Hg` | "
                f"Chol Δ `{v_deltas.get('chol_delta', 0):+} mg/dl` | "
                f"ST Depression Δ `{v_deltas.get('oldpeak_delta', 0.0):+} mm`"
            )
        else:
            trend_str = "- **Status:** Baseline assessment. No previous monitoring history on file."

        md = f"""# Cardiovascular Risk Assessment & Monitoring Report

**Patient ID:** `{meta['patient_id']}`  
**Assessment ID:** `{meta['assessment_id']}`  
**Evaluation Date / Time:** `{meta['timestamp']}`  
**System:** {meta['agent_name']} (v{meta['agent_version']})  
**Evaluating Model:** {diag['model_used']}  

---

## 1. Diagnostic Risk Stratification

| Clinical Evaluation Field | Assessment Finding |
| :--- | :--- |
| **Risk Classification Category** | **{tier_badge}** |
| **Estimated Coronary Disease Probability** | **{diag['risk_percentage']}** |
| **Model Binary Classification** | `{diag['predicted_class']}` — {diag['prediction_label']} |
| **Diagnostic Operating Threshold** | `{diag['decision_threshold']}` |

{warning_block}

---

## 2. Longitudinal Patient History & Monitoring Trend

{trend_str}

---

## 3. Patient Clinical Baseline Summary

| Parameter | Patient Value | Parameter | Patient Value |
| :--- | :--- | :--- | :--- |
| **Age** | {inputs.get('age')} years | **Biological Sex** | {sex_label} |
| **Chest Pain Type (cp)** | {cp_str} | **Resting Blood Pressure** | {inputs.get('trestbps')} mm Hg |
| **Serum Cholesterol** | {inputs.get('chol')} mg/dl | **Fasting Blood Sugar > 120** | {'Yes' if inputs.get('fbs')==1 else 'No'} |
| **Resting ECG** | {inputs.get('restecg')} | **Max Achieved Heart Rate** | {inputs.get('thalach')} bpm |
| **Exercise Angina** | {'Yes' if inputs.get('exang')==1 else 'No'} | **ST Depression (oldpeak)** | {inputs.get('oldpeak')} mm |
| **ST Slope** | {inputs.get('slope')} | **Fluoroscopy Vessels (ca)** | {inputs.get('ca')} vessel(s) |
| **Thalassemia Status** | Code {inputs.get('thal')} | — | — |

---

## 4. Key Contributing Clinical Factors

### Risk Elevating Factors
{elev_items}

### Protective / Favorable Cardiovascular Factors
{prot_items}

---

## 5. Recommended Clinical Management & Lifestyle Action Plan

{rec_items}

---

## 6. Regulatory & Medical Disclaimer

> [!IMPORTANT]
> {disclaimer}
"""
        return md
