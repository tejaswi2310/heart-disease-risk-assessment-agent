"""
View components and screen renderers for the Heart Disease Risk Assessment Agent UI.
"""

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.agent.config import (
    AGENT_NAME,
    AGENT_VERSION,
    CLINICAL_DISCLAIMER,
    PHYSIOLOGICAL_BOUNDS,
    RISK_TIERS,
)
from src.agent.validation import PatientValidationError
from src.config import (
    CHAMPION_METADATA_PATH,
    FIGURES_DIR,
    MODEL_RESULTS_CSV_PATH,
    PROJECT_ROOT,
)
from src.ui.styles import (
    render_clinical_header,
    render_disclaimer_card,
    render_triage_badge,
)


def render_home_view():
    """Renders Home & Overview section."""
    st.markdown(
        render_clinical_header(
            "Heart Disease Risk Assessment Agent",
            "Intelligent Clinical Decision-Support & Longitudinal Cardiovascular Monitoring System",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(render_disclaimer_card(), unsafe_allow_html=True)

    # Core Distinction Box
    st.markdown(
        """
        <div class="distinction-box">
            <h4>Clinical Terminology & Scope Differentiation</h4>
            <ul>
                <li><strong>Statistical Prediction:</strong> The raw algorithmic output of the supervised machine learning model (e.g. Class <code>1</code> vs <code>0</code>, Posterior Probability <code>87.9%</code>).</li>
                <li><strong>Risk Assessment:</strong> The clinical contextualization of the model's output into standardized triage tiers (Low, Moderate, High, Critical), identifying patient-specific risk drivers and lifestyle guidance.</li>
                <li><strong>Medical Diagnosis:</strong> A formal, authoritative clinical determination made exclusively by a licensed physician using physical examination, comprehensive history, laboratory assays, and cardiac imaging. <em>This system does NOT diagnose or treat patients.</em></li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            ### 🔬 Phase 1 — Machine Learning Engine
            - **Benchmark Across 6 Algorithms:** Rigorously compared Logistic Regression, Random Forest, XGBoost, Support Vector Machine, K-Nearest Neighbors, and Multi-Layer Perceptron.
            - **Leak-Free Methodology:** Unmasked **723 duplicate rows (70.5%)** in the raw Kaggle dataset; deduplicated down to 302 true clinical cases before train/test splitting to prevent artificial ~99% test leakage.
            - **Champion Selection:** **Support Vector Machine (Linear Kernel, C=0.1)** achieved the highest clinical sensitivity (**87.88% Recall**, only 4 false negatives) and lowest generalization drop (**3.1% overfitting gap**).
            """
        )

    with col2:
        st.markdown(
            """
            ### 🤖 Phase 2 — AI Healthcare Monitoring Agent
            - **Clinical Intake & Validation:** Enforces physiological limits across all 13 cardiac predictors and flags crisis vitals.
            - **Risk Attribution:** Evaluates patient-specific risk drivers (e.g. ST depression, vessel stenosis) and favorable protective factors.
            - **Longitudinal Audit Logging:** Tracks cardiovascular trajectories over time, detecting improvement or deterioration across follow-up visits.
            - **EHR-Ready Reports:** Exports structured JSON payloads and formatted consultation Markdown documents.
            """
        )

    st.info("💡 **Getting Started:** Navigate to **Patient Assessment** in the left sidebar to enter patient vitals or load clinical presets.")


def render_patient_assessment_view(agent):
    """Renders Patient Intake & Assessment form."""
    st.markdown(
        render_clinical_header(
            "Patient Clinical Intake & Evaluation",
            "Enter cardiovascular markers, exercise stress data, and fluoroscopy findings",
        ),
        unsafe_allow_html=True,
    )

    # Clinical Presets for rapid evaluation
    preset_choice = st.selectbox(
        "📋 Quick-Fill Clinical Presets (Optional Demonstration Profiles):",
        [
            "Select a preset or enter manually...",
            "Preset 1: High Risk (Severe ST Depression, Hypertension, Vessel Stenosis)",
            "Preset 2: Low Risk (Optimal BP, Normal ECG, Clear Vessels)",
            "Preset 3: Moderate Risk (Borderline Cholesterol, Atypical Angina)",
        ],
    )

    default_vals = {
        "patient_id": "PT-DEMO-2026",
        "age": 55,
        "sex": 1,
        "cp": 0,
        "trestbps": 130,
        "chol": 240,
        "fbs": 0,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 0.8,
        "slope": 1,
        "ca": 0,
        "thal": 2,
    }

    if "High Risk" in preset_choice:
        default_vals.update({
            "patient_id": "PT-HIGH-RISK-01",
            "age": 64,
            "sex": 1,
            "cp": 3,
            "trestbps": 165,
            "chol": 290,
            "fbs": 1,
            "restecg": 1,
            "thalach": 125,
            "exang": 1,
            "oldpeak": 2.8,
            "slope": 1,
            "ca": 2,
            "thal": 2,
        })
    elif "Low Risk" in preset_choice:
        default_vals.update({
            "patient_id": "PT-LOW-RISK-01",
            "age": 42,
            "sex": 0,
            "cp": 1,
            "trestbps": 115,
            "chol": 185,
            "fbs": 0,
            "restecg": 0,
            "thalach": 172,
            "exang": 0,
            "oldpeak": 0.0,
            "slope": 2,
            "ca": 0,
            "thal": 1,
        })
    elif "Moderate Risk" in preset_choice:
        default_vals.update({
            "patient_id": "PT-MODERATE-01",
            "age": 56,
            "sex": 1,
            "cp": 2,
            "trestbps": 138,
            "chol": 255,
            "fbs": 0,
            "restecg": 1,
            "thalach": 142,
            "exang": 0,
            "oldpeak": 1.2,
            "slope": 1,
            "ca": 1,
            "thal": 2,
        })

    with st.form("patient_assessment_form"):
        # Section 1: Demographics & Identifiers
        st.subheader("1. Patient Demographics & Identifiers")
        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            patient_id = st.text_input(
                "Patient Identifier",
                value=default_vals["patient_id"],
                help="Unique patient ID. If blank, an anonymous pseudonym will be generated.",
            )
        with col_d2:
            age = st.number_input("Age (years)", min_value=18, max_value=105, value=default_vals["age"], step=1)
        with col_d3:
            sex_label = st.radio(
                "Biological Sex",
                options=["Male", "Female"],
                index=0 if default_vals["sex"] == 1 else 1,
                horizontal=True,
            )
            sex = 1 if sex_label == "Male" else 0

        # Section 2: Resting Vitals & Blood Chemistry
        st.subheader("2. Resting Hemodynamics & Laboratory Chemistry")
        col_v1, col_v2, col_v3 = st.columns(3)
        with col_v1:
            trestbps = st.number_input(
                "Resting Blood Pressure (mm Hg)",
                min_value=60,
                max_value=260,
                value=default_vals["trestbps"],
                step=1,
                help="Hospital admission systolic blood pressure in mm Hg.",
            )
        with col_v2:
            chol = st.number_input(
                "Serum Cholesterol (mg/dl)",
                min_value=80,
                max_value=650,
                value=default_vals["chol"],
                step=1,
                help="Fasting total serum cholesterol level in mg/dl.",
            )
        with col_v3:
            fbs_choice = st.radio(
                "Fasting Blood Sugar > 120 mg/dl",
                options=["No (<= 120 mg/dl)", "Yes (> 120 mg/dl)"],
                index=default_vals["fbs"],
                horizontal=True,
                help="Indicator for elevated fasting blood glucose / diabetic state.",
            )
            fbs = 1 if "Yes" in fbs_choice else 0

        # Section 3: Electrocardiogram & Symptoms
        st.subheader("3. Clinical Presentation & Resting Electrocardiogram")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            cp_options = {
                0: "0: Typical Angina (Substernal exertional chest pressure)",
                1: "1: Atypical Angina (Non-classical exertional pain)",
                2: "2: Non-anginal Chest Pain (Sharp, position-dependent)",
                3: "3: Asymptomatic (No overt angina; silent presentation)",
            }
            cp = st.selectbox(
                "Chest Pain Type (cp)",
                options=list(cp_options.keys()),
                format_func=lambda x: cp_options[x],
                index=default_vals["cp"],
            )
        with col_s2:
            ecg_options = {
                0: "0: Normal Resting ECG",
                1: "1: ST-T Wave Abnormality (T-wave inversions / ST depression > 0.05 mV)",
                2: "2: Left Ventricular Hypertrophy (Estes criteria)",
            }
            restecg = st.selectbox(
                "Resting Electrocardiographic Results (restecg)",
                options=list(ecg_options.keys()),
                format_func=lambda x: ecg_options[x],
                index=default_vals["restecg"],
            )

        # Section 4: Exercise Stress Testing & Fluoroscopy
        st.subheader("4. Exercise Stress Test & Fluoroscopic Markers")
        col_e1, col_e2, col_e3 = st.columns(3)
        with col_e1:
            thalach = st.number_input(
                "Maximum Heart Rate Achieved (bpm)",
                min_value=50,
                max_value=230,
                value=default_vals["thalach"],
                step=1,
                help="Peak heart rate reached during cardiovascular exercise treadmill test.",
            )
        with col_e2:
            exang_choice = st.radio(
                "Exercise-Induced Angina (exang)",
                options=["No (Negative)", "Yes (Positive)"],
                index=default_vals["exang"],
                horizontal=True,
                help="Whether physical exercise triggered acute ischemic chest angina.",
            )
            exang = 1 if "Yes" in exang_choice else 0
        with col_e3:
            oldpeak = st.number_input(
                "ST Depression Induced by Exercise (mm)",
                min_value=0.0,
                max_value=10.0,
                value=float(default_vals["oldpeak"]),
                step=0.1,
                format="%.1f",
                help="Magnitude of ST segment depression relative to baseline rest.",
            )

        col_e4, col_e5, col_e6 = st.columns(3)
        with col_e4:
            slope_options = {
                0: "0: Upsloping (Rapid upstroke)",
                1: "1: Flat (Horizontal ST profile)",
                2: "2: Downsloping (Declining ST segment)",
            }
            slope = st.selectbox(
                "Slope of Peak Exercise ST Segment (slope)",
                options=list(slope_options.keys()),
                format_func=lambda x: slope_options[x],
                index=default_vals["slope"],
            )
        with col_e5:
            ca_options = {
                0: "0: 0 Major Vessels Stained",
                1: "1: 1 Major Vessel Stained",
                2: "2: 2 Major Vessels Stained",
                3: "3: 3 Major Vessels Stained",
                4: "4: Unrecorded / Missing Fluoroscopy (Auto-Imputed)",
            }
            ca = st.selectbox(
                "Major Vessels Colored by Fluoroscopy (ca)",
                options=list(ca_options.keys()),
                format_func=lambda x: ca_options[x],
                index=default_vals["ca"],
                help="Number of major coronary arteries (0-3) visualized with fluoroscopic contrast.",
            )
        with col_e6:
            thal_options = {
                1: "1: Normal Myocardial Perfusion",
                2: "2: Fixed Perfusion Defect (Prior Infarction/Scar)",
                3: "3: Reversible Perfusion Defect (Inducible Ischemia)",
                0: "0: Unrecorded Thalassemia Status (Auto-Imputed)",
            }
            thal = st.selectbox(
                "Thalassemia Blood / Perfusion Status (thal)",
                options=list(thal_options.keys()),
                format_func=lambda x: thal_options[x],
                index=list(thal_options.keys()).index(default_vals["thal"]),
            )

        submit_btn = st.form_submit_button("🚀 Run AI Agent Cardiovascular Assessment", type="primary", use_container_width=True)

    if submit_btn:
        patient_dict = {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal,
        }

        try:
            with st.spinner("Executing clinical intake, validation, and AI Agent inference..."):
                assessment_result = agent.assess_patient(
                    patient_record=patient_dict,
                    patient_id=patient_id.strip() if patient_id.strip() else None,
                    save_report=True,
                )

            st.session_state["latest_assessment"] = assessment_result
            st.success("✅ Assessment completed successfully! View the results below or in the 'Assessment Results' tab.")
            render_risk_results_view(assessment_result)

        except PatientValidationError as pve:
            st.error(f"🛑 **Clinical Validation Error:** {pve}")
            st.warning("Please correct the flagged input parameters above and resubmit.")
        except Exception as e:
            st.error(f"⚠️ **System Error during assessment:** {str(e)}")


def render_risk_results_view(assessment_result: Dict[str, Any]):
    """Renders the Risk Assessment and Health-Risk Report view."""
    res = assessment_result
    json_rep = res["json_report"]
    diag = json_rep["diagnostic_assessment"]
    attr = json_rep["clinical_attribution"]
    recs = json_rep["recommendations"]
    warnings = json_rep.get("validation_warnings", [])

    st.markdown(
        render_clinical_header(
            "Cardiovascular Health-Risk Assessment Report",
            f"Patient ID: {res['patient_id']} | Assessment ID: {res['assessment_id']} | Mode: {res['assessment_type']}",
        ),
        unsafe_allow_html=True,
    )

    # Triage Banner
    st.markdown(render_disclaimer_card(), unsafe_allow_html=True)

    # Key Metrics Row
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.markdown(
            f"""
            <div class="metric-callout">
                <div class="val">{render_triage_badge(res['risk_category'])}</div>
                <div class="lbl">Triage Category</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col2:
        st.markdown(
            f"""
            <div class="metric-callout">
                <div class="val">{res['risk_probability']*100:.1f}%</div>
                <div class="lbl">Calculated Risk Probability</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col3:
        pred_label = "Disease Present" if res["predicted_class"] == 1 else "No Disease"
        st.markdown(
            f"""
            <div class="metric-callout">
                <div class="val">{pred_label}</div>
                <div class="lbl">Model Classification (Thr=0.50)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col4:
        st.markdown(
            f"""
            <div class="metric-callout">
                <div class="val">{res['trend_status']}</div>
                <div class="lbl">Longitudinal Trajectory</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Display Validation Warnings if present
    if warnings:
        st.warning("⚠️ **Physiological Alerts & Inconsistencies Detected:**\n" + "\n".join([f"- {w}" for w in warnings]))

    # Longitudinal Comparison if Follow-up
    if res["assessment_type"] == "FOLLOWUP_ASSESSMENT":
        st.info(f"📈 **Longitudinal Tracking Update:** {res['trend_summary']}")

    # Tabs for Report Details
    tab_summary, tab_drivers, tab_recs, tab_export = st.tabs(
        ["📋 Clinical Summary", "🔍 Contributing Factors", "💡 Clinical Guidance", "📥 Export Consultation Report"]
    )

    with tab_summary:
        st.markdown("### Patient Vitals Snapshot")
        inputs = json_rep["clinical_input_summary"]
        df_inputs = pd.DataFrame([inputs]).T.rename(columns={0: "Recorded Value"})
        st.dataframe(df_inputs, use_container_width=True)

    with tab_drivers:
        col_elev, col_prot = st.columns(2)
        with col_elev:
            st.markdown("#### 🚨 Primary Risk Elevating Factors")
            if attr["elevating_risk_factors"]:
                for factor in attr["elevating_risk_factors"]:
                    st.markdown(f"• **{factor['factor']}** (`{factor['value']}`): {factor['clinical_impact']}")
            else:
                st.write("*No acute high-risk drivers identified.*")

        with col_prot:
            st.markdown("#### 🛡️ Favorable / Protective Indicators")
            if attr["protective_factors"]:
                for factor in attr["protective_factors"]:
                    st.markdown(f"• **{factor['factor']}** (`{factor['value']}`): {factor['clinical_impact']}")
            else:
                st.write("*No major protective factors noted.*")

    with tab_recs:
        st.markdown("### Evidence-Based Management Recommendations (AHA/ACC Aligned)")
        for i, rec in enumerate(recs):
            st.markdown(f"**{i+1}.** {rec}")

    with tab_export:
        st.markdown("### Export Consultation Records")
        c_exp1, c_exp2 = st.columns(2)
        with c_exp1:
            st.download_button(
                label="📄 Download Full Markdown Report (.md)",
                data=res["markdown_report"],
                file_name=f"{res['patient_id']}_{res['assessment_id']}_report.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with c_exp2:
            st.download_button(
                label="💾 Download Structured JSON Data (.json)",
                data=json.dumps(json_rep, indent=2),
                file_name=f"{res['patient_id']}_{res['assessment_id']}_data.json",
                mime="application/json",
                use_container_width=True,
            )


def render_monitoring_view(agent):
    """Renders longitudinal monitoring history and risk trends over time."""
    st.markdown(
        render_clinical_header(
            "Longitudinal Patient Monitoring & Audit Trail",
            "Track historical risk shifts, vitals progression, and therapeutic response over time",
        ),
        unsafe_allow_html=True,
    )

    all_records = agent.monitoring_service.get_all_records()
    if not all_records:
        st.info("ℹ️ No patient monitoring records found. Perform a patient assessment first to initialize the audit log.")
        return

    # Extract unique patient IDs
    patient_ids = sorted(list({r.get("patient_id") for r in all_records if r.get("patient_id")}))

    selected_pid = st.selectbox(
        "🔎 Select Patient Identifier to View Historical Progression:",
        options=patient_ids,
        index=0,
    )

    history = agent.get_monitoring_history(selected_pid)

    if history["status"] == "NO_RECORDS_FOUND" or not history["trajectory"]:
        st.warning(f"No records found for patient '{selected_pid}'.")
        return

    # Metrics Summary Row
    h_col1, h_col2, h_col3, h_col4 = st.columns(4)
    with h_col1:
        st.metric("Total Encounters", history["assessment_count"])
    with h_col2:
        st.metric("Baseline Risk", f"{history['baseline_risk']*100:.1f}%")
    with h_col3:
        st.metric("Latest Risk", f"{history['latest_risk']*100:.1f}%")
    with h_col4:
        delta_pct = history["overall_risk_shift"] * 100
        st.metric("Net Risk Shift", f"{delta_pct:+.1f}%", delta=f"{delta_pct:+.1f}%", delta_color="inverse")

    # Longitudinal Trajectory Chart
    st.subheader("Cardiovascular Risk Trajectory Over Time")
    traj = history["trajectory"]

    df_traj = pd.DataFrame(traj)
    if len(df_traj) > 1:
        df_traj["encounter"] = [f"Visit {i+1}" for i in range(len(df_traj))]
        df_traj["Risk %"] = df_traj["risk_probability"] * 100

        # Create plot
        fig, ax = plt.subplots(figsize=(9, 3.8))
        ax.plot(df_traj["encounter"], df_traj["Risk %"], marker="o", color="#E74C3C", linewidth=2.5, label="Risk Probability (%)")
        ax.axhline(30, color="#2ECC71", linestyle="--", alpha=0.7, label="Low Risk Threshold (30%)")
        ax.axhline(60, color="#F39C12", linestyle="--", alpha=0.7, label="Moderate Risk Threshold (60%)")
        ax.axhline(85, color="#C0392B", linestyle="--", alpha=0.7, label="High Risk Threshold (85%)")
        ax.set_ylim(0, 100)
        ax.set_ylabel("Risk Probability (%)", fontweight="bold")
        ax.set_title(f"Patient {selected_pid}: Longitudinal Risk Trajectory", fontweight="bold", fontsize=11)
        ax.legend(loc="upper right", fontsize=8.5)
        for i, r in enumerate(df_traj["Risk %"]):
            ax.annotate(f"{r:.1f}%", (i, r + 3), ha="center", fontweight="bold", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.info("ℹ️ Only one baseline encounter is on file for this patient. Trajectory curves will appear as follow-up assessments are logged.")

    # Historical Encounter Table
    st.subheader("Chronological Encounter Log")
    df_history = pd.DataFrame(agent.monitoring_service.get_patient_history(selected_pid))
    cols_to_show = ["assessment_id", "timestamp", "prediction", "risk_probability", "risk_category", "trend_status"]
    available_cols = [c for c in cols_to_show if c in df_history.columns]
    st.dataframe(df_history[available_cols], use_container_width=True)


def render_model_info_view():
    """Renders Phase 1 Model Architecture, Benchmark Comparison, and Governance."""
    st.markdown(
        render_clinical_header(
            "Machine Learning Architecture & Diagnostic Benchmarking",
            "Model selection rationale, evaluation metrics, feature attribution, and governance",
        ),
        unsafe_allow_html=True,
    )

    # 1. Model Champion Specs
    st.markdown(
        """
        ### 🏆 Selected Champion Model: Support Vector Machine (SVM)
        - **Algorithm:** Support Vector Classifier with Linear Kernel (`C = 0.1`, `kernel = 'linear'`).
        - **Selection Rationale:** Prioritized **Recall / Sensitivity (87.88%)** to minimize fatal False Negatives (only 4 missed cases out of 33 diseased holdout test patients), while demonstrating the lowest overfitting gap (**3.13%**).
        - **Pipeline Integration:** Serialized together with Scikit-Learn `ColumnTransformer` (median/mode imputation, robust scaling, one-hot encoding) in `models/champion_model.joblib`.
        """
    )

    # 2. Benchmark Metrics Table
    if MODEL_RESULTS_CSV_PATH.exists():
        st.subheader("Model Performance Comparison (Holdout Test Set N=61)")
        df_bench = pd.read_csv(MODEL_RESULTS_CSV_PATH)
        st.dataframe(df_bench, use_container_width=True)

    # 3. Diagnostic Visualizations
    st.subheader("Clinical Diagnostic Visualizations")
    tab_roc, tab_cm, tab_fi, tab_overfit = st.tabs(
        ["ROC Curves", "Confusion Matrices", "Feature Attribution", "Overfitting Audit"]
    )

    with tab_roc:
        roc_img = FIGURES_DIR / "roc_curves_comparison.png"
        if roc_img.exists():
            st.image(str(roc_img), caption="Combined Receiver Operating Characteristic (ROC) Curves across all 6 models.")

    with tab_cm:
        cm_img = FIGURES_DIR / "confusion_matrices.png"
        if cm_img.exists():
            st.image(str(cm_img), caption="Grid of 6 confusion matrices highlighting False Negatives in clinical context.")

    with tab_fi:
        fi_img = FIGURES_DIR / "feature_importances.png"
        if fi_img.exists():
            st.image(str(fi_img), caption="Top clinical predictive drivers for Random Forest, XGBoost, and Logistic Regression / SVM.")

    with tab_overfit:
        overfit_img = FIGURES_DIR / "train_vs_test_overfitting.png"
        if overfit_img.exists():
            st.image(str(overfit_img), caption="Training vs. 5-Fold CV vs. Holdout Test set accuracy comparison.")

    # 4. Clinical Limitations & Governance Boundaries
    st.subheader("Clinical Governance & Model Limitations")
    st.markdown(
        """
        1. **Sample Cohort Size:** The model was trained on the deduplicated Cleveland cohort ($N = 302$ unique patients). While methodologically leak-free, validation on modern multi-center demographic cohorts is recommended before acute hospital deployment.
        2. **Threshold Boundary:** The ground-truth angiographic target reflects ≥ 50% coronary artery stenosis. It does not grade microvascular ischemia or non-obstructive plaque vulnerability.
        3. **Decision-Support Constraint:** Machine learning predictions are assistive and probabilistic; they must never replace direct physical examination, physician judgment, or formal electrocardiographic/imaging workups.
        """
    )

