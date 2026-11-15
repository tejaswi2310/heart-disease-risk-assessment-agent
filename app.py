"""
Heart Disease Risk Assessment Agent — Main Streamlit Application Entry Point
============================================================================
Coordinates navigation, session state, and agent initialization.
Every UI screen delegates to src/ui/views.py; no ML or agent logic lives here.

Usage:
    streamlit run app.py
"""

import sys
from pathlib import Path

# ── Ensure project root is on sys.path before any src.* import ──────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

# ── Page config must be the FIRST Streamlit call ────────────────────────────
st.set_page_config(
    page_title="Heart Disease Risk Assessment Agent",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/johnsmith88/heart-disease-dataset",
        "Report a bug": None,
        "About": (
            "**Heart Disease Risk Assessment Agent** v1.0\n\n"
            "Phase 1 ML Engine · Phase 2 AI Healthcare Monitoring Agent\n\n"
            "⚠️ This tool is for educational and decision-support purposes only. "
            "It does NOT diagnose or treat any medical condition."
        ),
    },
)

# ── Import UI helpers (after page config) ───────────────────────────────────
from src.ui.styles import CUSTOM_CSS
from src.ui.views import (
    render_home_view,
    render_model_info_view,
    render_monitoring_view,
    render_patient_assessment_view,
    render_risk_results_view,
)

# ── Inject custom CSS ────────────────────────────────────────────────────────
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ── Agent singleton: initialise once per browser session ────────────────────
def _load_agent():
    """Loads the HeartDiseaseRiskAssessmentAgent, caching it in session state."""
    if "agent" not in st.session_state:
        try:
            from src.agent.agent import HeartDiseaseRiskAssessmentAgent

            st.session_state["agent"] = HeartDiseaseRiskAssessmentAgent()
            st.session_state["agent_load_error"] = None
        except FileNotFoundError as exc:
            st.session_state["agent"] = None
            st.session_state["agent_load_error"] = (
                f"**Model artefact not found:** {exc}\n\n"
                "Run `python scripts/train_and_compare_models.py` from the project root "
                "to train and save `models/champion_model.joblib`."
            )
        except Exception as exc:  # noqa: BLE001
            st.session_state["agent"] = None
            st.session_state["agent_load_error"] = (
                f"**Unexpected error loading AI Agent:** {type(exc).__name__}: {exc}"
            )
    return st.session_state["agent"], st.session_state.get("agent_load_error")


# ── Sidebar navigation ───────────────────────────────────────────────────────
NAV_SECTIONS = {
    "🏥  Overview & Scope": "home",
    "🩺  Patient Assessment": "assess",
    "📊  Assessment Results": "results",
    "📈  Longitudinal Monitoring": "monitoring",
    "🔬  Model Benchmarks & Governance": "model_info",
}


def _build_sidebar() -> str:
    """Renders sidebar and returns the active section key."""
    with st.sidebar:
        st.markdown(
            """
            <div style='text-align:center; padding: 10px 0 18px 0;'>
                <span style='font-size:46px;'>🫀</span><br>
                <strong style='font-size:15px; color:#1E3A8A;'>
                    Heart Disease Risk<br>Assessment Agent
                </strong><br>
                <span style='font-size:11px; color:#64748B;'>v1.0 · Clinical Decision-Support</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()
        st.markdown(
            "<p style='font-size:11px; color:#94A3B8; font-weight:600; "
            "text-transform:uppercase; margin-bottom:4px;'>Navigation</p>",
            unsafe_allow_html=True,
        )

        # Initialise default nav selection
        if "nav_selection" not in st.session_state:
            st.session_state["nav_selection"] = "🏥  Overview & Scope"

        for label in NAV_SECTIONS:
            is_active = st.session_state["nav_selection"] == label
            btn_type = "primary" if is_active else "secondary"
            if st.button(label, key=f"nav_{label}", use_container_width=True, type=btn_type):
                st.session_state["nav_selection"] = label
                st.rerun()

        st.divider()

        # Quick-action: jump to results if assessment available
        if st.session_state.get("latest_assessment"):
            if st.button("📊  View Latest Results", use_container_width=True):
                st.session_state["nav_selection"] = "📊  Assessment Results"
                st.rerun()

        # Agent status indicator
        agent, err = _load_agent()
        if agent:
            st.success("🟢 AI Agent: Online", icon=None)
        else:
            st.error("🔴 AI Agent: Offline")

        st.divider()
        st.markdown(
            """
            <div style='font-size:10.5px; color:#94A3B8; line-height:1.5;'>
            ⚠️ <strong>Medical Disclaimer:</strong><br>
            This system provides automated risk stratification only.
            It does <strong>not</strong> diagnose heart disease.<br>
            Always consult a licensed cardiologist.
            </div>
            """,
            unsafe_allow_html=True,
        )

    return NAV_SECTIONS[st.session_state["nav_selection"]]


# ── Main router ──────────────────────────────────────────────────────────────
def main():
    agent, agent_error = _load_agent()
    active_section = _build_sidebar()

    # Show persistent agent error banner (non-fatal for static pages)
    if agent_error and active_section in ("assess", "results", "monitoring"):
        st.error(agent_error)
        st.stop()

    # ── Route to view ────────────────────────────────────────────────────────
    if active_section == "home":
        render_home_view()

    elif active_section == "assess":
        render_patient_assessment_view(agent)

    elif active_section == "results":
        result = st.session_state.get("latest_assessment")
        if result is None:
            st.markdown(
                """
                <div style='text-align:center; padding:60px 20px;'>
                    <span style='font-size:52px;'>📋</span><br><br>
                    <h3 style='color:#1E3A8A;'>No Assessment on Record</h3>
                    <p style='color:#64748B;'>
                        No patient assessment has been run in this session.<br>
                        Navigate to <strong>Patient Assessment</strong> to submit a clinical record.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("🩺  Go to Patient Assessment", type="primary"):
                st.session_state["nav_selection"] = "🩺  Patient Assessment"
                st.rerun()
        else:
            render_risk_results_view(result)

    elif active_section == "monitoring":
        render_monitoring_view(agent)

    elif active_section == "model_info":
        render_model_info_view()


if __name__ == "__main__":
    main()

