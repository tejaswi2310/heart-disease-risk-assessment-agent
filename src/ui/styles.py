"""
Custom styling and CSS themes for the Heart Disease Risk Assessment Agent UI.
Applies clean, accessible, hospital-grade typography, metric cards, and triage badges.
"""

CUSTOM_CSS = """
<style>
    /* Main container and font settings */
    .main {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header brand banner */
    .clinical-header {
        background: linear-gradient(135deg, #1E3A8A 0%, #0F766E 100%);
        padding: 24px 28px;
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .clinical-header h1 {
        color: #FFFFFF !important;
        font-size: 28px;
        font-weight: 700;
        margin: 0 0 6px 0;
    }
    .clinical-header p {
        color: #E2E8F0 !important;
        font-size: 14px;
        margin: 0;
    }

    /* Disclaimer Card */
    .disclaimer-card {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 14px 18px;
        border-radius: 6px;
        margin-bottom: 20px;
        color: #92400E;
        font-size: 13px;
        line-height: 1.5;
    }
    .disclaimer-card strong {
        color: #78350F;
    }

    /* Triage Badges */
    .badge-low {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 14px;
        display: inline-block;
        border: 1px solid #31C48D;
    }
    .badge-moderate {
        background-color: #FEF08A;
        color: #854D0E;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 14px;
        display: inline-block;
        border: 1px solid #FACC15;
    }
    .badge-high {
        background-color: #FFEDD5;
        color: #9A3412;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 14px;
        display: inline-block;
        border: 1px solid #FB923C;
    }
    .badge-critical {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 14px;
        display: inline-block;
        border: 1px solid #F98080;
    }

    /* Summary Card */
    .clinical-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 18px 20px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .clinical-card h4 {
        margin-top: 0;
        color: #1E293B;
        font-size: 16px;
        border-bottom: 1px solid #F1F5F9;
        padding-bottom: 8px;
    }

    /* Metric Callout Box */
    .metric-callout {
        background: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
    }
    .metric-callout .val {
        font-size: 26px;
        font-weight: 800;
        color: #0F172A;
    }
    .metric-callout .lbl {
        font-size: 12px;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 600;
        margin-top: 4px;
    }

    /* Distinction Box: Prediction vs Risk vs Diagnosis */
    .distinction-box {
        background: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 16px 0;
        font-size: 13.5px;
    }
    .distinction-box strong {
        color: #166534;
    }
</style>
"""


def render_clinical_header(title: str, subtitle: str) -> str:
    return f"""
    <div class="clinical-header">
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """


def render_disclaimer_card() -> str:
    return """
    <div class="disclaimer-card">
        <strong>⚠️ Clinical Decision-Support & Medical Disclaimer:</strong><br>
        This platform is an automated machine-learning-based risk stratification and decision-support system. 
        It does <strong>NOT</strong> provide a definitive medical diagnosis of coronary artery disease, acute coronary syndrome, 
        or myocardial infarction. All predictions and calculated probabilities must be independently reviewed and interpreted 
        by a licensed cardiologist or attending physician alongside formal diagnostic evaluations (e.g., electrocardiograms, 
        cardiac biomarkers, stress imaging, and angiography).
    </div>
    """


def render_triage_badge(tier: str) -> str:
    tier_upper = tier.upper()
    if tier_upper == "LOW":
        return '<span class="badge-low">🟢 LOW RISK (&le; 30%)</span>'
    elif tier_upper == "MODERATE":
        return '<span class="badge-moderate">🟡 MODERATE RISK (31% - 60%)</span>'
    elif tier_upper == "HIGH":
        return '<span class="badge-high">🟠 HIGH RISK (61% - 85%)</span>'
    else:
        return '<span class="badge-critical">🔴 CRITICAL RISK (&gt; 85%)</span>'

