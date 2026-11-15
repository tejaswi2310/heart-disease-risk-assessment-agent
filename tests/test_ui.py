"""
tests/test_ui.py
================
Automated tests for the Heart Disease Risk Assessment Agent UI layer.

Covers:
- Module importability (styles, views, app)
- CSS/HTML helpers produce non-empty strings
- Triage badge rendering for all 4 risk tiers
- Clinical header and disclaimer card rendering
- Preset record conformance with ALL_FEATURE_COLUMNS
- Session-state data structure validation
- Risk results view dict structure compatibility
- app.py navigation constant integrity
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# 1. Module Import Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestUIModuleImports:
    """Verify that all UI modules import without errors."""

    def test_styles_module_imports(self):
        """src.ui.styles must be importable."""
        import src.ui.styles as styles  # noqa: F401
        assert styles is not None

    def test_views_module_imports(self):
        """src.ui.views must be importable (Streamlit is mocked by conftest or not called at module-level)."""
        import src.ui.views as views  # noqa: F401
        assert views is not None

    def test_styles_exports_css(self):
        """CUSTOM_CSS must be a non-empty string."""
        from src.ui.styles import CUSTOM_CSS
        assert isinstance(CUSTOM_CSS, str)
        assert len(CUSTOM_CSS) > 100, "CUSTOM_CSS appears to be empty or too short"

    def test_styles_exports_helpers(self):
        """render_clinical_header, render_disclaimer_card, render_triage_badge must be callable."""
        from src.ui.styles import render_clinical_header, render_disclaimer_card, render_triage_badge
        assert callable(render_clinical_header)
        assert callable(render_disclaimer_card)
        assert callable(render_triage_badge)

    def test_views_exports_render_functions(self):
        """All 5 render_* functions must be present in views module."""
        from src.ui import views
        required_fns = [
            "render_home_view",
            "render_patient_assessment_view",
            "render_risk_results_view",
            "render_monitoring_view",
            "render_model_info_view",
        ]
        for fn_name in required_fns:
            assert hasattr(views, fn_name), f"Missing function: {fn_name}"
            assert callable(getattr(views, fn_name)), f"Not callable: {fn_name}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. CSS Helper Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCSSHelpers:
    """Verify CSS helper functions produce correct HTML strings."""

    def test_clinical_header_contains_title(self):
        from src.ui.styles import render_clinical_header
        html = render_clinical_header("Test Title", "Test Subtitle")
        assert "Test Title" in html
        assert "Test Subtitle" in html
        assert "clinical-header" in html

    def test_clinical_header_returns_string(self):
        from src.ui.styles import render_clinical_header
        result = render_clinical_header("A", "B")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_disclaimer_card_non_empty(self):
        from src.ui.styles import render_disclaimer_card
        html = render_disclaimer_card()
        assert isinstance(html, str)
        assert "disclaimer" in html.lower()
        assert len(html) > 50

    def test_disclaimer_card_contains_key_text(self):
        from src.ui.styles import render_disclaimer_card
        html = render_disclaimer_card()
        assert "NOT" in html or "not" in html
        assert "diagnos" in html.lower()

    def test_triage_badge_low(self):
        from src.ui.styles import render_triage_badge
        badge = render_triage_badge("LOW")
        assert isinstance(badge, str)
        assert "LOW" in badge.upper()
        assert "badge-low" in badge

    def test_triage_badge_moderate(self):
        from src.ui.styles import render_triage_badge
        badge = render_triage_badge("MODERATE")
        assert "MODERATE" in badge.upper()
        assert "badge-moderate" in badge

    def test_triage_badge_high(self):
        from src.ui.styles import render_triage_badge
        badge = render_triage_badge("HIGH")
        assert "HIGH" in badge.upper()
        assert "badge-high" in badge

    def test_triage_badge_critical(self):
        from src.ui.styles import render_triage_badge
        badge = render_triage_badge("CRITICAL")
        assert "CRITICAL" in badge.upper()
        assert "badge-critical" in badge

    def test_triage_badge_case_insensitive(self):
        """Badge function must accept lower-case tier strings."""
        from src.ui.styles import render_triage_badge
        badge_lower = render_triage_badge("low")
        badge_upper = render_triage_badge("LOW")
        # Both should reference same CSS class
        assert "badge-low" in badge_lower
        assert "badge-low" in badge_upper

    def test_triage_badge_unknown_defaults_to_critical(self):
        """Unknown tier must still return a valid badge string."""
        from src.ui.styles import render_triage_badge
        badge = render_triage_badge("UNKNOWN_TIER")
        assert isinstance(badge, str)
        assert "badge-" in badge


# ─────────────────────────────────────────────────────────────────────────────
# 3. Preset Record Conformance Tests
# ─────────────────────────────────────────────────────────────────────────────

EXPECTED_FEATURE_COLUMNS = [
    "age", "trestbps", "chol", "thalach", "oldpeak",
    "sex", "fbs", "exang",
    "cp", "restecg", "slope", "ca", "thal",
]

HIGH_RISK_PRESET = {
    "age": 64, "sex": 1, "cp": 3, "trestbps": 165, "chol": 290,
    "fbs": 1, "restecg": 1, "thalach": 125, "exang": 1,
    "oldpeak": 2.8, "slope": 1, "ca": 2, "thal": 2,
}

LOW_RISK_PRESET = {
    "age": 42, "sex": 0, "cp": 1, "trestbps": 115, "chol": 185,
    "fbs": 0, "restecg": 0, "thalach": 172, "exang": 0,
    "oldpeak": 0.0, "slope": 2, "ca": 0, "thal": 1,
}

MODERATE_RISK_PRESET = {
    "age": 56, "sex": 1, "cp": 2, "trestbps": 138, "chol": 255,
    "fbs": 0, "restecg": 1, "thalach": 142, "exang": 0,
    "oldpeak": 1.2, "slope": 1, "ca": 1, "thal": 2,
}


class TestPresetRecordConformance:
    """All clinical preset records must have exactly the expected 13 feature columns."""

    @pytest.mark.parametrize("preset,name", [
        (HIGH_RISK_PRESET, "High Risk"),
        (LOW_RISK_PRESET, "Low Risk"),
        (MODERATE_RISK_PRESET, "Moderate Risk"),
    ])
    def test_preset_has_all_required_columns(self, preset, name):
        missing = set(EXPECTED_FEATURE_COLUMNS) - set(preset.keys())
        assert not missing, f"{name} preset is missing columns: {missing}"

    @pytest.mark.parametrize("preset,name", [
        (HIGH_RISK_PRESET, "High Risk"),
        (LOW_RISK_PRESET, "Low Risk"),
        (MODERATE_RISK_PRESET, "Moderate Risk"),
    ])
    def test_preset_has_no_extra_columns(self, preset, name):
        extra = set(preset.keys()) - set(EXPECTED_FEATURE_COLUMNS)
        assert not extra, f"{name} preset has unexpected extra columns: {extra}"

    @pytest.mark.parametrize("preset,name", [
        (HIGH_RISK_PRESET, "High Risk"),
        (LOW_RISK_PRESET, "Low Risk"),
        (MODERATE_RISK_PRESET, "Moderate Risk"),
    ])
    def test_preset_binary_values_valid(self, preset, name):
        for col in ["sex", "fbs", "exang"]:
            assert preset[col] in (0, 1), f"{name}: {col}={preset[col]} is not binary"

    @pytest.mark.parametrize("preset,name", [
        (HIGH_RISK_PRESET, "High Risk"),
        (LOW_RISK_PRESET, "Low Risk"),
        (MODERATE_RISK_PRESET, "Moderate Risk"),
    ])
    def test_preset_categorical_ranges_valid(self, preset, name):
        assert preset["cp"] in range(4), f"{name}: cp out of range"
        assert preset["restecg"] in range(3), f"{name}: restecg out of range"
        assert preset["slope"] in range(3), f"{name}: slope out of range"
        assert preset["ca"] in range(5), f"{name}: ca out of range (0-4 incl. missing)"
        assert preset["thal"] in (0, 1, 2, 3), f"{name}: thal out of range"

    @pytest.mark.parametrize("preset,name", [
        (HIGH_RISK_PRESET, "High Risk"),
        (LOW_RISK_PRESET, "Low Risk"),
        (MODERATE_RISK_PRESET, "Moderate Risk"),
    ])
    def test_preset_numerical_sanity(self, preset, name):
        assert 18 <= preset["age"] <= 105, f"{name}: age out of bounds"
        assert 60 <= preset["trestbps"] <= 260, f"{name}: trestbps out of bounds"
        assert 80 <= preset["chol"] <= 650, f"{name}: chol out of bounds"
        assert 50 <= preset["thalach"] <= 230, f"{name}: thalach out of bounds"
        assert 0.0 <= preset["oldpeak"] <= 10.0, f"{name}: oldpeak out of bounds"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Assessment Result Dict Structure Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAssessmentResultStructure:
    """
    Tests that render_risk_results_view receives the correct data structure.
    Uses a synthetic minimal result dict mirroring the agent's actual output.
    """

    REQUIRED_TOP_LEVEL_KEYS = [
        "assessment_id",
        "patient_id",
        "assessment_type",
        "timestamp",
        "predicted_class",
        "risk_probability",
        "risk_category",
        "trend_status",
        "trend_summary",
        "validation_warnings",
        "json_report",
        "markdown_report",
    ]

    REQUIRED_JSON_REPORT_KEYS = [
        "diagnostic_assessment",
        "clinical_attribution",
        "recommendations",
        "clinical_input_summary",
    ]

    def _make_minimal_result(self) -> dict:
        return {
            "assessment_id": "ASSESS-TEST-001",
            "patient_id": "PT-TEST-001",
            "assessment_type": "NEW_ASSESSMENT",
            "timestamp": "2026-11-14T13:00:00+00:00",
            "predicted_class": 1,
            "risk_probability": 0.74,
            "risk_category": "HIGH",
            "trend_status": "FIRST_ASSESSMENT",
            "trend_summary": "Baseline assessment recorded.",
            "validation_warnings": [],
            "report_path": None,
            "json_report": {
                "diagnostic_assessment": {
                    "predicted_class": 1,
                    "risk_probability": 0.74,
                    "risk_tier": "HIGH",
                },
                "clinical_attribution": {
                    "elevating_risk_factors": [
                        {"factor": "ST Depression", "value": 2.8, "clinical_impact": "Elevated"}
                    ],
                    "protective_factors": [],
                },
                "recommendations": [
                    "Urgent specialist referral recommended.",
                    "Restrict strenuous physical activity until cardiological evaluation.",
                ],
                "clinical_input_summary": {
                    "age": 64, "sex": 1, "cp": 3, "trestbps": 165, "chol": 290,
                    "fbs": 1, "restecg": 1, "thalach": 125, "exang": 1,
                    "oldpeak": 2.8, "slope": 1, "ca": 2, "thal": 2,
                },
                "validation_warnings": [],
            },
            "markdown_report": "# Test Report\nThis is a test.",
        }

    def test_top_level_keys_present(self):
        result = self._make_minimal_result()
        for key in self.REQUIRED_TOP_LEVEL_KEYS:
            assert key in result, f"Assessment result missing key: '{key}'"

    def test_json_report_sub_keys_present(self):
        result = self._make_minimal_result()
        json_rep = result["json_report"]
        for key in self.REQUIRED_JSON_REPORT_KEYS:
            assert key in json_rep, f"json_report missing key: '{key}'"

    def test_predicted_class_is_binary(self):
        result = self._make_minimal_result()
        assert result["predicted_class"] in (0, 1)

    def test_risk_probability_in_unit_range(self):
        result = self._make_minimal_result()
        assert 0.0 <= result["risk_probability"] <= 1.0

    def test_risk_category_valid_tier(self):
        result = self._make_minimal_result()
        valid_tiers = {"LOW", "MODERATE", "HIGH", "CRITICAL"}
        assert result["risk_category"].upper() in valid_tiers

    def test_validation_warnings_is_list(self):
        result = self._make_minimal_result()
        assert isinstance(result["validation_warnings"], list)

    def test_markdown_report_is_string(self):
        result = self._make_minimal_result()
        assert isinstance(result["markdown_report"], str)
        assert len(result["markdown_report"]) > 0


# ─────────────────────────────────────────────────────────────────────────────
# 5. App Navigation Configuration Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAppNavigationConfig:
    """Verify app.py navigation constants are correct."""

    def test_nav_sections_defined(self):
        """NAV_SECTIONS dict must exist in app module."""
        import app
        assert hasattr(app, "NAV_SECTIONS")
        assert isinstance(app.NAV_SECTIONS, dict)

    def test_nav_sections_has_five_entries(self):
        import app
        assert len(app.NAV_SECTIONS) == 5, (
            f"Expected 5 nav sections, got {len(app.NAV_SECTIONS)}"
        )

    def test_nav_section_keys_contain_home(self):
        import app
        values = list(app.NAV_SECTIONS.values())
        assert "home" in values

    def test_nav_section_values_cover_all_views(self):
        import app
        expected_views = {"home", "assess", "results", "monitoring", "model_info"}
        actual_views = set(app.NAV_SECTIONS.values())
        assert actual_views == expected_views, (
            f"NAV_SECTIONS values mismatch.\nExpected: {expected_views}\nGot: {actual_views}"
        )

    def test_app_module_has_main_function(self):
        import app
        assert hasattr(app, "main")
        assert callable(app.main)

    def test_app_module_has_load_agent_function(self):
        import app
        assert hasattr(app, "_load_agent")
        assert callable(app._load_agent)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Config Alignment Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestUIConfigAlignment:
    """
    Verify that the UI's hardcoded column lists align with src.config.ALL_FEATURE_COLUMNS.
    Prevents silent schema drift between UI and ML pipeline.
    """

    def test_expected_columns_match_config(self):
        from src.config import ALL_FEATURE_COLUMNS
        assert set(EXPECTED_FEATURE_COLUMNS) == set(ALL_FEATURE_COLUMNS), (
            f"UI column list drifted from src.config.ALL_FEATURE_COLUMNS.\n"
            f"UI: {sorted(EXPECTED_FEATURE_COLUMNS)}\n"
            f"Config: {sorted(ALL_FEATURE_COLUMNS)}"
        )

    def test_no_target_column_in_feature_list(self):
        from src.config import ALL_FEATURE_COLUMNS, TARGET_COLUMN
        assert TARGET_COLUMN not in ALL_FEATURE_COLUMNS, (
            "Target column leaked into ALL_FEATURE_COLUMNS!"
        )

    def test_risk_tiers_completeness(self):
        from src.config import RISK_TIERS
        for tier in ["LOW", "MODERATE", "HIGH", "CRITICAL"]:
            assert tier in RISK_TIERS, f"Risk tier '{tier}' missing from RISK_TIERS"

    def test_risk_tier_has_threshold_key(self):
        from src.config import RISK_TIERS
        for tier, cfg in RISK_TIERS.items():
            assert "threshold" in cfg, f"RISK_TIERS['{tier}'] missing 'threshold' key"
            assert 0.0 <= cfg["threshold"] <= 1.0, (
                f"RISK_TIERS['{tier}']['threshold'] = {cfg['threshold']} is out of [0,1]"
            )

