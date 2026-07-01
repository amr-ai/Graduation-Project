"""
Tests for the shared reporting layer: grounding / faithfulness checks,
markdown → HTML export, and the deterministic forecast-report builder.
"""

from __future__ import annotations

from agents.reporting.grounding import check_grounding, extract_claims
from agents.reporting.export import build_html_report, markdown_to_html
from agents.forecasting.report import build_forecast_report_md


_PAYLOAD = {
    "kpi": {
        "revenue": 128450.75,
        "orders": 3120,
        "aov": 41.17,
        "total_customers": 1875,
        "monthly_growth_avg": 0.083,  # stored as a fraction -> 8.3%
        "top_products": {"Latte": 42000.5, "Espresso": 33110.0},
    },
    "rfm": {"segments": {"Champions": {"revenue_pct": 34.2, "count": 210}}},
}


# ── Grounding ───────────────────────────────────────────────────────────────

def test_grounding_verifies_real_figures():
    report = (
        "Revenue reached $128,450 across 3,120 orders with an AOV of $41.17. "
        "Champions drive 34.2% of revenue."
    )
    g = check_grounding(report, _PAYLOAD)
    assert g.total == 4
    assert g.verified_count == 4
    assert g.status == "clean"
    assert g.coverage == 1.0


def test_grounding_flags_invented_figures():
    report = "Our churn rate is 27% versus an industry benchmark of 15%."
    g = check_grounding(report, _PAYLOAD)
    assert g.total == 2
    assert g.verified_count == 0
    assert len(g.unverified) == 2
    assert g.status == "weak"


def test_grounding_percent_stored_as_fraction():
    g = check_grounding("Monthly growth is around 8.3%.", _PAYLOAD)
    assert g.verified_count == 1


def test_grounding_currency_magnitude_suffix():
    g = check_grounding("Top line is about $1.2M this year.", {"revenue": 1_200_000})
    assert g.total == 1
    assert g.verified_count == 1


def test_grounding_ignores_years_and_small_counts():
    # "10" and "2024" are structural, not data claims -> not counted.
    g = check_grounding("The top 10 products in 2024 performed well.", {})
    assert g.total == 0
    assert g.coverage == 1.0


def test_grounding_empty_report_is_neutral():
    g = check_grounding("", _PAYLOAD)
    assert g.total == 0
    assert g.status == "none"
    assert g.coverage == 1.0


def test_extract_claims_dedupes():
    claims = extract_claims("$100,000 ... again $100,000 and 50%")
    kinds = sorted(c.kind for c in claims)
    assert kinds == ["currency", "percent"]  # duplicate currency collapsed


# ── Markdown → HTML ─────────────────────────────────────────────────────────

def test_markdown_to_html_blocks():
    md = (
        "# Title\n\nSome **bold** and *italic* and `code`.\n\n"
        "- a\n- b\n\n| X | Y |\n| --- | --- |\n| 1 | 2 |\n\n---\nEnd."
    )
    html = markdown_to_html(md)
    assert "<h1>Title</h1>" in html
    assert "<strong>bold</strong>" in html
    assert "<em>italic</em>" in html
    assert "<code>code</code>" in html
    assert "<ul>" in html and "<li>a</li>" in html
    assert "<table>" in html and "<th>X</th>" in html and "<td>1</td>" in html
    assert "<hr>" in html


def test_markdown_escapes_raw_html():
    html = markdown_to_html("Danger <script>alert(1)</script> here")
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_build_html_report_is_self_contained():
    g = check_grounding("Revenue was $128,450.", _PAYLOAD)
    doc = build_html_report(
        "Business Insights",
        "## Summary\nRevenue was **$128,450**.",
        subtitle="Test",
        meta={"Agent": "Insights", "Rows": "3,120"},
        kpis=[("Revenue", "$128,451")],
        grounding=g,
    )
    assert doc.startswith("<!DOCTYPE html>")
    assert "<style>" in doc  # CSS embedded (no external assets)
    assert "Business Insights" in doc
    assert "Smart Analyst" in doc  # brand + footer
    assert "trace back to your data" in doc  # grounding badge text


# ── Forecast report builder ─────────────────────────────────────────────────

def test_forecast_report_md_assembles_sections():
    results = {
        "forecast_outputs": [
            {
                "metric": "total_revenue",
                "current_value": 100000,
                "forecasted_value": 112000,
                "change_percent": 12.0,
                "forecast_horizon": "30_days",
                "selected_model": "Prophet",
                "model_selection_reason": "lowest out-of-sample MAPE",
                "evaluation": {"mae": 5.0, "rmse": 7.0, "mape": 6.0},
                "skill_score": 0.35,
                "confidence_score": 0.82,
                "business_impact": "High",
                "horizons": {"7d": 26000, "30d": 112000},
                "business_summary": "Revenue is set to grow.",
                "risks": ["Supply constraints"],
                "opportunities": ["Scale ad spend"],
                "recommended_actions": ["Increase inventory"],
            }
        ]
    }
    md = build_forecast_report_md(results)
    assert "# Sales & Demand Forecast" in md
    assert "## Total Revenue" in md
    assert "Prophet" in md
    assert "+12.0%" in md
    assert "Supply constraints" in md
    # The assembled report should be groundable against the same numbers.
    g = check_grounding(md, results["forecast_outputs"])
    assert g.verified_count == g.total and g.total > 0


def test_forecast_report_md_empty():
    assert build_forecast_report_md({"forecast_outputs": []}) == ""
