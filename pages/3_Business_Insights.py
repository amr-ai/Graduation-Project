"""
Business Insights page — senior BI report with HITL.
Before generation: user provides business context.
After generation: user can drill into specific sections via chat.

The analytics engine (KPI, timeseries, quality) runs automatically before
the LLM call so the report is grounded in computed metrics.
"""

from __future__ import annotations

import json
import os
import re

import streamlit as st
from groq import Groq

from agents.insights.agent import generate_insights
from agents.analytics.engine import run_analytics
from tools.sandbox import df_context, run_analysis


def _data_available() -> bool:
    return (
        st.session_state.clean_df is not None
        or (st.session_state.pg_stored and st.session_state.pg_table)
    )


def _get_data():
    if st.session_state.pg_stored and st.session_state.pg_table:
        return None, st.session_state.pg_table.lower()
    return st.session_state.clean_df, ""


def _render_kpi_card(kpi: dict):
    cols = st.columns(5)
    with cols[0]:
        st.metric("Revenue", f"${kpi.get('revenue', 0):,.0f}")
    with cols[1]:
        st.metric("Orders", kpi.get("orders", 0))
    with cols[2]:
        aov = kpi.get("aov", 0)
        st.metric("AOV", f"${aov:,.2f}" if aov else "—")
    with cols[3]:
        cust = kpi.get("total_customers", 0)
        st.metric("Customers", cust if cust else "—")
    with cols[4]:
        trend = kpi.get("growth_trend_direction", "—")
        st.metric("Trend", trend.capitalize() if trend else "—")


# ── Page content ───────────────────────────────────────────────────────

st.title("Business Insights")
st.markdown("Generate a consulting-level business report from your cleaned data.")

if not _data_available():
    st.info("No cleaned data available. Go to **Data Cleaning** and run the pipeline first.")
    st.stop()

data_df, table = _get_data()

# ── HITL Step 1: Business context form ─────────────────────────────────

with st.expander("Business context (optional but recommended)", expanded=True):
    st.markdown(
        "Help the analyst understand your business so the report is more relevant."
    )
    col1, col2 = st.columns(2)
    with col1:
        industry = st.selectbox(
            "Industry",
            [
                "",
                "Fashion / Apparel",
                "Electronics",
                "Grocery / FMCG",
                "Home & Garden",
                "Health & Beauty",
                "Sports & Outdoors",
                "Books & Media",
                "Automotive",
                "General retail",
                "B2B / Wholesale",
                "Other",
            ],
        )
    with col2:
        primary_goal = st.selectbox(
            "Primary goal",
            [
                "",
                "Revenue growth",
                "Profitability",
                "Customer retention",
                "Operational efficiency",
                "Product optimization",
                "Market expansion",
            ],
        )

    additional_context = st.text_area(
        "Additional context (target audience, season, promotions, etc.)",
        placeholder="E.g. 'This is Q4 holiday season data, we ran a 20% off campaign in November.'",
    )

    baseline_hint = st.text_input(
        "Baseline period (if known)",
        placeholder="E.g. 'Compare to previous quarter' or leave blank",
    )

    context_parts = []
    if industry:
        context_parts.append(f"Industry: {industry}")
    if primary_goal:
        context_parts.append(f"Primary goal: {primary_goal}")
    if additional_context:
        context_parts.append(f"Additional context: {additional_context}")
    if baseline_hint:
        context_parts.append(f"Baseline: {baseline_hint}")

    business_context = "\n".join(context_parts) if context_parts else ""

# ── HITL Step 2: Generate / regenerate ─────────────────────────────────

col_gen, _ = st.columns([1, 3])
with col_gen:
    generate_clicked = st.button(
        "Generate Insights Report",
        type="primary",
        use_container_width=True,
    )

if generate_clicked:
    with st.spinner("Running analytics engine …"):
        analytics_payload = run_analytics(data_df, table)
        st.session_state.analytics_payload = analytics_payload

    with st.spinner("Generating business report …"):
        report = generate_insights(
            data_df, table,
            business_context=business_context,
            analytics_payload=analytics_payload,
        )
        if report:
            st.session_state.insights_report = report
            st.rerun()

# ── KPI summary cards (from cached analytics) ──────────────────────────

analytics_payload = st.session_state.get("analytics_payload")
if analytics_payload:
    st.divider()
    st.subheader("Analytics Summary")
    kpi = analytics_payload.get("kpi", {})
    if kpi:
        _render_kpi_card(kpi)

    ts = analytics_payload.get("timeseries", {})
    if ts.get("available"):
        c1, c2 = st.columns(2)
        with c1:
            st.caption(f"Trend: **{ts.get('trend_direction', '—').capitalize()}** "
                       f"(slope {ts.get('trend_slope', 0):+.4f})")
            anomalies = ts.get("anomalies", {})
            st.caption(f"Anomalies detected: {anomalies.get('z_score_count', 0)} "
                       f"(z-score) / {anomalies.get('iqr_count', 0)} (IQR)")
        with c2:
            breaks = ts.get("structural_breaks", [])
            if breaks:
                st.caption(f"Structural breaks: {len(breaks)} detected")
            daily = ts.get("aggregations", {}).get("daily", {})
            if daily:
                st.caption(f"Date range: {daily.get('min_date', '—')} → {daily.get('max_date', '—')}")

    quality = analytics_payload.get("quality", [])
    if quality:
        missing = next((q for q in quality if q["check"] == "missing_values"), None)
        if missing and missing["total_missing"] > 0:
            st.caption(f"Quality: {missing['total_missing']} missing values "
                       f"({missing['overall_pct']}%), severity **{missing['severity']}**")

# ── Display report ─────────────────────────────────────────────────────

insights_report = st.session_state.get("insights_report", "")
if insights_report:
    st.divider()
    st.markdown(insights_report)

    col_down, _ = st.columns([1, 4])
    with col_down:
        st.download_button(
            "Download Report (.md)",
            data=insights_report,
            file_name="business_insights_report.md",
            mime="text/markdown",
            use_container_width=True,
        )

# ── HITL Step 3: Follow-up chat to drill into sections ─────────────────

if insights_report:
    st.divider()
    st.subheader("Ask a follow-up")
    st.caption("Drill into any section of the report or ask for more detail.")

    insights_messages = st.session_state.setdefault("insights_messages", [])

    for msg in insights_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if followup := st.chat_input("Ask about the report …"):
        insights_messages.append({"role": "user", "content": followup})
        with st.chat_message("user"):
            st.markdown(followup)

        with st.chat_message("assistant"):
            with st.spinner(""):
                try:
                    client = Groq(api_key=os.environ.get("GROQ_API_KEY") or "")
                    payload_str = (
                        f"\n\nPre-computed analytics:\n```json\n{json.dumps(analytics_payload, indent=2, default=str)}\n```"
                        if analytics_payload else ""
                    )
                    sys_prompt = (
                        "You are a senior BI analyst. The user just received a business report "
                        "and is asking follow-up questions about it.\n\n"
                        f"## Full Report\n\n{insights_report}\n"
                        f"{payload_str}\n\n"
                        "Answer concisely based on the report. If the user asks for additional "
                        "analysis, use Python code with `df` to compute it."
                    )
                    schema = df_context(data_df) if data_df is not None else ""
                    user_msg = followup
                    if schema:
                        user_msg += f"\n\nDataFrame context:\n{schema}"

                    reply = client.chat.completions.create(
                        model=st.session_state.get("model", "llama-3.3-70b-versatile"),
                        messages=[
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": user_msg},
                        ],
                        temperature=0.3,
                        max_tokens=2048,
                    )
                    answer = reply.choices[0].message.content.strip()
                except Exception as exc:
                    answer = f"Error: {exc}"

            code_blocks = re.findall(r"```python\n(.*?)```", answer, re.DOTALL)
            if code_blocks and data_df is not None:
                for code in code_blocks:
                    result = run_analysis(code, data_df)
                    if result["error"]:
                        answer += f"\n\n**Execution error:**\n```\n{result['error']}\n```"
                    else:
                        if result["output"]:
                            answer += f"\n\n**Output:**\n```\n{result['output']}\n```"
                        if "fig" in result and result["fig"] is not None:
                            st.plotly_chart(result["fig"], use_container_width=True, config={"displaylogo": False, "modeBarButtonsToRemove": ["sendDataToCloud"]})
                        if result["df"] is not None:
                            st.dataframe(result["df"])

            st.markdown(answer)
            insights_messages.append({"role": "assistant", "content": answer})
