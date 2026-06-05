"""
Smart Analyst — multipage Streamlit app.
Router + shared sidebar; each agent is a separate page with human-in-the-loop.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
import streamlit as st

load_dotenv()
st.set_page_config(page_title="Smart Analyst", page_icon="", layout="wide")

# ── Session state defaults ─────────────────────────────────────────────

_DEFAULTS = {
    "phase": "idle",
    "plan": "",
    "metadata": None,
    "log": "",
    "final_state": None,
    "chat_messages": [],
    "post_chat_messages": [],
    "clean_df": None,
    "viz_figures": [],
    "pg_stored": False,
    "pg_table": "",
    "pg_error": "",
    "insights_report": "",
    "marketing_report": "",
    "marketing_payload": None,
    "marketing_campaigns": None,
    "churn_payload": None,
    "churn_report": "",
}

for key, val in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── Shared sidebar (visible on every page) ─────────────────────────────

with st.sidebar:
    st.header("Configuration")

    api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Your Groq API key. Get one at console.groq.com",
        value=os.environ.get("GROQ_API_KEY", ""),
    )

    if api_key:
        os.environ["GROQ_API_KEY"] = api_key

    from agents.constants import (
        AVAILABLE_MODELS,
        DEFAULT_CHURN_MODEL,
        DEFAULT_CODER_MODEL,
        DEFAULT_DASHBOARD_MODEL,
        DEFAULT_FORECAST_MODEL,
        DEFAULT_INSIGHTS_MODEL,
        DEFAULT_MARKETING_MODEL,
        DEFAULT_PLANNER_MODEL,
        DEFAULT_VIZ_MODEL,
    )

    st.subheader("Agent Models")
    with st.expander("Per-Agent Settings", expanded=True):
        st.selectbox("Planner", AVAILABLE_MODELS,
                      index=AVAILABLE_MODELS.index(DEFAULT_PLANNER_MODEL),
                      key="planner_model",
                      help="Model for generating the cleaning plan")
        st.selectbox("Coder", AVAILABLE_MODELS,
                      index=AVAILABLE_MODELS.index(DEFAULT_CODER_MODEL),
                      key="coder_model",
                      help="Model for generating cleaning code")
        st.selectbox("Viz Coder", AVAILABLE_MODELS,
                      index=AVAILABLE_MODELS.index(DEFAULT_VIZ_MODEL),
                      key="viz_model",
                      help="Model for chat-driven chart generation")
        st.selectbox("Dashboard Generator", AVAILABLE_MODELS,
                      index=AVAILABLE_MODELS.index(DEFAULT_DASHBOARD_MODEL),
                      key="dashboard_model",
                      help="Model for auto-generating the chart dashboard")
        st.selectbox("Insights Agent", AVAILABLE_MODELS,
                      index=AVAILABLE_MODELS.index(DEFAULT_INSIGHTS_MODEL),
                      key="insights_model",
                      help="Model for generating the business insights report")
        st.selectbox("Forecast Agent", AVAILABLE_MODELS,
                      index=AVAILABLE_MODELS.index(DEFAULT_FORECAST_MODEL),
                      key="forecast_model",
                      help="Model for forecasting pipeline (planner, coder, analyzer, interpreter)")
        st.selectbox("Marketing Agent", AVAILABLE_MODELS,
                      index=AVAILABLE_MODELS.index(DEFAULT_MARKETING_MODEL),
                      key="marketing_model",
                      help="Model for marketing strategy, campaigns and ad copy")
        st.selectbox("Churn Agent", AVAILABLE_MODELS,
                      index=AVAILABLE_MODELS.index(DEFAULT_CHURN_MODEL),
                      key="churn_model",
                      help="Model for the churn retention-strategy narrative")

    st.divider()
    st.caption(f"Phase: **{st.session_state.phase}**")

    has_data = (
        st.session_state.clean_df is not None
        or (st.session_state.pg_stored and st.session_state.pg_table)
    )
    if not has_data and st.session_state.phase == "idle":
        st.info("Upload a CSV in **Data Cleaning** to get started.")

# ── Navigation ─────────────────────────────────────────────────────────

pages = [
    st.Page("pages/1_Data_Cleaning.py", title="Data Cleaning"),
    st.Page("pages/2_Visualization.py", title="Visualization"),
    st.Page("pages/3_Business_Insights.py", title="Business Insights"),
    st.Page("pages/4_Forecasting.py", title="Forecasting"),
    st.Page("pages/5_Marketing.py", title="Marketing"),
    st.Page("pages/6_Churn.py", title="Churn"),
]
pg = st.navigation(pages)
pg.run()
