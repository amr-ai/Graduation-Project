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

    model = st.selectbox(
        "Model",
        [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "qwen/qwen3-32b",
        ],
        help="Groq model for planning and code generation",
    )
    st.session_state.model = model

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
    st.Page("pages/4_Marketing_Agent.py", title="Marketing Agent"),
]
pg = st.navigation(pages)
pg.run()
