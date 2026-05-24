"""
Smart Analyst — Streamlit UI for the Data Cleaning Pipeline.

Run with:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import io
import os
import sys

import pandas as pd
import streamlit as st

from core.state import GraphState
from graphs.cleaning_graph import build_cleaning_graph

st.set_page_config(page_title="Smart Analyst", page_icon="", layout="wide")

st.title("Smart Analyst")
st.markdown("LLM-powered data cleaning pipeline using Groq + LangGraph.")

# ── Sidebar ────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Configuration")

    api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Your Groq API key. Get one at console.groq.com",
    )

    model = st.selectbox(
        "Model",
        [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ],
        help="Groq model to use for planning and code generation",
    )

    uploaded_file = st.file_uploader("Upload CSV", type="csv")

    run_disabled = not (api_key and uploaded_file)
    run_btn = st.button("Run Pipeline", disabled=run_disabled, type="primary")

# ── Main area ──────────────────────────────────────────────────────────

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    st.info(f"Loaded **{len(raw_df)}** rows × **{len(raw_df.columns)}** columns")

if run_btn and uploaded_file is not None:
    os.environ["GROQ_API_KEY"] = api_key

    initial_state: GraphState = {
        "raw_df": raw_df,
        "file_path": uploaded_file.name,
        "model_name": model,
        "metadata": {},
        "cleaning_plan": "",
        "generated_code": "",
        "execution_result": {},
        "validation_report": {},
        "transformation_log": [],
        "retry_count": 0,
        "last_error": "",
        "messages": [],
    }

    # Capture print() output from graph nodes
    old_stdout = sys.stdout
    sys.stdout = captured = io.StringIO()

    try:
        with st.spinner("Running pipeline …"):
            graph = build_cleaning_graph()
            final_state = graph.invoke(initial_state)
    finally:
        sys.stdout = old_stdout

    log_output = captured.getvalue()

    # ── Results ─────────────────────────────────────────────────────

    report = final_state.get("validation_report", {})
    passed = report.get("passed", False)

    col1, col2, col3 = st.columns(3)
    col1.metric("Validation", "PASSED" if passed else "FAILED")
    col2.metric(
        "Rows",
        f"{report.get('rows_before', '?')} → {report.get('rows_after', '?')}",
    )
    col3.metric("Retries", final_state.get("retry_count", 0))

    with st.expander("Pipeline Log", expanded=True):
        st.text(log_output)

    st.subheader("Cleaning Plan")
    st.text(final_state.get("cleaning_plan", "No plan generated"))

    st.subheader("Validation Report")
    st.json(report, expanded=False)

    log = final_state.get("transformation_log", [])
    if log:
        st.subheader("Transformation Log")
        for i, entry in enumerate(log, 1):
            st.write(f"{i}. {entry}")

    clean_df = final_state.get("execution_result", {}).get("clean_df")
    if clean_df is not None:
        st.subheader("Cleaned Data Preview")
        st.dataframe(clean_df.head(20))

        csv_bytes = clean_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Cleaned CSV",
            data=csv_bytes,
            file_name="cleaned_data.csv",
            mime="text/csv",
        )
else:
    st.info("Upload a CSV file and enter your Groq API key in the sidebar to start.")
