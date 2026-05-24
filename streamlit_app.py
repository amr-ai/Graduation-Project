"""
Smart Analyst — Streamlit UI with human-in-the-loop.

Two-stage pipeline:
  1. Planning (profiler → planner) — paused for human review + chat
  2. Cleaning (coder → executor → validator) — runs with the approved plan

Run with:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import contextlib
import io
import os

import pandas as pd
import streamlit as st
from groq import Groq

from core.state import GraphState
from graphs.planner_graph import build_planner_graph
from graphs.cleaning_graph import build_cleaning_graph

# ── Page config ────────────────────────────────────────────────────────

st.set_page_config(page_title="Smart Analyst", page_icon="", layout="wide")
st.title("Smart Analyst")
st.markdown("LLM-powered data cleaning pipeline with human review.")

# ── Initialise session state ───────────────────────────────────────────

_DEFAULTS = {
    "phase": "idle",           # idle | planning | review | cleaning | done
    "plan": "",
    "metadata": None,
    "log": "",
    "final_state": None,
    "chat_messages": [],
}

for key, val in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

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
        help="Groq model for planning and code generation",
    )

    uploaded_file = st.file_uploader("Upload CSV", type="csv")

    phase_help = {
        "idle": "",
        "planning": "Profiling and planning in progress …",
        "review": "Plan ready — review, edit, or chat below.",
        "cleaning": "Cleaning in progress …",
        "done": "Pipeline complete.",
    }
    hint = phase_help.get(st.session_state.phase, "")
    if hint:
        st.info(hint)

# ── Helpers ────────────────────────────────────────────────────────────


def _base_state(raw_df, fname, mdl) -> GraphState:
    return {
        "raw_df": raw_df,
        "file_path": fname,
        "model_name": mdl,
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


# ── Trigger planning ───────────────────────────────────────────────────

if st.session_state.phase == "idle" and uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    st.info(f"Loaded **{len(raw_df)}** rows × **{len(raw_df.columns)}** columns")

    if st.button("Run Pipeline", type="primary", disabled=not api_key):
        os.environ["GROQ_API_KEY"] = api_key
        st.session_state.phase = "planning"
        st.session_state.plan = ""
        st.session_state.metadata = None
        st.session_state.log = ""
        st.session_state.final_state = None
        st.session_state.chat_messages = []
        st.session_state.raw_df = raw_df
        st.session_state.file_name = uploaded_file.name
        st.session_state.model = model
        st.rerun()

# ── Stage 1: Planning ─────────────────────────────────────────────────

if st.session_state.phase == "planning":
    with st.spinner("Profiling and planning …"):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            graph = build_planner_graph()
            state = graph.invoke(
                _base_state(
                    st.session_state.raw_df,
                    st.session_state.file_name,
                    st.session_state.model,
                )
            )

    st.session_state.plan = state.get("cleaning_plan", "")
    st.session_state.metadata = state.get("metadata")
    st.session_state.raw_df = state.get("raw_df", st.session_state.raw_df)
    st.session_state.log = buffer.getvalue()
    st.session_state.phase = "review"
    st.rerun()

# ── Stage 2: Review ────────────────────────────────────────────────────

if st.session_state.phase == "review":
    col_plan, col_chat = st.columns([3, 2])

    with col_plan:
        st.subheader("Cleaning Plan")
        st.caption("Edit the plan below, then click **Approve** to continue.")

        edited_plan = st.text_area(
            "Plan",
            value=st.session_state.plan,
            height=400,
            label_visibility="collapsed",
        )

        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            if st.button("Approve Plan", type="primary", use_container_width=True):
                st.session_state.edited_plan = edited_plan
                st.session_state.phase = "cleaning"
                st.rerun()
        with col_btn2:
            if st.button("Cancel", use_container_width=True):
                for key in _DEFAULTS:
                    st.session_state[key] = _DEFAULTS[key]
                st.rerun()

    with col_chat:
        st.subheader("Chat with Agent")
        st.caption("Ask questions about the plan before approving.")

        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask about the plan …"):
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner(""):
                    try:
                        client = Groq(api_key=os.environ.get("GROQ_API_KEY") or api_key)
                        sys_prompt = (
                            "You are a data cleaning expert helping a business user "
                            "review a cleaning plan. The current plan is:\n\n"
                            f"{st.session_state.plan}\n\n"
                            "Answer the user's question about the plan concisely."
                        )
                        reply = client.chat.completions.create(
                            model=st.session_state.get("model", "llama-3.3-70b-versatile"),
                            messages=[
                                {"role": "system", "content": sys_prompt},
                                {"role": "user", "content": prompt},
                            ],
                            temperature=0.3,
                            max_tokens=512,
                        )
                        answer = reply.choices[0].message.content.strip()
                    except Exception as exc:
                        answer = f"Error: {exc}"
                st.markdown(answer)
                st.session_state.chat_messages.append({"role": "assistant", "content": answer})

# ── Stage 3: Cleaning ──────────────────────────────────────────────────

if st.session_state.phase == "cleaning":
    with st.spinner("Generating code and cleaning data …"):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            graph = build_cleaning_graph()
            state = _base_state(
                st.session_state.raw_df,
                st.session_state.file_name,
                st.session_state.model,
            )
            state["metadata"] = st.session_state.metadata
            state["cleaning_plan"] = st.session_state.get("edited_plan", st.session_state.plan)
            result = graph.invoke(state)

    st.session_state.final_state = result
    st.session_state.log += buffer.getvalue()
    st.session_state.phase = "done"
    st.rerun()

# ── Results ────────────────────────────────────────────────────────────

if st.session_state.phase == "done" and st.session_state.final_state is not None:
    result = st.session_state.final_state

    report = result.get("validation_report", {})
    passed = report.get("passed", False)

    col1, col2, col3 = st.columns(3)
    col1.metric("Validation", "PASSED" if passed else "FAILED")
    col2.metric(
        "Rows",
        f"{report.get('rows_before', '?')} → {report.get('rows_after', '?')}",
    )
    col3.metric("Retries", result.get("retry_count", 0))

    with st.expander("Pipeline Log", expanded=True):
        st.text(st.session_state.log or "No log captured.")

    st.subheader("Cleaning Plan")
    st.text(result.get("cleaning_plan", "No plan generated"))

    st.subheader("Validation Report")
    st.json(report, expanded=False)

    log = result.get("transformation_log", [])
    if log:
        st.subheader("Transformation Log")
        for i, entry in enumerate(log, 1):
            st.write(f"{i}. {entry}")

    clean_df = result.get("execution_result", {}).get("clean_df")
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

    if st.button("Start Over"):
        for key in _DEFAULTS:
            st.session_state[key] = _DEFAULTS[key]
        st.rerun()

# ── Empty state ────────────────────────────────────────────────────────

if st.session_state.phase == "idle" and uploaded_file is None:
    st.info("Upload a CSV file and enter your Groq API key in the sidebar to start.")
