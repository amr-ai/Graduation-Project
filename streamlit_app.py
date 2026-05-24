"""
Smart Analyst — Streamlit UI with human-in-the-loop and post-cleaning chat.

Two-stage pipeline:
  1. Planning (profiler → planner) — paused for human review + chat
  2. Cleaning (coder → executor → validator) — runs with the approved plan
  3. Post-cleaning chat — ask questions / give commands about the clean data

Run with:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import contextlib
import io
import os
import re
import traceback

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
    "phase": "idle",
    "plan": "",
    "metadata": None,
    "log": "",
    "final_state": None,
    "chat_messages": [],
    "post_chat_messages": [],
    "clean_df": None,
}

for key, val in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── Sandbox helpers (reused from executor) ─────────────────────────────

_SAFE_BUILTINS: dict = {
    "range": range,
    "len": len,
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    "round": round,
    "min": min,
    "max": max,
    "abs": abs,
    "sum": sum,
    "sorted": sorted,
    "enumerate": enumerate,
    "zip": zip,
    "isinstance": isinstance,
    "hasattr": hasattr,
    "getattr": getattr,
    "type": type,
    "filter": filter,
    "map": map,
    "any": any,
    "all": all,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "Exception": Exception,
}


def _strip_fences(code: str) -> str:
    code = code.strip()
    code = re.sub(r"^```(?:python)?\s*\n?", "", code)
    code = re.sub(r"\n?```\s*$", "", code)
    return code.strip()


def _run_analysis(code: str, df: pd.DataFrame) -> dict:
    """Execute LLM-generated code on a DataFrame copy. Returns {output, error, df}."""
    code = _strip_fences(code)
    buffer = io.StringIO()

    def _sandbox_print(*a, **kw):
        kw["file"] = buffer
        print(*a, **kw)

    g = {
        "pd": pd,
        "df": df.copy(),
        "__builtins__": {**_SAFE_BUILTINS, "print": _sandbox_print},
    }
    try:
        exec(code, g)
        out = buffer.getvalue()
        new_df = g.get("df", df)
        return {"output": out, "error": None, "df": new_df}
    except Exception:
        return {"output": buffer.getvalue(), "error": traceback.format_exc(), "df": df}


def _df_context(df: pd.DataFrame) -> str:
    """Short textual summary of a DataFrame for LLM context."""
    buf = io.StringIO()
    buf.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")
    buf.write(f"Columns: {list(df.columns)}\n")
    buf.write(f"Dtypes:\n{df.dtypes.to_string()}\n")
    buf.write(f"\nFirst 5 rows:\n{df.head(5).to_string()}\n")
    if len(df) > 0:
        num_cols = df.select_dtypes(include="number").columns
        if len(num_cols) > 0:
            buf.write(f"\nNumeric summary:\n{df[num_cols].describe().to_string()}\n")
    return buf.getvalue()


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
        "planning": "Profiling and planning …",
        "review": "Review the plan below.",
        "cleaning": "Cleaning in progress …",
        "done": "Pipeline complete — chat with your data below.",
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


def _render_chat(container, messages_key, prompt_placeholder, system_prompt_fn, df=None):
    """Reusable chat widget. Renders messages, handles input, calls LLM (+ optionally executes code)."""
    messages = st.session_state[messages_key]

    for msg in messages:
        with container.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "df" in msg:
                st.dataframe(msg["df"])
            if "code" in msg:
                with st.expander("Generated code", expanded=False):
                    st.code(msg["code"], language="python")

    if prompt := container.chat_input(prompt_placeholder):
        messages.append({"role": "user", "content": prompt})
        with container.chat_message("user"):
            st.markdown(prompt)

        with container.chat_message("assistant"):
            with st.spinner(""):
                try:
                    client = Groq(api_key=os.environ.get("GROQ_API_KEY") or api_key)
                    sys_prompt = system_prompt_fn()

                    # Ask LLM to respond with Python code when appropriate
                    analysis_context = ""
                    if df is not None:
                        analysis_context = (
                            "IMPORTANT: You have access to a pandas DataFrame called `df`. "
                            "When the user asks a question that needs data analysis or transformation, "
                            "WRITE Python code using `df` and the code will be executed automatically. "
                            "Wrap code in ```python ... ``` fences. "
                            "Use print() to show text results, or assign back to `df` to keep changes. "
                            f"\n\nDataFrame context:\n{_df_context(df)}"
                        )

                    reply = client.chat.completions.create(
                        model=st.session_state.get("model", "llama-3.3-70b-versatile"),
                        messages=[
                            {"role": "system", "content": sys_prompt + "\n\n" + analysis_context if analysis_context else sys_prompt},
                            *[{"role": m["role"], "content": m["content"]} for m in messages[-6:]],
                        ],
                        temperature=0.2,
                        max_tokens=2048,
                    )
                    answer = reply.choices[0].message.content.strip()
                except Exception as exc:
                    answer = f"Error: {exc}"

            # Check if response contains executable code
            code_blocks = re.findall(r"```python\n(.*?)```", answer, re.DOTALL)
            if code_blocks and df is not None:
                for code in code_blocks:
                    result = _run_analysis(code, df)
                    if result["error"]:
                        answer += f"\n\n**Execution error:**\n```\n{result['error']}\n```"
                    else:
                        if result["output"]:
                            answer += f"\n\n**Output:**\n```\n{result['output']}\n```"
                        if result["df"] is not None and not result["df"].equals(df):
                            st.session_state.clean_df = result["df"]
                            df = result["df"]
                            st.markdown(answer)
                            st.dataframe(result["df"])
                            messages.append({
                                "role": "assistant",
                                "content": answer,
                                "df": result["df"],
                                "code": code,
                            })
                            return

            st.markdown(answer)
            msg_entry = {"role": "assistant", "content": answer}
            if code_blocks:
                msg_entry["code"] = code_blocks[0]
            messages.append(msg_entry)


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
        st.session_state.post_chat_messages = []
        st.session_state.clean_df = None
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

        def _review_sys_prompt():
            return (
                "You are a data cleaning expert helping a business user "
                "review a cleaning plan. The current plan is:\n\n"
                f"{st.session_state.plan}\n\n"
                "Answer the user's question about the plan concisely."
            )

        _render_chat(
            container=col_chat,
            messages_key="chat_messages",
            prompt_placeholder="Ask about the plan …",
            system_prompt_fn=_review_sys_prompt,
        )

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
    clean_df = result.get("execution_result", {}).get("clean_df")
    if clean_df is not None:
        st.session_state.clean_df = clean_df.copy()
    st.session_state.phase = "done"
    st.rerun()

# ── Results + Post-cleaning Chat ───────────────────────────────────────

if st.session_state.phase == "done" and st.session_state.final_state is not None:
    result = st.session_state.final_state
    report = result.get("validation_report", {})

    col_results, col_chat = st.columns([3, 2])

    with col_results:
        passed = report.get("passed", False)
        c1, c2, c3 = st.columns(3)
        c1.metric("Validation", "PASSED" if passed else "FAILED")
        c2.metric(
            "Rows",
            f"{report.get('rows_before', '?')} → {report.get('rows_after', '?')}",
        )
        c3.metric("Retries", result.get("retry_count", 0))

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

    with col_chat:
        st.subheader("Ask About Your Data")
        st.caption("Ask questions, request analysis, or transform the cleaned data.")

        def _post_sys_prompt():
            return (
                "You are a data analyst assistant. The user has a cleaned pandas DataFrame. "
                "Help them explore, analyze, and transform their data. "
                "When you need to perform analysis, write Python code using the `df` variable. "
                "The code will be executed and results shown to the user. "
                "Use print() for text output, or modify `df` in place to keep changes."
            )

        _render_chat(
            container=col_chat,
            messages_key="post_chat_messages",
            prompt_placeholder="Ask about your data …",
            system_prompt_fn=_post_sys_prompt,
            df=st.session_state.clean_df,
        )

# ── Empty state ────────────────────────────────────────────────────────

if st.session_state.phase == "idle" and uploaded_file is None:
    st.info("Upload a CSV file and enter your Groq API key in the sidebar to start.")
