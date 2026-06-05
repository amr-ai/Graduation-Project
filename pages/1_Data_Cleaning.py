"""
Data Cleaning page — upload CSV, profile, review plan (HITL), execute cleaning,
validate results, and chat with cleaned data.
"""

from __future__ import annotations

import contextlib
import io
import os
import re

import pandas as pd
import streamlit as st
from groq import Groq

from agents.constants import DEFAULT_CODER_MODEL, DEFAULT_PLANNER_MODEL
from core.state import GraphState
from graphs.planner_graph import build_planner_graph
from graphs.cleaning_graph import build_cleaning_graph
from tools.db_tools import is_pg_configured, store_df_to_pg
from tools.sandbox import df_context, run_analysis

# ── Helpers ────────────────────────────────────────────────────────────


def _base_state(raw_df, fname, planner_mdl, coder_mdl) -> GraphState:
    return {
        "raw_df": raw_df,
        "file_path": fname,
        "planner_model": planner_mdl,
        "coder_model": coder_mdl,
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
    """Reusable chat widget — renders messages, handles input, calls LLM + executes code."""
    messages = st.session_state[messages_key]

    for msg in messages:
        with container.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "fig" in msg:
                st.plotly_chart(msg["fig"], use_container_width=True, config={"displaylogo": False, "modeBarButtonsToRemove": ["sendDataToCloud"]})
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
                    client = Groq(api_key=os.environ.get("GROQ_API_KEY") or "")
                    sys_prompt = system_prompt_fn()
                    analysis_context = ""
                    if df is not None:
                        analysis_context = (
                            "IMPORTANT: You have access to a pandas DataFrame called `df`. "
                            "When the user asks a question that needs data analysis or transformation, "
                            "WRITE Python code using `df` and the code will be executed automatically. "
                            "Wrap code in ```python ... ``` fences. "
                            "Use print() to show text results, or assign back to `df` to keep changes. "
                            f"\n\nDataFrame context:\n{df_context(df)}"
                        )

                    reply = client.chat.completions.create(
                        model=st.session_state.get("planner_model", DEFAULT_PLANNER_MODEL),
                        messages=[
                            {
                                "role": "system",
                                "content": sys_prompt + "\n\n" + analysis_context
                                if analysis_context
                                else sys_prompt,
                            },
                            *[{"role": m["role"], "content": m["content"]} for m in messages[-6:]],
                        ],
                        temperature=0.2,
                        max_tokens=2048,
                    )
                    answer = reply.choices[0].message.content.strip()
                except Exception as exc:
                    answer = f"Error: {exc}"

            code_blocks = re.findall(r"```python\n(.*?)```", answer, re.DOTALL)
            if code_blocks and df is not None:
                for code in code_blocks:
                    result = run_analysis(code, df)
                    if result["error"]:
                        answer += f"\n\n**Execution error:**\n```\n{result['error']}\n```"
                    else:
                        if result["output"]:
                            answer += f"\n\n**Output:**\n```\n{result['output']}\n```"

                        has_fig = "fig" in result and result["fig"] is not None
                        has_new_df = result["df"] is not None and not result["df"].equals(df)

                        if has_fig or has_new_df:
                            if has_new_df:
                                st.session_state.clean_df = result["df"]
                                df = result["df"]

                            st.markdown(answer)
                            if has_fig:
                                st.plotly_chart(result["fig"], use_container_width=True, config={"displaylogo": False, "modeBarButtonsToRemove": ["sendDataToCloud"]})
                            if has_new_df:
                                st.dataframe(result["df"])

                            msg_entry = {"role": "assistant", "content": answer}
                            if has_fig:
                                msg_entry["fig"] = result["fig"]
                            if has_new_df:
                                msg_entry["df"] = result["df"]
                            if code_blocks:
                                msg_entry["code"] = code_blocks[0]
                            messages.append(msg_entry)
                            return

            st.markdown(answer)
            msg_entry = {"role": "assistant", "content": answer}
            if code_blocks:
                msg_entry["code"] = code_blocks[0]
            messages.append(msg_entry)


# ── Page content ───────────────────────────────────────────────────────

st.title("Data Cleaning")
st.markdown("Upload a CSV, review and approve the cleaning plan, then inspect the results.")

# ── Upload ─────────────────────────────────────────────────────────────

if st.session_state.phase == "idle":
    uploaded_file = st.file_uploader("Upload CSV", type="csv")

    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        st.info(f"Loaded **{len(raw_df)}** rows × **{len(raw_df.columns)}** columns")

        has_key = bool(os.environ.get("GROQ_API_KEY"))
        if st.button("Run Pipeline", type="primary", disabled=not has_key):
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
            st.rerun()

# ── Planning ───────────────────────────────────────────────────────────

if st.session_state.phase == "planning":
    with st.spinner("Profiling and planning …"):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            graph = build_planner_graph()
            state = graph.invoke(
                _base_state(
                    st.session_state.raw_df,
                    st.session_state.file_name,
                    st.session_state.planner_model,
                    st.session_state.coder_model,
                )
            )

    st.session_state.plan = state.get("cleaning_plan", "")
    st.session_state.metadata = state.get("metadata")
    st.session_state.raw_df = state.get("raw_df", st.session_state.raw_df)
    st.session_state.log = buffer.getvalue()
    st.session_state.phase = "review"
    st.rerun()

# ── Review (HITL) ──────────────────────────────────────────────────────

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
                for key in ("phase", "plan", "metadata", "log", "final_state",
                            "chat_messages", "post_chat_messages", "clean_df",
                            "viz_figures", "pg_stored", "pg_table", "pg_error",
                            "insights_report"):
                    if key in st.session_state:
                        del st.session_state[key]
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

# ── Cleaning ───────────────────────────────────────────────────────────

if st.session_state.phase == "cleaning":
    with st.spinner("Generating code and cleaning data …"):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            graph = build_cleaning_graph()
            state = _base_state(
                st.session_state.raw_df,
                st.session_state.file_name,
                st.session_state.planner_model,
                st.session_state.coder_model,
            )
            state["metadata"] = st.session_state.metadata
            state["cleaning_plan"] = st.session_state.get("edited_plan", st.session_state.plan)
            result = graph.invoke(state)

    st.session_state.final_state = result
    st.session_state.log += buffer.getvalue()
    clean_df = result.get("execution_result", {}).get("clean_df")
    if clean_df is not None:
        st.session_state.clean_df = clean_df.copy()

    if is_pg_configured() and clean_df is not None:
        try:
            raw_table = (
                st.session_state.file_name.replace(".csv", "").replace(".", "_")
            )
            table_name = f"clean_{raw_table}".lower()
            store_df_to_pg(clean_df, table_name)
            st.session_state.pg_table = table_name
            st.session_state.pg_stored = True
        except Exception as e:
            st.session_state.pg_stored = False
            st.session_state.pg_error = str(e)

    st.session_state.phase = "done"
    st.rerun()

# ── Done — results + post-cleaning chat (HITL: inspect, re-ask, download) ──

if st.session_state.phase == "done" and st.session_state.final_state is not None:
    result = st.session_state.final_state
    report = result.get("validation_report", {})

    tab_results, tab_chat = st.tabs(["Results", "Chat"])

    with tab_results:
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
            for key in ("phase", "plan", "metadata", "log", "final_state",
                        "chat_messages", "post_chat_messages", "clean_df",
                        "viz_figures", "pg_stored", "pg_table", "pg_error",
                        "insights_report", "marketing_report", "marketing_payload",
                        "marketing_campaigns"):
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    with tab_chat:
        st.subheader("Ask About Your Data")
        st.caption("Ask questions, request analysis, or transform the cleaned data.")

        def _post_sys_prompt():
            return (
                "You are a data analyst assistant. The user has a cleaned pandas DataFrame. "
                "Help them explore, analyze, and transform their data. "
                "When you need to perform analysis, write Python code using the `df` variable. "
                "The code will be executed and results shown to the user. "
                "Use print() for text output, or modify `df` in place to keep changes.\n\n"
                "For visualizations, use plotly.express (imported as `px`) or plotly.graph_objects "
                "(imported as `go`). Create a figure variable named `fig` so it renders automatically.\n"
                "Examples:\n"
                "- fig = px.bar(df, x='category', y='total_spent', title='Sales by Category')\n"
                "- fig = px.line(df, x='order_date', y='total_spent', title='Trend')\n"
                "- fig = px.scatter(df, x='quantity', y='total_spent', color='category')\n"
                "- fig = px.pie(df, names='product', values='total_spent')\n"
                "- fig = px.histogram(df, x='total_spent', nbins=20)"
            )

        _render_chat(
            container=tab_chat,
            messages_key="post_chat_messages",
            prompt_placeholder="Ask about your data …",
            system_prompt_fn=_post_sys_prompt,
            df=st.session_state.clean_df,
        )
