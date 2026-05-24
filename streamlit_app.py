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

from dotenv import load_dotenv
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from groq import Groq

from core.state import GraphState
from graphs.planner_graph import build_planner_graph
from graphs.cleaning_graph import build_cleaning_graph
from tools.db_tools import get_schema_info, is_pg_configured, load_df_from_pg, store_df_to_pg
from agents.visualization.viz_graph import build_viz_graph

# ── Page config ────────────────────────────────────────────────────────

load_dotenv()

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
    "viz_figures": [],
    "pg_stored": False,
    "pg_table": "",
    "pg_error": "",
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
    "__import__": __import__,
}


def _strip_fences(code: str) -> str:
    code = code.strip()
    code = re.sub(r"^```(?:python)?\s*\n?", "", code)
    code = re.sub(r"\n?```\s*$", "", code)
    return code.strip()


def _run_analysis(code: str, df: pd.DataFrame) -> dict:
    """Execute LLM-generated code on a DataFrame copy. Returns {output, error, df, fig}."""
    code = _strip_fences(code)
    buffer = io.StringIO()

    def _sandbox_print(*a, **kw):
        kw["file"] = buffer
        print(*a, **kw)

    g = {
        "pd": pd,
        "px": px,
        "go": go,
        "df": df.copy(),
        "__builtins__": {**_SAFE_BUILTINS, "print": _sandbox_print},
    }
    try:
        exec(code, g)
        out = buffer.getvalue()
        new_df = g.get("df", df)
        fig = g.get("fig")
        result = {"output": out, "error": None, "df": new_df}
        if fig is not None:
            result["fig"] = fig
        return result
    except Exception:
        return {"output": buffer.getvalue(), "error": traceback.format_exc(), "df": df}


def _df_context(df: pd.DataFrame) -> str:
    """Short textual summary of a DataFrame for LLM context."""
    buf = io.StringIO()
    buf.write(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")
    buf.write(f"Columns: {list(df.columns)}\n")
    buf.write(f"Dtypes:\n{df.dtypes.to_string()}\n")
    buf.write(f"\nFirst 5 rows:\n{df.head(5).to_string()}\n")
    if len(df) > 0:
        num_cols = df.select_dtypes(include="number").columns
        if len(num_cols) > 0:
            buf.write(f"\nNumeric summary:\n{df[num_cols].describe().to_string()}\n")
    return buf.getvalue()


def _generate_dashboard(data_df: pd.DataFrame | None, table_name: str) -> list[dict]:
    """Auto-generate 5-7 diverse charts using the LLM, with retry on error."""
    if data_df is not None:
        schema = _df_context(data_df)
        src_df = data_df
    else:
        schema = get_schema_info(table_name)
        src_df = load_df_from_pg(table_name)

    client = Groq(api_key=os.environ.get("GROQ_API_KEY") or api_key)
    model = st.session_state.get("model", "llama-3.3-70b-versatile")

    base_prompt = (
        "You are a data visualization expert. Generate a comprehensive dashboard.\n\n"
        f"Data schema:\n{schema}\n\n"
        "Create 5 to 7 diverse charts that best explore and explain this data. "
        "Use a variety of chart types (bar, line, pie, scatter, histogram, box, area). "
        "Assign each chart to fig1, fig2, fig3, fig4, fig5, fig6, fig7. "
        "Every figure MUST have a descriptive title set via the title parameter.\n\n"
        "IMPORTANT:\n"
        "- Do NOT use import statements. "
        "pandas as pd, plotly.express as px, and plotly.graph_objects as go are already loaded.\n"
        "- The data is in a DataFrame called `df`.\n"
        "- Use EXACT column names as shown in the schema (case-sensitive, spaces matter).\n"
        "- For px.pie(), use names= for the category column and values= for the numeric column.\n"
        "- Wrap ALL code in a single ```python ... ``` block."
    )

    messages = [{"role": "system", "content": base_prompt}]

    for attempt in range(3):
        reply = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=4096,
        )
        raw = reply.choices[0].message.content.strip()

        code_match = re.search(r"```python\n(.*?)```", raw, re.DOTALL)
        code = code_match.group(1).strip() if code_match else raw

        buffer = io.StringIO()

        def _sandbox_print(*a, **kw):
            kw["file"] = buffer
            print(*a, **kw)

        g = {
            "pd": pd, "px": px, "go": go,
            "df": src_df.copy(),
            "__builtins__": {
                **_SAFE_BUILTINS, "print": _sandbox_print, "__import__": __import__,
            },
        }

        try:
            exec(code, g)
        except Exception:
            tb = traceback.format_exc()
            if attempt < 2:
                messages.append({
                    "role": "assistant",
                    "content": raw,
                })
                messages.append({
                    "role": "user",
                    "content": (
                        "The code above failed with this error:\n"
                        f"```\n{tb}\n```\n"
                        "Fix the issue and regenerate ALL charts. "
                        "Ensure column names match exactly and all imports are omitted."
                    ),
                })
                continue
            else:
                st.error(f"Dashboard generation failed after 3 attempts:\n```\n{tb}\n```")
                return []

        # Collect all fig1..figN + fallback fig
        charts = []
        i = 1
        while g.get(f"fig{i}") is not None:
            fig = g[f"fig{i}"]
            title = fig.layout.title.text if fig.layout.title else f"Chart {i}"
            charts.append({"query": title, "fig": fig, "code": code})
            i += 1

        if not charts and g.get("fig"):
            fig = g["fig"]
            title = fig.layout.title.text if fig.layout.title else "Chart"
            charts.append({"query": title, "fig": fig, "code": code})

        return charts

    return []


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
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "qwen/qwen3-32b",
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
            if "fig" in msg:
                st.plotly_chart(msg["fig"], use_container_width=True)
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

                        has_fig = "fig" in result and result["fig"] is not None
                        has_new_df = result["df"] is not None and not result["df"].equals(df)

                        if has_fig or has_new_df:
                            if has_new_df:
                                st.session_state.clean_df = result["df"]
                                df = result["df"]

                            st.markdown(answer)
                            if has_fig:
                                st.plotly_chart(result["fig"], use_container_width=True)
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

    # Store to PostgreSQL if configured
    if is_pg_configured() and clean_df is not None:
        try:
            raw_table = (
                st.session_state.file_name.replace(".csv", "").replace(".", "_")
            )
            table_name = f"clean_{raw_table}"
            store_df_to_pg(clean_df, table_name)
            st.session_state.pg_table = table_name
            st.session_state.pg_stored = True
        except Exception as e:
            st.session_state.pg_stored = False
            st.session_state.pg_error = str(e)

    st.session_state.phase = "done"
    st.rerun()

# ── Results + Post-cleaning Chat ──────────────────────────────────────

if st.session_state.phase == "done" and st.session_state.final_state is not None:
    result = st.session_state.final_state
    report = result.get("validation_report", {})

    tab_results, tab_chat = st.tabs(["Results", "Chat"])

    # ── Results tab ──────────────────────────────────────────────────
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
            for key in _DEFAULTS:
                st.session_state[key] = _DEFAULTS[key]
            st.rerun()

    # ── Chat tab ─────────────────────────────────────────────────────
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

# ── Persistent visualization dashboard ─────────────────────────────────
# Always visible whenever data is available (clean_df or PG).

_dash_has_data = (
    st.session_state.clean_df is not None
    or (st.session_state.pg_stored and st.session_state.pg_table)
)

if _dash_has_data:
    st.divider()
    st.subheader("Visualization Dashboard")

    use_pg = st.session_state.pg_stored and st.session_state.pg_table
    table = st.session_state.pg_table if use_pg else ""
    data_df = None if use_pg else st.session_state.clean_df

    if use_pg:
        st.caption(f"Reading from PostgreSQL table **{table}**.")
    else:
        st.caption("Using in-memory cleaned data. Run pipeline with PG configured to persist.")

    # Show existing charts in a 2-column grid
    figures = st.session_state.get("viz_figures", [])

    col_auto, col_clear = st.columns([3, 1])
    with col_auto:
        if st.button("Auto-Generate Dashboard", type="primary", use_container_width=True):
            with st.spinner("Analyzing data and generating 5-7 charts …"):
                new_charts = _generate_dashboard(data_df, table)
                if new_charts:
                    st.session_state.viz_figures = figures + new_charts
                    st.rerun()
    with col_clear:
        if figures and st.button("Clear All Charts", use_container_width=True):
            st.session_state.viz_figures = []
            st.rerun()

    if figures:
        for i in range(0, len(figures), 2):
            cols = st.columns(2)
            for j in range(2):
                idx = i + j
                if idx < len(figures):
                    entry = figures[idx]
                    with cols[j]:
                        st.caption(f"_{entry['query']}_")
                        st.plotly_chart(entry["fig"], use_container_width=True)
                        with st.expander("Code", expanded=False):
                            st.code(entry["code"], language="python")

    viz_query = st.chat_input(
        "Ask for a chart (e.g. 'bar chart of sales by category')"
    )

    if viz_query:
        figures = st.session_state.setdefault("viz_figures", [])
        with st.spinner("Generating chart …"):
            try:
                graph = build_viz_graph()

                # Build the state dict — pass df directly for in-memory mode
                state = {
                    "user_query": viz_query,
                    "schema_info": "",
                    "table_name": table,
                    "existing_chart_count": len(figures),
                    "generated_code": "",
                    "execution_result": {},
                    "retry_count": 0,
                    "last_error": "",
                }
                if data_df is not None:
                    state["df"] = data_df

                viz_state = graph.invoke(state)
                exec_result = viz_state.get("execution_result", {})
                figures_list = exec_result.get("figures") or []
                single_fig = exec_result.get("fig")

                if figures_list:
                    for fig in figures_list:
                        title = fig.layout.title.text if fig.layout.title else viz_query
                        figures.append({
                            "query": title,
                            "fig": fig,
                            "code": viz_state.get("generated_code", ""),
                        })
                    st.session_state.viz_figures = figures
                    st.rerun()
                elif single_fig:
                    figures.append({
                        "query": viz_query,
                        "fig": single_fig,
                        "code": viz_state.get("generated_code", ""),
                    })
                    st.session_state.viz_figures = figures
                    st.rerun()
                else:
                    st.error(
                        f"Chart generation failed: {exec_result.get('error', 'Unknown error')}"
                    )
            except Exception as e:
                st.error(f"Error: {e}")

# ── Empty state ────────────────────────────────────────────────────────

if st.session_state.phase == "idle" and uploaded_file is None:
    st.info("Upload a CSV file and enter your Groq API key in the sidebar to start.")
