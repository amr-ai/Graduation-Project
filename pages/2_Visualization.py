"""
Visualization page — auto-generated and chat-driven charts with HITL.
Before generation: user selects focus areas.
After generation: user can remove individual charts or ask for custom ones.
"""

from __future__ import annotations

import streamlit as st

from agents.visualization.viz_graph import build_viz_graph
from agents.visualization.dashboard import generate_dashboard

FOCUS_OPTIONS = [
    "Revenue & Profit",
    "Orders & Volume",
    "Customer segments",
    "Product performance",
    "Time trends",
    "Category breakdown",
    "Geographic distribution",
]

CHART_TYPE_OPTIONS = [
    "Bar",
    "Line",
    "Pie / Donut",
    "Scatter",
    "Histogram",
    "Box plot",
    "Heatmap",
]


def _data_available() -> bool:
    return (
        st.session_state.clean_df is not None
        or (st.session_state.pg_stored and st.session_state.pg_table)
    )


def _data_source_info():
    if st.session_state.pg_stored and st.session_state.pg_table:
        return f"PostgreSQL table **{st.session_state.pg_table}**"
    return "in-memory cleaned data"


def _get_data():
    if st.session_state.pg_stored and st.session_state.pg_table:
        return None, st.session_state.pg_table.lower()
    return st.session_state.clean_df, ""


# ── Page content ───────────────────────────────────────────────────────

st.title("Visualization Dashboard")
st.markdown("Create charts to explore your data.")

if not _data_available():
    st.info("No cleaned data available. Go to **Data Cleaning** and run the pipeline first.")
    st.stop()

data_df, table = _get_data()
use_pg = data_df is None

st.caption(f"Source: {_data_source_info()}")

# ── HITL: Focus & chart-type selection before generation ───────────────

with st.expander("Chart preferences (optional)", expanded=False):
    st.markdown("Tell the agent what you care about so it generates more relevant charts.")
    col_focus, col_types = st.columns(2)

    with col_focus:
        selected_focus = st.multiselect(
            "Focus areas",
            FOCUS_OPTIONS,
            default=["Revenue & Profit", "Time trends"],
        )
    with col_types:
        selected_types = st.multiselect(
            "Chart types",
            CHART_TYPE_OPTIONS,
            default=["Bar", "Line", "Pie / Donut"],
        )

    focus_context = ""
    if selected_focus or selected_types:
        parts = []
        if selected_focus:
            parts.append(f"Focus on: {', '.join(selected_focus)}")
        if selected_types:
            parts.append(f"Preferred chart types: {', '.join(selected_types)}")
        focus_context = "; ".join(parts)

# ── Generate / Clear buttons ───────────────────────────────────────────

figures = st.session_state.get("viz_figures", [])

col_auto, col_clear = st.columns([3, 1])
with col_auto:
    if st.button("Auto-Generate Dashboard", type="primary", use_container_width=True):
        with st.spinner("Analyzing data and generating 5-7 charts …"):
            new_charts = generate_dashboard(data_df, table, focus_context=focus_context)
            if new_charts:
                st.session_state.viz_figures = figures + new_charts
                st.rerun()
with col_clear:
    if figures and st.button("Clear All Charts", use_container_width=True):
        st.session_state.viz_figures = []
        st.rerun()

# ── Chart grid with per-chart Remove button (HITL) ─────────────────────

if figures:
    for i in range(0, len(figures), 2):
        cols = st.columns(2)
        for j in range(2):
            idx = i + j
            if idx < len(figures):
                entry = figures[idx]
                with cols[j]:
                    st.caption(f"_{entry['query']}_")
                    try:
                        st.plotly_chart(
                            entry["fig"],
                            use_container_width=True,
                            config={"displaylogo": False, "modeBarButtonsToRemove": ["sendDataToCloud"]},
                        )
                    except Exception:
                        st.error(f"Could not render this chart (non-serializable data).")
                        if st.button(f"Remove broken chart", key=f"rm_broken_{idx}"):
                            st.session_state.viz_figures.pop(idx)
                            st.rerun()
                    col_code, col_rm = st.columns([3, 1])
                    with col_code:
                        with st.expander("Code", expanded=False):
                            st.code(entry["code"], language="python")
                    with col_rm:
                        if st.button(f"Remove", key=f"rm_chart_{idx}", use_container_width=True):
                            st.session_state.viz_figures.pop(idx)
                            st.rerun()

    st.divider()

# ── Chat input for custom charts (HITL: conversational chart creation) ─

viz_query = st.chat_input("Ask for a chart (e.g. 'bar chart of sales by category')")

if viz_query:
    figures = st.session_state.setdefault("viz_figures", [])
    with st.spinner("Generating chart …"):
        try:
            graph = build_viz_graph()
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
