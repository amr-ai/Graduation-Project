"""
Visualization page — auto-generated and chat-driven charts with HITL.
Charts are displayed in tabs for easy navigation.
"""

from __future__ import annotations

import streamlit as st

from agents.constants import DEFAULT_VIZ_MODEL
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


st.title("Visualization Dashboard")
st.markdown("Create charts to explore your data.")

if not _data_available():
    st.info("No cleaned data available. Go to **Data Cleaning** and run the pipeline first.")
    st.stop()

data_df, table = _get_data()

st.caption(f"Source: {_data_source_info()}")

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

# ── Buttons ──────────────────────────────────────────────────────────────

col_new, col_add, col_clear = st.columns([1, 1, 1])
with col_new:
    if st.button("New Dashboard", type="primary", use_container_width=True):
        with st.spinner("Analyzing data and generating 5-7 charts …"):
            new_charts = generate_dashboard(data_df, table, focus_context=focus_context)
            if new_charts:
                st.session_state.viz_figures = new_charts
                st.rerun()
with col_add:
    if st.button("Add More Charts", use_container_width=True):
        with st.spinner("Generating additional charts …"):
            existing = st.session_state.get("viz_figures", [])
            new_charts = generate_dashboard(
                data_df, table,
                focus_context=f"{focus_context}; Avoid repeating existing chart types" if focus_context else "Avoid repeating existing chart types",
            )
            if new_charts:
                st.session_state.viz_figures = existing + new_charts
                st.rerun()
with col_clear:
    if st.session_state.get("viz_figures") and st.button("Clear All", use_container_width=True):
        st.session_state.viz_figures = []
        st.rerun()

# ── Display charts in tabs ──────────────────────────────────────────────

figures = st.session_state.get("viz_figures", [])
if figures:
    tab_labels = []
    for entry in figures:
        raw = entry.get("query", "Chart")
        label = raw[:20] if len(raw) > 20 else raw
        tab_labels.append(label)

    tabs = st.tabs(tab_labels)
    removed = None
    for idx, tab in enumerate(tabs):
        entry = figures[idx]
        with tab:
            st.caption(f"_{entry['query']}_")
            fig = entry.get("fig")
            if fig is not None:
                try:
                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        config={"displaylogo": False, "modeBarButtonsToRemove": ["sendDataToCloud"]},
                    )
                except Exception:
                    st.error("Could not render this chart.")
            with st.expander("Code", expanded=False):
                st.code(entry.get("code", ""), language="python")
            if st.button("Remove", key=f"rm_{idx}", use_container_width=True):
                removed = idx

    if removed is not None:
        st.session_state.viz_figures.pop(removed)
        st.rerun()

    st.divider()

# ── Chat input for custom charts ────────────────────────────────────────

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
                "model": st.session_state.get("viz_model", DEFAULT_VIZ_MODEL),
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

            added = False
            if figures_list:
                for fig in figures_list:
                    title = fig.layout.title.text if fig.layout.title else viz_query
                    figures.append({
                        "query": title,
                        "fig": fig,
                        "code": viz_state.get("generated_code", ""),
                    })
                added = True
            elif single_fig:
                figures.append({
                    "query": viz_query,
                    "fig": single_fig,
                    "code": viz_state.get("generated_code", ""),
                })
                added = True

            if added:
                st.session_state.viz_figures = figures
                st.rerun()
            else:
                st.error(
                    f"Chart generation failed: {exec_result.get('error', 'Unknown error')}"
                )
        except Exception as e:
            st.error(f"Error: {e}")
