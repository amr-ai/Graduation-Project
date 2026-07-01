"""
Visualization Dashboard page.

Two layers:
  1. **Executive Dashboard** — a deterministic, board-quality dashboard built
     from the schema/KPI engine (always correct, consistent, instantly
     professional), with live slicers (date range + category) and one-click
     HTML export.
  2. **AI charts** — chat-driven, ad-hoc charts for anything the fixed
     dashboard doesn't cover, themed to match the house style.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from agents.analytics.schema_intel import build_schema_summary
from agents.constants import DEFAULT_VIZ_MODEL
from agents.visualization.builder import build_executive_dashboard
from agents.visualization.dashboard import generate_dashboard
from agents.visualization.export import build_dashboard_html
from agents.visualization.theme import style_figure
from agents.visualization.viz_graph import build_viz_graph

_PLOTLY_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["sendDataToCloud"]}


# ── Data resolution ─────────────────────────────────────────────────────────

def _data_available() -> bool:
    return (
        st.session_state.clean_df is not None
        or (st.session_state.pg_stored and st.session_state.pg_table)
    )


def _resolve_df() -> tuple[pd.DataFrame | None, str]:
    if st.session_state.clean_df is not None:
        return st.session_state.clean_df, "in-memory cleaned data"
    if st.session_state.pg_stored and st.session_state.pg_table:
        try:
            from tools.db_tools import load_df_from_pg
            tbl = st.session_state.pg_table.lower()
            return load_df_from_pg(tbl), f"PostgreSQL table {tbl}"
        except Exception as exc:
            st.error(f"PostgreSQL load failed: {exc}")
    return None, ""


# ── Rendering helpers ───────────────────────────────────────────────────────

def _render_kpis(kpis: list[dict]) -> None:
    if not kpis:
        return
    cols = st.columns(len(kpis))
    for col, c in zip(cols, kpis):
        direction = c.get("direction")
        color = {"up": "#1a7f47", "down": "#e5484d"}.get(direction, "#5b6472")
        arrow = {"up": "▲", "down": "▼"}.get(direction, "")
        with col:
            st.markdown(
                f"""<div style="background:#fff;border:1px solid #e4e8ee;border-radius:12px;
                    padding:14px 16px;height:100%;">
                  <div style="font-size:12px;color:#5b6472;text-transform:uppercase;
                       letter-spacing:.04em;">{c['label']}</div>
                  <div style="font-size:26px;font-weight:700;color:#1f2a3d;line-height:1.2;">
                       {c['value']}</div>
                  {f'<div style="font-size:12px;color:{color};margin-top:2px;">{arrow} {direction}</div>' if direction else ''}
                </div>""",
                unsafe_allow_html=True,
            )


def _render_chart(ch: dict) -> None:
    st.plotly_chart(ch["fig"], use_container_width=True, config=_PLOTLY_CONFIG)
    if ch.get("insight"):
        st.caption(ch["insight"])


def _render_grid(charts: list[dict]) -> None:
    buf: list[dict] = []

    def flush() -> None:
        if not buf:
            return
        for col, ch in zip(st.columns(len(buf)), buf):
            with col:
                _render_chart(ch)
        buf.clear()

    for ch in charts:
        if ch.get("width") == "full":
            flush()
            _render_chart(ch)
        else:
            buf.append(ch)
            if len(buf) == 2:
                flush()
    flush()


# ── Page ─────────────────────────────────────────────────────────────────────

st.title("Visualization Dashboard")
st.caption("A board-quality executive dashboard, built automatically from your data — plus AI charts for anything else.")

if not _data_available():
    st.info("No cleaned data available. Go to **Data Cleaning** and run the pipeline first.")
    st.stop()

df, source_label = _resolve_df()
if df is None:
    st.stop()

schema = build_schema_summary(df)

# ── Slicers ──────────────────────────────────────────────────────────────────

fdf = df
filter_bits: list[str] = []

with st.expander("Filters", expanded=False):
    fc1, fc2 = st.columns(2)

    time_col = schema.get("time_column")
    if time_col and time_col in df.columns:
        parsed = pd.to_datetime(df[time_col], errors="coerce")
        valid = parsed.dropna()
        if not valid.empty:
            dmin, dmax = valid.min().date(), valid.max().date()
            with fc1:
                date_range = st.date_input(
                    "Date range", value=(dmin, dmax), min_value=dmin, max_value=dmax,
                )
            if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
                lo, hi = date_range
                mask = (parsed.dt.date >= lo) & (parsed.dt.date <= hi)
                fdf = fdf[mask.reindex(fdf.index, fill_value=False)]
                if (lo, hi) != (dmin, dmax):
                    filter_bits.append(f"{lo}→{hi}")

    cats = schema.get("category_columns") or []
    cat_col = next((c for c in cats if df[c].nunique() <= 50), None)
    if cat_col:
        options = sorted(df[cat_col].dropna().astype(str).unique().tolist())
        with fc2:
            chosen = st.multiselect(
                f"{cat_col.replace('_', ' ').title()}", options, default=options,
            )
        if chosen and len(chosen) != len(options):
            fdf = fdf[df[cat_col].astype(str).isin(chosen).reindex(fdf.index, fill_value=False)]
            filter_bits.append(f"{cat_col}: {len(chosen)}/{len(options)}")

st.caption(
    f"Source: {source_label} · {len(fdf):,} of {len(df):,} rows"
    + (f" · filtered ({'; '.join(filter_bits)})" if filter_bits else "")
)

if fdf.empty:
    st.warning("No rows match the current filters. Widen the date range or category selection.")
    st.stop()

# ── Executive dashboard (auto-rebuilds when filters change) ──────────────────

sig = (source_label, tuple(filter_bits), len(fdf))
if st.session_state.get("exec_dash_sig") != sig or "exec_dashboard" not in st.session_state:
    with st.spinner("Building executive dashboard …"):
        st.session_state.exec_dashboard = build_executive_dashboard(fdf)
        st.session_state.exec_dash_sig = sig

dash = st.session_state.get("exec_dashboard", {})
kpis = dash.get("kpis", [])
charts = dash.get("charts", [])

st.subheader("Executive Dashboard")
_render_kpis(kpis)
st.write("")

if charts:
    _render_grid(charts)

    meta = {"Source": source_label, "Rows": f"{len(fdf):,}",
            "Revenue basis": dash.get("revenue_label", "")}
    if filter_bits:
        meta["Filters"] = "; ".join(filter_bits)
    html = build_dashboard_html(
        "Executive Dashboard", kpis, charts,
        subtitle="Automated business dashboard, grounded in your data", meta=meta,
    )
    st.download_button(
        "⬇ Download dashboard (HTML / PDF)",
        data=html.encode("utf-8"),
        file_name="executive_dashboard.html",
        mime="text/html",
        help="Self-contained interactive dashboard. Open it and Save as PDF to share.",
    )
else:
    st.info("Couldn't derive standard business charts from this data. Use the AI charts below.")

# ── AI charts (ad-hoc) ───────────────────────────────────────────────────────

st.divider()
st.subheader("AI Charts")
st.caption("Ask for any chart the dashboard above doesn't cover — it respects the filters and matches the house style.")

ai_cols = st.columns([1, 1, 2])
with ai_cols[0]:
    if st.button("Auto-generate charts", use_container_width=True):
        with st.spinner("Analyzing data and generating charts …"):
            new_charts = generate_dashboard(fdf, "", focus_context="")
            if new_charts:
                st.session_state.viz_figures = st.session_state.get("viz_figures", []) + new_charts
                st.rerun()
with ai_cols[1]:
    if st.session_state.get("viz_figures") and st.button("Clear AI charts", use_container_width=True):
        st.session_state.viz_figures = []
        st.rerun()

ai_figures = st.session_state.get("viz_figures", [])
if ai_figures:
    removed = None
    grid: list[dict] = []
    for idx, entry in enumerate(ai_figures):
        fig = entry.get("fig")
        if fig is not None:
            grid.append({"fig": fig, "insight": entry.get("query", ""), "width": "half"})
    _render_grid(grid)

    with st.expander("Manage AI charts / view code"):
        for idx, entry in enumerate(ai_figures):
            st.markdown(f"**{entry.get('query', 'Chart')}**")
            st.code(entry.get("code", ""), language="python")
            if st.button("Remove", key=f"rm_ai_{idx}"):
                removed = idx
    if removed is not None:
        st.session_state.viz_figures.pop(removed)
        st.rerun()

viz_query = st.chat_input("Ask for a chart (e.g. 'revenue by region as a treemap')")
if viz_query:
    figures = st.session_state.setdefault("viz_figures", [])
    with st.spinner("Generating chart …"):
        try:
            graph = build_viz_graph()
            state = {
                "user_query": viz_query,
                "schema_info": "",
                "table_name": "",
                "model": st.session_state.get("viz_model", DEFAULT_VIZ_MODEL),
                "existing_chart_count": len(figures),
                "generated_code": "",
                "execution_result": {},
                "retry_count": 0,
                "last_error": "",
                "df": fdf,
            }
            viz_state = graph.invoke(state)
            exec_result = viz_state.get("execution_result", {})
            figs = exec_result.get("figures") or ([exec_result["fig"]] if exec_result.get("fig") else [])
            if figs:
                for fig in figs:
                    style_figure(fig)
                    title = fig.layout.title.text if fig.layout.title else viz_query
                    figures.append({"query": title, "fig": fig,
                                    "code": viz_state.get("generated_code", "")})
                st.session_state.viz_figures = figures
                st.rerun()
            else:
                st.error(f"Chart generation failed: {exec_result.get('error', 'Unknown error')}")
        except Exception as exc:
            st.error(f"Error: {exc}")
