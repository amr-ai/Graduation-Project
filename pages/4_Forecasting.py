"""
Forecasting Agent — production pipeline UI.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime

import pandas as pd
import streamlit as st
from groq import Groq

from agents.forecasting.metric_discovery import rank_metrics
from agents.constants import DEFAULT_FORECAST_MODEL
from agents.forecasting.pipeline import ForecastPipeline, DEFAULT_HORIZONS
from agents.forecasting.tools.engines import ForecastEngine
from agents.forecasting.validation import list_forecastable_metrics
from tools.sandbox import df_context, run_analysis


def _data_available() -> bool:
    return (
        st.session_state.clean_df is not None
        or (st.session_state.pg_stored and st.session_state.pg_table)
    )


def _resolve_dataframe() -> tuple[pd.DataFrame | None, str]:
    if st.session_state.clean_df is not None:
        return st.session_state.clean_df, ""
    if st.session_state.pg_stored and st.session_state.pg_table:
        try:
            from tools.db_tools import load_df_from_pg
            return load_df_from_pg(st.session_state.pg_table.lower()), st.session_state.pg_table.lower()
        except Exception as exc:
            st.error(f"PostgreSQL load failed: {exc}")
    return None, ""


def _clear_session() -> None:
    for key in ("forecast_results", "forecast_messages", "forecast_display_run_id"):
        st.session_state.pop(key, None)


st.title("Forecasting Agent")
st.caption(
    "Deterministic forecasting pipeline · LLM used only for business interpretation · "
    "Outputs stored for Strategy / Marketing agents"
)

if not _data_available():
    st.info("Upload and clean data in **Data Cleaning** first.")
    st.stop()

df, table = _resolve_dataframe()
if df is None:
    st.stop()

date_col, metrics = list_forecastable_metrics(df)
if not date_col or not metrics:
    st.error("Need a date column and numeric business metrics.")
    st.stop()

st.subheader("Configuration")

model_options = ["Auto"] + [m for m in ForecastEngine.SUPPORTED_MODELS if m != "Auto"]
model_override = st.selectbox(
    "Model selection",
    model_options,
    index=0,
    help=(
        "Auto back-tests every candidate (Prophet, Holt-Winters, Seasonal Naive, "
        "Drift, Linear Trend, …) with rolling-origin cross-validation and keeps the "
        "lowest out-of-sample MAPE. Pick a specific model to force it."
    ),
)

selected_metrics = st.multiselect(
    "Metrics to forecast",
    metrics,
    default=rank_metrics(metrics)[:3],
    max_selections=3,
)

c1, c2, c3 = st.columns(3)
with c1:
    horizon = st.selectbox("Primary horizon", [7, 30, 90], index=1)
with c2:
    extra_h = st.multiselect("Also compute", DEFAULT_HORIZONS, default=DEFAULT_HORIZONS)
with c3:
    business_context = st.text_area("Business context", height=80, placeholder="Optional…")

if st.button("Clear results"):
    _clear_session()
    st.rerun()

if st.button("Run Forecast", type="primary"):
    _clear_session()
    run_id = f"fc_{uuid.uuid4().hex[:12]}"
    st.session_state.forecast_display_run_id = run_id

    if not selected_metrics:
        st.error("Select at least one metric.")
        st.stop()

    with st.spinner("Running forecasting pipeline…"):
        try:
            pipeline = ForecastPipeline()
            result = pipeline.run(
                df,
                horizon_days=horizon,
                standard_horizons=sorted(set(extra_h) | {horizon}),
                metrics=selected_metrics,
                model_override=model_override,
                business_context=business_context,
                forecast_model=st.session_state.get("forecast_model", DEFAULT_FORECAST_MODEL),
                table_name=table,
                run_id=run_id,
            )
            result["completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.forecast_results = result
        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")
            st.stop()

display_id = st.session_state.get("forecast_display_run_id")
results = st.session_state.get("forecast_results")
if not results or results.get("run_id") != display_id:
    st.stop()

if not results.get("forecastable", True):
    st.warning(results.get("validation_message", "Data not forecastable."))
    st.stop()

st.caption(f"Run `{results.get('run_id')}` · {results.get('completed_at', '')}")

with st.expander("Validation", expanded=False):
    st.json(results.get("validation", {}))
    st.markdown(results.get("validation_message", ""))

outputs = results.get("forecast_outputs", [])
figures = results.get("execution_figures", [])

for i, out in enumerate(outputs):
    st.divider()
    st.subheader(out.get("metric", "Metric").replace("_", " ").title())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", out.get("selected_model", "—"))
    c2.metric("Forecast", f"{out.get('forecasted_value', 0):,.0f}")
    c3.metric("Change", f"{out.get('change_percent', 0):+.1f}%")
    c4.metric("Confidence", f"{out.get('confidence_score', 0):.0%}")

    ev = out.get("evaluation", {})
    skill = out.get("skill_score")
    skill_txt = f" · Skill vs naive: **{skill * 100:+.0f}%**" if skill is not None else ""
    st.caption(
        f"Back-test (out-of-sample) — MAE {ev.get('mae', '—')} · RMSE {ev.get('rmse', '—')} · "
        f"MAPE {ev.get('mape', '—')}%{skill_txt} · Impact: **{out.get('business_impact', '—')}**"
    )

    reason = out.get("model_selection_reason")
    if reason:
        st.caption(f"🧠 {reason}")

    cv_results = out.get("cv_results") or []
    if cv_results:
        with st.expander("Model back-test leaderboard"):
            lb = pd.DataFrame(cv_results)[["model", "mape", "rmse", "mae", "folds"]]
            lb = lb.rename(columns={
                "model": "Model", "mape": "MAPE %", "rmse": "RMSE",
                "mae": "MAE", "folds": "Folds",
            })
            st.dataframe(lb, use_container_width=True, hide_index=True)
            st.caption("Lower MAPE = better. Selection is by out-of-sample MAPE, then RMSE.")

    if i < len(figures) and figures[i] is not None:
        st.plotly_chart(figures[i], use_container_width=True)

    if out.get("horizons"):
        st.markdown("**Horizon forecasts (mean)**")
        st.json(out["horizons"])

    st.markdown(f"**Summary:** {out.get('business_summary', '')}")
    with st.expander("Full agent output (Strategy / Marketing)"):
        st.json(out)

storage = results.get("storage_result", {})
if storage.get("status") == "stored":
    st.success(f"Saved to PostgreSQL · `{storage.get('run_id', results.get('run_id'))}`")
