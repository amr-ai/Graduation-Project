from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from agents.forecasting.forecast_models import ForecastState
from agents.forecasting.schema import ForecastEvaluation, ForecastOutput
from agents.forecasting.tools.engines import (
    ForecastEngine,
    SeriesData,
    prepare_data,
)
from agents.forecasting.validation import (
    _check_missing_dates,
    _detect_numeric_metrics,
    _determine_frequency,
    _detect_date_column,
    MIN_HISTORY,
)


def _business_impact(change_pct: float) -> str:
    a = abs(change_pct)
    if a > 15:
        return "high"
    if a > 5:
        return "medium"
    return "low"


def _confidence_from_metrics(metrics: dict) -> float:
    mape = metrics.get("mape")
    if mape is not None:
        return round(max(0.1, min(0.95, 1.0 - mape / 100)), 2)
    rmse = metrics.get("rmse", 0)
    if rmse:
        return round(max(0.1, min(0.95, 1.0 - rmse / 1000)), 2)
    return 0.7


def _build_chart(
    dates: np.ndarray,
    y: np.ndarray,
    forecast_dates: pd.DatetimeIndex,
    forecast: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    target: str,
    model_name: str = "Forecast",
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=y, mode="lines+markers", name="Historical",
        line=dict(color="royalblue", width=1.5), marker=dict(size=3),
    ))
    if len(y) >= 7:
        roll = pd.Series(y).rolling(7, min_periods=1).mean()
        fig.add_trace(go.Scatter(x=dates, y=roll, mode="lines", name="7-day avg", line=dict(width=2.5, color="lightblue")))

    fig.add_trace(go.Scatter(
        x=forecast_dates, y=forecast, mode="lines+markers", name="Forecast",
        line=dict(color="crimson", width=2.5, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=pd.concat([pd.Series(forecast_dates), pd.Series(forecast_dates[::-1])]),
        y=pd.concat([pd.Series(upper), pd.Series(lower[::-1])]),
        fill="toself", fillcolor="rgba(220,20,60,0.1)",
        line=dict(color="rgba(0,0,0,0)"), name="Confidence",
    ))
    fig.update_layout(
        title=f"{target.replace('_', ' ').title()} — {model_name} Forecast",
        template="plotly_white",
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title=target,
    )
    return fig


def _forecast_one_target(
    target: str,
    df: pd.DataFrame,
    date_col: str,
    freq: str,
    horizon: int,
    horizons: list[int],
    run_id: str,
    model: str = "Auto",
) -> dict[str, Any]:
    max_horizon = max(horizons)
    prep = prepare_data(df, date_col, target)
    y = prep["y"].values.astype(float)
    dates = prep["ds"].values

    series = SeriesData(y=y, dates=dates, frequency=freq)
    result = ForecastEngine.run(model, series, max_horizon)

    fv = result["forecast"]
    horizon_values: dict[str, float] = {}
    for h in horizons:
        if len(fv) >= h:
            horizon_values[f"{h}_days"] = float(np.mean(fv[:h]))

    current_val = float(y[-1]) if len(y) else 0.0
    forecast_val = horizon_values.get(f"{horizon}_days", float(fv[-1]) if len(fv) else current_val)
    change_pct = round((forecast_val - current_val) / abs(current_val) * 100, 2) if current_val else 0.0

    metrics = result.get("metrics", {})
    confidence = _confidence_from_metrics(metrics)
    trend = "growing" if change_pct > 5 else "declining" if change_pct < -5 else "stable"

    selected_model = result.get("selected_model", "Prophet")
    cv_results = result.get("cv_results", [])
    skill_score = result.get("skill_score")

    output = ForecastOutput(
        metric=target,
        current_value=round(current_val, 2),
        forecasted_value=round(forecast_val, 2),
        change_percent=change_pct,
        confidence_score=confidence,
        forecast_horizon=f"{horizon}_days",
        selected_model=selected_model,
        model_selection_reason=result.get("model_selection_reason", ""),
        evaluation=ForecastEvaluation(
            mae=metrics.get("mae"),
            rmse=metrics.get("rmse"),
            mape=metrics.get("mape"),
        ),
        business_impact=_business_impact(change_pct),
        trend=trend,
        horizons=horizon_values,
        cv_results=cv_results,
        skill_score=skill_score,
    )

    forecast_dates = result["dates"][:horizon]
    forecast_values = fv[:horizon]
    lower = result["lower"][:horizon]
    upper = result["upper"][:horizon]

    fig = _build_chart(dates, y, forecast_dates, forecast_values, lower, upper, target, selected_model)

    forecast_df = pd.DataFrame({
        "ds": forecast_dates,
        "yhat": forecast_values,
        "yhat_lower": lower,
        "yhat_upper": upper,
    })

    history_rows = [
        {"run_id": run_id, "metric_name": target, "ds": str(d), "y": float(v), "kind": "actual"}
        for d, v in zip(dates, y)
    ]
    for d, v, lo, hi in zip(forecast_df["ds"], forecast_df["yhat"], forecast_df["yhat_lower"], forecast_df["yhat_upper"]):
        history_rows.append({
            "run_id": run_id,
            "metric_name": target,
            "ds": str(d),
            "y": float(v),
            "yhat_lower": float(lo),
            "yhat_upper": float(hi),
            "kind": "forecast",
        })

    return {
        "output": output.to_agent_dict(),
        "fig": fig,
        "forecast_df": forecast_df,
        "metrics": {"metric": target, **metrics},
        "history_rows": history_rows,
    }


def validate_node(state: ForecastState) -> dict:
    df = state["df"]
    date_col = _detect_date_column(df)

    if date_col is None:
        return {
            "date_column": "",
            "frequency": "",
            "history_length": 0,
            "forecastable": False,
            "validation": {"frequency": "", "history_length": 0, "forecastable": False, "missing_dates_count": 0},
            "validation_message": "No date/time column detected.",
            "missing_dates_count": 0,
            "available_metrics": [],
        }

    dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
    if dates.empty:
        return {
            "date_column": date_col,
            "frequency": "",
            "history_length": 0,
            "forecastable": False,
            "validation": {"frequency": "", "history_length": 0, "forecastable": False, "missing_dates_count": 0},
            "validation_message": f"Date column '{date_col}' has no valid dates.",
            "missing_dates_count": 0,
            "available_metrics": [],
        }

    freq = _determine_frequency(dates)
    history = int(len(dates))
    missing = _check_missing_dates(dates, freq)
    metrics = _detect_numeric_metrics(df, date_col)
    min_req = MIN_HISTORY.get(freq, 30)
    forecastable = history >= min_req and len(metrics) > 0

    msg_parts = [
        f"Date column: **{date_col}**",
        f"Frequency: **{freq}**",
        f"Historical records: **{history}**",
        f"Forecastable: **{'yes' if forecastable else 'no'}** (minimum {min_req} periods)",
    ]
    if missing:
        msg_parts.append(f"Missing dates: **{len(missing)}** gaps")
    if metrics:
        msg_parts.append(f"Metrics: **{', '.join(metrics[:8])}**")

    return {
        "date_column": date_col,
        "frequency": freq,
        "history_length": history,
        "forecastable": forecastable,
        "validation": {
            "frequency": freq,
            "history_length": history,
            "forecastable": forecastable,
            "missing_dates_count": len(missing),
        },
        "validation_message": "\n\n".join(msg_parts),
        "missing_dates_count": len(missing),
        "available_metrics": metrics,
    }


def prepare_node(state: ForecastState) -> dict:
    return {}


def forecast_node(state: ForecastState) -> dict:
    targets = state.get("forecast_targets", [])
    if not targets:
        return {}

    df = state["df"]
    date_col = state["date_column"]
    freq = state.get("frequency", "daily")
    horizon = state.get("horizon_days", 30)
    horizons = state.get("standard_horizons") or [7, 30, 90]
    horizons = sorted({h for h in horizons if h > 0} | {horizon})
    run_id = state.get("run_id", "")
    model = state.get("model", "Auto")

    all_outputs: list[dict] = []
    all_figures: list[go.Figure] = []
    all_tables: list[pd.DataFrame] = []
    all_evaluations: list[dict] = []
    all_history: list[dict] = []
    model_selections: list[dict] = []

    with ThreadPoolExecutor(max_workers=min(len(targets), 4)) as pool:
        futures = {
            pool.submit(_forecast_one_target, t, df, date_col, freq, horizon, horizons, run_id, model): t
            for t in targets
        }
        for future in as_completed(futures):
            try:
                r = future.result()
                all_outputs.append(r["output"])
                all_figures.append(r["fig"])
                all_tables.append(r["forecast_df"])
                all_evaluations.append(r["metrics"])
                all_history.extend(r["history_rows"])
                model_selections.append({
                    "target": r["output"]["metric"],
                    "model": r["output"].get("selected_model", "Prophet"),
                    "reason": r["output"].get("model_selection_reason", ""),
                })
            except Exception as exc:
                t = futures[future]
                all_outputs.append({
                    "metric": t, "error": str(exc),
                    "current_value": 0, "forecasted_value": 0,
                    "change_percent": 0, "confidence_score": 0,
                })
                all_evaluations.append({"metric": t, "error": str(exc)})

    return {
        "forecast_outputs": all_outputs,
        "model_selections": model_selections,
        "execution_figures": all_figures,
        "execution_tables": all_tables,
        "forecast_evaluations": all_evaluations,
        "forecast_history_rows": all_history,
    }


__all__ = [
    "validate_node",
    "prepare_node",
    "forecast_node",
]
