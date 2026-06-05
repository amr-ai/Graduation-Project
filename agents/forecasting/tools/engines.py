from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

from agents.forecasting.tools.backtest import select_model, skill_vs_naive
from agents.forecasting.tools.models import (
    MODELS,
    available_models,
    freq_to_pandas,
    get_model_fn,
    is_available,
    season_length,
)

logger = logging.getLogger(__name__)


def _infer_frequency(dates: pd.Series) -> str:
    deltas = pd.Series(dates.sort_values()).diff().dropna()
    if deltas.empty:
        return "D"
    median = deltas.median()
    try:
        days = median.total_seconds() / 86400
    except Exception:
        return "D"
    if days < 1.5:
        return "H" if days < 0.08 else "D"
    if days < 4:
        return "D"
    if days < 10:
        return "W"
    if days < 20:
        return "W"
    if days < 45:
        return "MS"
    return "MS"


def _replace_outliers_mad(y: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if len(y) < 7:
        return y
    med = np.median(y)
    mad = np.median(np.abs(y - med)) * 1.4826
    if mad == 0:
        return y
    z = np.abs((y - med) / mad)
    outlier_mask = z > threshold
    if not outlier_mask.any():
        return y
    roll = pd.Series(y).rolling(7, min_periods=1, center=True).median().values
    y = y.copy()
    y[outlier_mask] = roll[outlier_mask]
    return y


def prophet_params(df_p: pd.DataFrame) -> dict[str, Any]:
    y = df_p["y"].values
    ds = df_p["ds"]
    n = len(df_p)
    freq = _infer_frequency(ds)

    cv = float(y.std() / y.mean()) if float(y.mean()) != 0 else 0
    if cv > 0.3:
        params: dict[str, Any] = {"seasonality_mode": "multiplicative"}
        sps = 20.0
    else:
        params = {"seasonality_mode": "additive"}
        sps = 15.0

    cps = 0.3
    nc = min(25, max(10, n // 3))

    params["changepoint_prior_scale"] = cps
    params["changepoint_range"] = 0.9
    params["seasonality_prior_scale"] = sps
    params["uncertainty_samples"] = 300
    params["n_changepoints"] = nc
    params["interval_width"] = 0.8
    params["daily_seasonality"] = False

    if freq == "D":
        params["daily_seasonality"] = False
        params["weekly_seasonality"] = 12 if n >= 10 else False
        params["yearly_seasonality"] = 6 if n >= 180 else False
    elif freq == "W":
        params["weekly_seasonality"] = False
        params["daily_seasonality"] = False
        params["yearly_seasonality"] = 5 if n >= 52 else False
    elif freq == "MS":
        params["weekly_seasonality"] = False
        params["daily_seasonality"] = False
        params["yearly_seasonality"] = 5 if n >= 12 else False
    else:
        params["weekly_seasonality"] = n >= 10
        params["yearly_seasonality"] = n >= 180

    if n < 20:
        params["changepoint_prior_scale"] = 0.5
        params["n_changepoints"] = max(5, n // 2)

    return params


def prepare_data(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    work = df[[date_col, value_col]].dropna().copy()
    work["ds"] = pd.to_datetime(work[date_col])
    work["y"] = pd.to_numeric(work[value_col])
    work = work.sort_values("ds").reset_index(drop=True)
    work = work.groupby("ds", as_index=False)["y"].mean()
    return work[["ds", "y"]]


def run_prophet(
    y: np.ndarray,
    dates: np.ndarray,
    horizon: int,
    freq_str: str = "D",
) -> dict[str, Any]:
    df_p = pd.DataFrame({"ds": pd.to_datetime(dates), "y": y})
    df_p = df_p.groupby("ds", as_index=False)["y"].mean().sort_values("ds").reset_index(drop=True)

    y_safe = df_p["y"].values.astype(float)
    n = len(df_p)
    ds = df_p["ds"]

    if n >= 7:
        y_safe = _replace_outliers_mad(y_safe, threshold=3.0)
        df_p["y"] = y_safe

    params = prophet_params(df_p)
    safe_keys = {
        "growth", "changepoints", "n_changepoints", "changepoint_range",
        "changepoint_prior_scale", "seasonality_prior_scale",
        "yearly_seasonality", "weekly_seasonality", "daily_seasonality",
        "holidays", "seasonality_mode", "uncertainty_samples",
        "mcmc_samples", "interval_width",
    }
    safe = {k: v for k, v in params.items() if k in safe_keys}

    model = Prophet(**safe)

    span = (ds.max() - ds.min()).days
    freq = _infer_frequency(ds)

    if span >= 30:
        model.add_seasonality(name="monthly", period=30.5, fourier_order=8)
        model.add_seasonality(name="biweekly", period=14, fourier_order=5)

    if freq == "D" and span >= 90:
        model.add_seasonality(name="quarterly", period=91.25, fourier_order=4)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(df_p)

    future = model.make_future_dataframe(periods=horizon, freq=freq_str)
    pred = model.predict(future)
    tail = pred.tail(horizon)

    fv = tail["yhat"].values.astype(float)
    in_sample = model.predict(df_p[["ds"]])
    fitted = in_sample["yhat"].values.astype(float)
    n_train = len(df_p)

    lower = tail["yhat_lower"].values.astype(float)
    upper = tail["yhat_upper"].values.astype(float)

    y_actual = df_p["y"].values
    yhat_train = fitted[:n_train]
    rmse = float(np.sqrt(mean_squared_error(y_actual, yhat_train)))
    mae = float(mean_absolute_error(y_actual, yhat_train))
    mape = float(np.mean(np.abs((y_actual - yhat_train) / (y_actual + 1e-10))) * 100)

    last_date = pd.to_datetime(dates[-1])
    future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=freq_str)[1:]

    return {
        "forecast": fv,
        "fitted": fitted,
        "dates": future_dates,
        "lower": lower,
        "upper": upper,
        "method": "Prophet",
        "metrics": {"rmse": round(rmse, 4), "mae": round(mae, 4), "mape": round(mape, 2)},
    }


@dataclass
class SeriesData:
    y: np.ndarray
    dates: np.ndarray
    frequency: str = "daily"


def _freq_to_pandas(freq: str) -> str:
    return freq_to_pandas(freq)


def _empirical_intervals(
    forecast: np.ndarray, residual_std: float | None, fallback_y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """80% prediction band from back-test residual spread, widening with horizon."""
    sigma = residual_std
    if not sigma or sigma <= 0:
        diffs = np.diff(np.asarray(fallback_y, dtype=float))
        sigma = float(np.std(diffs)) if len(diffs) else float(np.std(fallback_y) or 1.0)
    z = 1.2816  # ~80% interval
    steps = np.sqrt(np.arange(1, len(forecast) + 1))
    band = z * sigma * steps
    return forecast - band, forecast + band


def _final_forecast(
    selected: str,
    y: np.ndarray,
    dates: np.ndarray,
    horizon: int,
    freq: str,
    residual_std: float | None,
) -> dict[str, Any]:
    """Refit the chosen model on the full history and produce the shipped forecast."""
    alias = freq_to_pandas(freq)

    if selected == "Prophet":
        return run_prophet(y, dates, horizon, alias)

    fn = get_model_fn(selected) or get_model_fn("Naive")
    forecast = np.asarray(fn(y, horizon, freq, dates), dtype=float)
    last_date = pd.to_datetime(dates[-1])
    future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=alias)[1:]
    lower, upper = _empirical_intervals(forecast, residual_std, y)

    return {
        "forecast": forecast,
        "fitted": np.asarray([], dtype=float),
        "dates": future_dates,
        "lower": lower,
        "upper": upper,
        "method": selected,
        "metrics": {"rmse": None, "mae": None, "mape": None},
    }


def _auto_reason(selected: str, leaderboard: list[dict], cv_h: int) -> str:
    entry = next((r for r in leaderboard if r["model"] == selected), None)
    if not entry:
        return f"Auto-selected {selected} (insufficient history to back-test alternatives)."
    parts = [
        f"Auto-selected {selected}: lowest back-test MAPE {entry['mape']}% "
        f"across {entry['folds']} folds (~{cv_h}-step horizon)."
    ]
    skill = skill_vs_naive(leaderboard, selected)
    if skill is not None and selected != "Naive":
        if skill > 0:
            parts.append(f"{skill * 100:.0f}% more accurate than the naive baseline.")
        else:
            parts.append("On par with the naive baseline for this series.")
    elif selected == "Naive":
        parts.append("No candidate reliably beat a random walk on this series.")
    return " ".join(parts)


class ForecastEngine:
    # "Auto" runs the back-test selector; the rest force a specific model.
    SUPPORTED_MODELS = ["Auto", *MODELS.keys()]

    @staticmethod
    def run(model: str, series: SeriesData, horizon: int) -> dict[str, Any]:
        y = np.asarray(series.y, dtype=float)
        dates = series.dates
        freq = series.frequency
        n = len(y)

        candidates = available_models(n, freq)
        # Cap the CV horizon so multiple folds fit within the available history.
        cv_h = max(1, min(horizon, max(season_length(freq, n), n // 4)))
        leaderboard = select_model(y, dates, cv_h, freq, candidates)

        if model and model != "Auto":
            if is_available(model, n):
                selected = model
                reason = f"User-selected {model} (forced)."
            else:
                selected = leaderboard[0]["model"] if leaderboard else "Naive"
                reason = (
                    f"{model} unavailable for this dataset "
                    f"(needs more history or a missing dependency); used {selected} instead."
                )
        else:
            selected = leaderboard[0]["model"] if leaderboard else "Naive"
            reason = _auto_reason(selected, leaderboard, cv_h)

        sel_entry = next((r for r in leaderboard if r["model"] == selected), None)
        residual_std = sel_entry.get("residual_std") if sel_entry else None

        result = _final_forecast(selected, y, dates, horizon, freq, residual_std)

        # Always report the honest, out-of-sample back-test metrics for the
        # chosen model (run_prophet returns in-sample metrics — override them).
        if sel_entry:
            result["metrics"] = {
                "mae": sel_entry.get("mae"),
                "rmse": sel_entry.get("rmse"),
                "mape": sel_entry.get("mape"),
            }
        result["selected_model"] = selected
        result["model_selection_reason"] = reason
        result["cv_results"] = leaderboard
        result["skill_score"] = skill_vs_naive(leaderboard, selected)
        return result
