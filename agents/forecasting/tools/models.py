"""
Forecasting model library.

Each model is a pure function with the signature:

    forecast(train_y, h, freq, train_dates=None) -> np.ndarray  (length h)

so that the back-tester ([backtest.py]) and the engine ([engines.py]) can treat
every model uniformly.  Heavy / optional dependencies (statsmodels, prophet) are
imported lazily inside their model so the module stays importable without them.

This replaces the previous "Prophet only" reality behind the `Auto` selector —
the engine now back-tests these candidates and picks the winner out-of-sample.
"""

from __future__ import annotations

import importlib.util
import warnings
from typing import Callable

import numpy as np
import pandas as pd

# Canonical pandas offset alias per detected frequency label.
_FREQ_ALIAS = {"daily": "D", "weekly": "W", "monthly": "ME"}


def freq_to_pandas(freq: str) -> str:
    return _FREQ_ALIAS.get(freq, "D")


def season_length(freq: str, n: int) -> int:
    """Seasonal period for a frequency, capped so it fits the history."""
    base = {"daily": 7, "weekly": 52, "monthly": 12}.get(freq, 7)
    if base >= n:
        base = max(1, n // 2)
    return base


def _has(module: str | None) -> bool:
    return module is None or importlib.util.find_spec(module) is not None


# ── Baseline / classical models ─────────────────────────────────────────────


def naive(train_y, h, freq, train_dates=None) -> np.ndarray:
    """Repeat the last observed value (random-walk baseline)."""
    last = float(train_y[-1]) if len(train_y) else 0.0
    return np.full(h, last, dtype=float)


def drift(train_y, h, freq, train_dates=None) -> np.ndarray:
    """Last value plus the average per-step change over the history."""
    n = len(train_y)
    if n < 2:
        return naive(train_y, h, freq)
    slope = (float(train_y[-1]) - float(train_y[0])) / (n - 1)
    return float(train_y[-1]) + slope * np.arange(1, h + 1)


def moving_average(train_y, h, freq, train_dates=None) -> np.ndarray:
    """Flat forecast at the mean of the most recent season."""
    n = len(train_y)
    k = min(season_length(freq, n), n)
    if k < 1:
        return naive(train_y, h, freq)
    return np.full(h, float(np.mean(train_y[-k:])), dtype=float)


def seasonal_naive(train_y, h, freq, train_dates=None) -> np.ndarray:
    """Repeat the last full season."""
    n = len(train_y)
    m = season_length(freq, n)
    if m < 2 or n < m:
        return naive(train_y, h, freq)
    last_season = np.asarray(train_y[-m:], dtype=float)
    reps = int(np.ceil(h / m))
    return np.tile(last_season, reps)[:h]


def linear_trend(train_y, h, freq, train_dates=None) -> np.ndarray:
    """Ordinary-least-squares trend line extrapolated forward."""
    n = len(train_y)
    if n < 2:
        return naive(train_y, h, freq)
    x = np.arange(n)
    coef = np.polyfit(x, np.asarray(train_y, dtype=float), 1)
    return np.polyval(coef, np.arange(n, n + h))


def holt_winters(train_y, h, freq, train_dates=None) -> np.ndarray:
    """Exponential smoothing (trend + optional seasonal) via statsmodels."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    y = np.asarray(train_y, dtype=float)
    n = len(y)
    m = season_length(freq, n)
    seasonal = "add" if (m >= 2 and n >= 2 * m) else None
    trend = "add" if n >= 4 else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ExponentialSmoothing(
            y,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=m if seasonal else None,
            initialization_method="estimated",
        ).fit()
        fc = np.asarray(fit.forecast(h), dtype=float)
    return fc


def theta(train_y, h, freq, train_dates=None) -> np.ndarray:
    """Theta method — the M3-competition winner: a robust decomposition of the
    series into long-run trend and short-run curvature. Excellent default."""
    from statsmodels.tsa.forecasting.theta import ThetaModel

    y = np.asarray(train_y, dtype=float)
    n = len(y)
    m = season_length(freq, n)
    seasonal = m if (m >= 2 and n >= 2 * m) else None
    s = pd.Series(y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ThetaModel(
            s, period=seasonal, deseasonalize=bool(seasonal), method="auto"
        ).fit()
        fc = np.asarray(res.forecast(h), dtype=float)
    return fc


def ets(train_y, h, freq, train_dates=None) -> np.ndarray:
    """Error-Trend-Seasonal state-space exponential smoothing (damped trend)."""
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel

    y = np.asarray(train_y, dtype=float)
    n = len(y)
    m = season_length(freq, n)
    seasonal = "add" if (m >= 2 and n >= 2 * m) else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ETSModel(
            y, error="add", trend="add", damped_trend=True,
            seasonal=seasonal, seasonal_periods=m if seasonal else None,
        ).fit(disp=False)
        fc = np.asarray(fit.forecast(h), dtype=float)
    return fc


def arima(train_y, h, freq, train_dates=None) -> np.ndarray:
    """ARIMA(1,1,1) — a classical, differenced auto-regressive baseline."""
    from statsmodels.tsa.arima.model import ARIMA

    y = np.asarray(train_y, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ARIMA(y, order=(1, 1, 1)).fit()
        fc = np.asarray(fit.forecast(h), dtype=float)
    return fc


def prophet_point(train_y, h, freq, train_dates=None) -> np.ndarray:
    """Point forecast from a lightweight Prophet fit (no uncertainty sampling).

    Kept fast on purpose: the full interval-producing fit lives in
    ``engines.run_prophet`` and is only used once, for the chosen model.
    """
    from prophet import Prophet

    alias = freq_to_pandas(freq)
    if train_dates is not None:
        ds = pd.to_datetime(train_dates)
    else:
        ds = pd.date_range(end=pd.Timestamp.today().normalize(), periods=len(train_y), freq=alias)
    dfp = pd.DataFrame({"ds": ds, "y": np.asarray(train_y, dtype=float)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = Prophet(
            uncertainty_samples=0,
            weekly_seasonality=(freq == "daily" and len(train_y) >= 14),
            yearly_seasonality=(freq != "daily" and len(train_y) >= 24),
            daily_seasonality=False,
        )
        model.fit(dfp)
    future = model.make_future_dataframe(periods=h, freq=alias)
    pred = model.predict(future)
    return pred["yhat"].to_numpy(dtype=float)[-h:]


# ── Registry ─────────────────────────────────────────────────────────────────

# name -> {fn, min_n (min history), needs (optional import module)}
MODELS: dict[str, dict] = {
    "Naive": {"fn": naive, "min_n": 2, "needs": None},
    "Seasonal Naive": {"fn": seasonal_naive, "min_n": 4, "needs": None},
    "Drift": {"fn": drift, "min_n": 3, "needs": None},
    "Moving Average": {"fn": moving_average, "min_n": 3, "needs": None},
    "Linear Trend": {"fn": linear_trend, "min_n": 4, "needs": None},
    "Holt-Winters": {"fn": holt_winters, "min_n": 8, "needs": "statsmodels"},
    "Theta": {"fn": theta, "min_n": 8, "needs": "statsmodels"},
    "ETS": {"fn": ets, "min_n": 10, "needs": "statsmodels"},
    "ARIMA": {"fn": arima, "min_n": 12, "needs": "statsmodels"},
    "Prophet": {"fn": prophet_point, "min_n": 24, "needs": "prophet"},
}

# Fast models only — used to cap back-test cost when many metrics are requested.
FAST_MODELS = ["Naive", "Seasonal Naive", "Drift", "Moving Average", "Linear Trend"]


def get_model_fn(name: str) -> Callable | None:
    spec = MODELS.get(name)
    return spec["fn"] if spec else None


def is_available(name: str, n: int) -> bool:
    spec = MODELS.get(name)
    if not spec:
        return False
    return n >= spec["min_n"] and _has(spec["needs"])


def available_models(n: int, freq: str) -> list[str]:
    """Candidate models usable for ``n`` observations with installed deps."""
    return [name for name in MODELS if is_available(name, n)] or ["Naive"]
