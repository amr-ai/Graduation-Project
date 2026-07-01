"""
Tests for the forecasting upgrades: additive-aware aggregation, the new
professional models, back-testable seasonal windows, and the honest,
principled confidence score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from agents.forecasting.nodes import _confidence, _resolve_granularity
from agents.forecasting.tools.backtest import select_model
from agents.forecasting.tools.engines import aggregation_for, prepare_data
from agents.forecasting.tools.models import MODELS, get_model_fn, is_available


# ── Additive-aware aggregation ───────────────────────────────────────────────

def test_aggregation_for_additive_vs_rate():
    for col in ("revenue", "total_price", "quantity", "profit", "order_count"):
        assert aggregation_for(col) == "sum", col
    for col in ("unit_price", "discount_pct", "avg_order_value", "margin_ratio"):
        assert aggregation_for(col) == "mean", col


def test_prepare_data_sums_additive_metric():
    df = pd.DataFrame({
        "d": ["2024-01-01", "2024-01-01", "2024-01-02"],
        "total_price": [10.0, 20.0, 5.0],
    })
    out = prepare_data(df, "d", "total_price")
    # Two orders on Jan 1 must be SUMMED (30), not averaged (15).
    assert out.iloc[0]["y"] == 30.0
    assert out.iloc[1]["y"] == 5.0


def test_prepare_data_averages_rate_metric():
    df = pd.DataFrame({
        "d": ["2024-01-01", "2024-01-01"],
        "unit_price": [10.0, 20.0],
    })
    out = prepare_data(df, "d", "unit_price")
    assert out.iloc[0]["y"] == 15.0  # mean, not sum


def test_prepare_data_resamples_weekly():
    df = pd.DataFrame({
        "d": pd.date_range("2024-01-01", periods=14, freq="D"),
        "revenue": [1.0] * 14,
    })
    out = prepare_data(df, "d", "revenue", rule="W")
    # 14 daily 1.0s summed into weekly buckets -> ~7 per week.
    assert out["y"].sum() == 14.0
    assert len(out) <= 3


# ── New professional models ──────────────────────────────────────────────────

def test_new_models_registered():
    for name in ("Theta", "ETS", "ARIMA"):
        assert name in MODELS
        assert MODELS[name]["needs"] == "statsmodels"


def test_new_models_run_and_return_length():
    y = np.array([10.0 + i + (i % 7) for i in range(60)])
    dates = pd.date_range("2024-01-01", periods=60, freq="D").values
    for name in ("Theta", "ETS", "ARIMA"):
        if not is_available(name, len(y)):
            continue
        out = get_model_fn(name)(y, 5, "daily", dates)
        assert len(out) == 5
        assert np.all(np.isfinite(out))


def test_weekly_series_is_backtestable():
    # ~2 years of weekly points (season 52) used to fail to back-test because
    # min_train was 2*52 > available folds. It must now produce a leaderboard.
    rng = np.random.default_rng(0)
    y = 100 + 10 * np.sin(np.arange(105) * 2 * np.pi / 52) + rng.normal(0, 3, 105)
    dates = pd.date_range("2022-01-02", periods=105, freq="W").values
    board = select_model(y, dates, h=8, freq="weekly")
    assert board, "weekly series should be back-testable"
    assert board[0]["folds"] >= 1


# ── Honest confidence ────────────────────────────────────────────────────────

def test_confidence_none_when_not_backtested():
    assert _confidence(None, None, None) is None


def test_confidence_bounded_and_monotonic():
    good = _confidence(5.0, 0.6, 3)     # low error, beats naive
    poor = _confidence(90.0, 0.0, 3)    # high error, no skill
    assert 0.05 <= poor < good <= 0.95
    # A model that beats naive should score at least as high as one that doesn't.
    assert _confidence(30.0, 0.5, 3) >= _confidence(30.0, None, 3)


# ── Granularity resolution ───────────────────────────────────────────────────

def test_resolve_granularity():
    assert _resolve_granularity("weekly", "daily") == ("W", "weekly", 7)
    assert _resolve_granularity("monthly", "daily") == ("MS", "monthly", 30)
    # native falls back to the detected frequency, no resample rule.
    assert _resolve_granularity("native", "daily") == (None, "daily", 1)
