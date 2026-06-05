"""
Rolling-origin (expanding-window) back-testing for model selection.

For each candidate model we re-fit on an expanding training window and score the
*out-of-sample* error on the held-out next ``h`` points, across several folds.
This is what makes the ``Auto`` selector honest — the chosen model is the one
that actually generalises, not the one that fits the training data best.
"""

from __future__ import annotations

import numpy as np

from agents.forecasting.tools.evaluation import evaluate_forecast
from agents.forecasting.tools.models import (
    MODELS,
    available_models,
    season_length,
)


def _splits(n: int, h: int, folds: int, min_train: int) -> list[tuple[int, int]]:
    """Expanding-window (test_start, test_end) index pairs, oldest fold first."""
    splits: list[tuple[int, int]] = []
    for k in range(folds, 0, -1):
        test_start = n - k * h
        if test_start < min_train:
            continue
        splits.append((test_start, min(test_start + h, n)))
    return splits


def _min_train(n: int, h: int, freq: str) -> int:
    return max(2 * season_length(freq, n), 8, h)


def _resolve_folds(n: int, h: int, freq: str, folds: int) -> tuple[int, int]:
    """Reduce fold count until at least one valid split exists."""
    min_train = _min_train(n, h, freq)
    while folds > 1 and (folds * h + min_train) > n:
        folds -= 1
    return folds, min_train


def backtest_model(model_fn, y, dates, h: int, freq: str, folds: int = 3) -> dict | None:
    """Average out-of-sample metrics for one model. ``None`` if it cannot run."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    folds, min_train = _resolve_folds(n, h, freq, folds)
    actuals: list[np.ndarray] = []
    preds: list[np.ndarray] = []

    for ts, te in _splits(n, h, folds, min_train):
        train_y = y[:ts]
        train_d = dates[:ts] if dates is not None else None
        try:
            p = np.asarray(model_fn(train_y, te - ts, freq, train_d), dtype=float)
        except Exception:
            return None
        if len(p) < (te - ts) or not np.all(np.isfinite(p[: te - ts])):
            return None
        actuals.append(y[ts:te])
        preds.append(p[: te - ts])

    if not actuals:
        return None

    a = np.concatenate(actuals)
    p = np.concatenate(preds)
    metrics = evaluate_forecast(a, p)
    metrics["residual_std"] = round(float(np.std(a - p)), 4)
    metrics["folds"] = len(actuals)
    return metrics


def select_model(
    y,
    dates,
    h: int,
    freq: str,
    candidates: list[str] | None = None,
    folds: int = 3,
) -> list[dict]:
    """Back-test candidates and return a leaderboard sorted by MAPE then RMSE.

    Each entry: ``{model, mae, rmse, mape, residual_std, folds}``.
    """
    n = len(np.asarray(y))
    if candidates is None:
        candidates = available_models(n, freq)

    leaderboard: list[dict] = []
    for name in candidates:
        spec = MODELS.get(name)
        if not spec:
            continue
        metrics = backtest_model(spec["fn"], y, dates, h, freq, folds)
        if metrics and metrics.get("mape") is not None:
            leaderboard.append({"model": name, **metrics})

    leaderboard.sort(
        key=lambda r: (
            r["mape"] if r["mape"] is not None else float("inf"),
            r["rmse"] if r["rmse"] is not None else float("inf"),
        )
    )
    return leaderboard


def skill_vs_naive(leaderboard: list[dict], selected: str) -> float | None:
    """Relative MAPE improvement of the selected model over the Naive baseline.

    ``0.42`` means 42% lower error than naive; negative means worse.
    """
    by_model = {r["model"]: r for r in leaderboard}
    sel = by_model.get(selected)
    naive = by_model.get("Naive")
    if not sel or not naive:
        return None
    sel_mape = sel.get("mape")
    naive_mape = naive.get("mape")
    if sel_mape is None or naive_mape is None or naive_mape == 0:
        return None
    return round(1.0 - sel_mape / naive_mape, 3)
