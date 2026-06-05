"""
Forecast evaluation metrics — MAE, RMSE, MAPE (deterministic).
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def evaluate_forecast(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Optional[float]]:
    """Compute error metrics between aligned actual and predicted arrays."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    n = min(len(actual), len(predicted))
    if n == 0:
        return {"mae": None, "rmse": None, "mape": None}

    actual, predicted = actual[:n], predicted[:n]
    err = actual - predicted
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    mask = np.abs(actual) > 1e-9
    if mask.any():
        mape = float(np.mean(np.abs(err[mask] / actual[mask])) * 100)
    else:
        mape = None

    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "mape": round(mape, 2) if mape is not None else None}


def evaluate_in_sample(
    y: np.ndarray,
    fitted: np.ndarray | None,
    holdout: int = 0,
) -> dict[str, Optional[float]]:
    """Evaluate in-sample fit; optional tail holdout for pseudo-out-of-sample MAE."""
    y = np.asarray(y, dtype=float)
    if fitted is None or len(fitted) == 0:
        return {"mae": None, "rmse": None, "mape": None}

    fitted = np.asarray(fitted, dtype=float)
    n = min(len(y), len(fitted))
    y, fitted = y[-n:], fitted[-n:]

    if holdout > 0 and n > holdout + 5:
        return evaluate_forecast(y[-holdout:], fitted[-holdout:])

    return evaluate_forecast(y, fitted)
