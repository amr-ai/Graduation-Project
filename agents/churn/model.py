"""
Churn model: windowed labelling + gradient-boosted classifier, with a
recency heuristic as a graceful fallback for small / single-class datasets.

Labelling (no explicit churn column needed):
  cutoff   = snapshot - horizon
  features = each customer's behaviour up to `cutoff`
  label    = 1 (churned) if the customer did NOT purchase in (cutoff, snapshot]

The fitted model is then applied to features computed as-of `snapshot` to score
the probability that each current customer churns over the *next* horizon.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import compute_customer_features, feature_columns

MIN_TRAIN_CUSTOMERS = 40


def _label_churn(
    tx: pd.DataFrame, cutoff: pd.Timestamp, snapshot: pd.Timestamp, schema: dict
) -> pd.DataFrame:
    cust = schema["customer_column"]
    pre = tx[tx["_dt"] <= cutoff]
    feats = compute_customer_features(pre, cutoff, schema)
    post = tx[(tx["_dt"] > cutoff) & (tx["_dt"] <= snapshot)]
    active_after = set(post[cust].unique())
    feats["churned"] = (~feats.index.isin(active_after)).astype(int)
    return feats


def train_and_predict(
    tx: pd.DataFrame,
    snapshot: pd.Timestamp,
    horizon: int,
    schema: dict,
    test_size: float = 0.25,
    random_state: int = 42,
) -> dict | None:
    """Train the classifier and score all current customers.

    Returns ``None`` when the data can't support a supervised model
    (too few customers or only one class), so the caller can fall back.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import train_test_split

    cutoff = snapshot - pd.Timedelta(days=horizon)
    feats = _label_churn(tx, cutoff, snapshot, schema)
    cols = feature_columns(feats)
    if not cols or len(feats) < MIN_TRAIN_CUSTOMERS:
        return None

    y = feats["churned"].to_numpy(dtype=int)
    if len(np.unique(y)) < 2:
        return None

    X = feats[cols].to_numpy(dtype=float)
    strat = y if np.bincount(y).min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )

    model = GradientBoostingClassifier(random_state=random_state)
    model.fit(X_tr, y_tr)

    proba_te = model.predict_proba(X_te)[:, 1]
    pred_te = (proba_te >= 0.5).astype(int)
    auc = float(roc_auc_score(y_te, proba_te)) if len(np.unique(y_te)) > 1 else None

    metrics = {
        "name": "Gradient Boosting",
        "auc": round(auc, 3) if auc is not None else None,
        "accuracy": round(float(accuracy_score(y_te, pred_te)), 3),
        "precision": round(float(precision_score(y_te, pred_te, zero_division=0)), 3),
        "recall": round(float(recall_score(y_te, pred_te, zero_division=0)), 3),
        "f1": round(float(f1_score(y_te, pred_te, zero_division=0)), 3),
        "base_churn_rate": round(float(y.mean()), 3),
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "confusion_matrix": confusion_matrix(y_te, pred_te).tolist(),
        "threshold": 0.5,
    }

    importances = sorted(
        ({"feature": f, "importance": round(float(i), 4)}
         for f, i in zip(cols, model.feature_importances_)),
        key=lambda d: -d["importance"],
    )

    # Live scoring: features as-of the snapshot for every current customer.
    live = compute_customer_features(tx, snapshot, schema)
    live_proba = model.predict_proba(live[cols].to_numpy(dtype=float))[:, 1]
    live = live.copy()
    live["churn_probability"] = np.round(live_proba, 4)

    return {"metrics": metrics, "importances": importances, "scored": live,
            "feature_cols": cols, "cutoff": cutoff}


def heuristic_predict(
    tx: pd.DataFrame, snapshot: pd.Timestamp, horizon: int, schema: dict
) -> dict:
    """Fallback when a supervised model can't be trained.

    Probability rises with recency relative to the horizon: a customer whose
    last purchase is older than the horizon is treated as (almost) churned.
    """
    live = compute_customer_features(tx, snapshot, schema)
    recency = live["recency_days"].to_numpy(dtype=float)
    proba = np.clip(recency / max(horizon, 1), 0.0, 1.0)
    live = live.copy()
    live["churn_probability"] = np.round(proba, 4)
    metrics = {
        "name": "Recency heuristic (fallback)",
        "auc": None,
        "note": "Too little history/variation to train a classifier; "
                "scored by recency relative to the churn horizon.",
        "base_churn_rate": round(float((proba >= 0.5).mean()), 3),
    }
    return {"metrics": metrics, "importances": [], "scored": live,
            "feature_cols": feature_columns(live), "cutoff": snapshot - pd.Timedelta(days=horizon)}
