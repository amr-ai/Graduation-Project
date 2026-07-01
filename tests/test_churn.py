from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agents.analytics.schema_intel import build_schema_summary
from agents.churn.engine import run_churn_analysis
from agents.churn.features import compute_customer_features, parse_time
from agents.churn.model import _label_churn


def _build_tx(seed: int = 0) -> pd.DataFrame:
    """Synthetic transactions with a clear churn signal.

    'Active' customers keep buying right up to the snapshot; 'lapsed' customers
    stop after the first ~120 days. Over a 200-day span with a 30-day horizon,
    recency cleanly separates the two groups.
    """
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2024-01-01")
    rows = []
    oid = 0
    for i in range(45):  # active → not churned
        days = sorted(rng.integers(0, 201, size=int(rng.integers(6, 12))))
        days = list(days) + [int(rng.integers(175, 201))]  # ensure a recent buy
        for d in days:
            oid += 1
            rows.append(_row(f"A{i}", base + pd.Timedelta(days=int(d)), oid, rng))
    for i in range(45):  # lapsed → churned
        days = sorted(rng.integers(0, 121, size=int(rng.integers(5, 10))))
        for d in days:
            oid += 1
            rows.append(_row(f"L{i}", base + pd.Timedelta(days=int(d)), oid, rng))
    return pd.DataFrame(rows)


def _row(cust: str, dt: pd.Timestamp, oid: int, rng) -> dict:
    cats = ["Coffee", "Equipment", "Gifts"]
    return {
        "customer_id": cust,
        "order_id": f"ORD-{oid}",
        "order_date": dt.strftime("%Y-%m-%d"),
        "product": f"P{int(rng.integers(0, 8))}",
        "category": cats[int(rng.integers(0, len(cats)))],
        "quantity": int(rng.integers(1, 5)),
        "total_price": round(float(rng.uniform(10, 120)), 2),
    }


# ── Feature engineering ──────────────────────────────────────────────────────


def test_compute_features_basic() -> None:
    df = pd.DataFrame({
        "customer_id": ["C1", "C1", "C2"],
        "order_id": ["O1", "O2", "O3"],
        "order_date": ["2024-01-01", "2024-01-11", "2024-01-05"],
        "total_price": [100.0, 50.0, 30.0],
    })
    schema = build_schema_summary(df)
    tx = parse_time(df, schema["time_column"])
    cutoff = pd.Timestamp("2024-01-21")
    feats = compute_customer_features(tx, cutoff, schema)

    assert feats.loc["C1", "frequency"] == 2          # two distinct orders
    assert feats.loc["C1", "recency_days"] == 10       # last buy 2024-01-11
    assert feats.loc["C1", "monetary"] == 150.0
    assert feats.loc["C2", "frequency"] == 1
    assert not feats.isna().any().any()


def test_label_churn_window() -> None:
    df = pd.DataFrame({
        "customer_id": ["stay", "stay", "gone"],
        "order_id": ["1", "2", "3"],
        "order_date": ["2024-01-01", "2024-03-01", "2024-01-02"],
        "total_price": [20.0, 20.0, 20.0],
    })
    schema = build_schema_summary(df)
    tx = parse_time(df, schema["time_column"])
    snapshot = tx["_dt"].max()                 # 2024-03-01
    cutoff = snapshot - pd.Timedelta(days=30)  # 2024-01-31
    feats = _label_churn(tx, cutoff, snapshot, schema)
    # 'gone' only bought before the cutoff → churned; 'stay' bought after → not.
    assert feats.loc["gone", "churned"] == 1
    assert feats.loc["stay", "churned"] == 0


# ── End-to-end engine ────────────────────────────────────────────────────────


def test_run_churn_analysis_trains_model() -> None:
    df = _build_tx(seed=1)
    payload = run_churn_analysis(data_df=df, horizon_days=30, top_n=50)

    assert payload["available"] is True
    assert payload["model"]["name"] == "Gradient Boosting"
    # Recency is a clean signal here, so the model should be clearly skilful.
    assert payload["model"]["auc"] is not None and payload["model"]["auc"] > 0.75

    rd = payload["risk_distribution"]
    assert rd["high"] + rd["medium"] + rd["low"] == payload["customers_scored"]
    assert payload["revenue_at_risk"] is not None
    assert len(payload["at_risk_customers"]) <= 50

    # Lapsed customers should dominate the top of the at-risk ranking.
    top10 = [c["customer"] for c in payload["at_risk_customers"][:10]]
    assert sum(1 for c in top10 if c.startswith("L")) >= 7
    # The list is now ranked by expected value at risk (prob × spend), descending.
    assert payload["at_risk_ranked_by"] == "expected_value_at_risk"
    losses = [c["expected_loss"] for c in payload["at_risk_customers"]]
    assert losses == sorted(losses, reverse=True)
    assert all("expected_loss" in c and "monetary" in c for c in payload["at_risk_customers"])


def test_run_churn_analysis_requires_customer_and_date() -> None:
    no_cust = pd.DataFrame({"order_date": ["2024-01-01"], "total_price": [10.0]})
    assert run_churn_analysis(data_df=no_cust)["available"] is False

    no_date = pd.DataFrame({"customer_id": ["C1"], "total_price": [10.0]})
    assert run_churn_analysis(data_df=no_date)["available"] is False


def test_run_churn_analysis_short_history() -> None:
    df = pd.DataFrame({
        "customer_id": ["C1", "C2"],
        "order_date": ["2024-01-01", "2024-01-03"],
        "total_price": [10.0, 20.0],
    })
    out = run_churn_analysis(data_df=df, horizon_days=90)
    assert out["available"] is False
