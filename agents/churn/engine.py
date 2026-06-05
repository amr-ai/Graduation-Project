"""
Churn Analytics Engine — main orchestrator (no LLM here).

Validates the data, trains/scoring the churn model (or heuristic fallback), and
assembles a structured payload consumable by the UI and the LLM retention-plan
agent:

    {available, horizon_days, snapshot_date, cutoff_date, model, feature_importance,
     risk_distribution, customers_scored, revenue_at_risk, expected_revenue_at_risk,
     at_risk_customers, metadata}
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from agents.analytics.schema_intel import build_schema_summary
from tools.db_tools import load_df_from_pg

from .features import parse_time
from .model import heuristic_predict, train_and_predict

HIGH_RISK = 0.70
MED_RISK = 0.40


def _risk_tier(p: float) -> str:
    if p >= HIGH_RISK:
        return "High"
    if p >= MED_RISK:
        return "Medium"
    return "Low"


def _find_name_column(df: pd.DataFrame, customer_col: str) -> str | None:
    for col in df.columns:
        if col == customer_col:
            continue
        low = col.lower()
        if "name" in low and ("customer" in low or "client" in low or "contact" in low):
            return col
    return None


def run_churn_analysis(
    data_df: pd.DataFrame | None = None,
    table_name: str = "",
    horizon_days: int = 90,
    top_n: int = 200,
) -> dict:
    """Execute the churn pipeline. Returns the structured payload (see module doc)."""
    df = data_df if data_df is not None else load_df_from_pg(table_name)

    schema = build_schema_summary(df)
    cust = schema["customer_column"]
    time_col = schema["time_column"]

    if not cust:
        return {"available": False, "reason": "No customer identifier column detected."}
    if not time_col:
        return {"available": False,
                "reason": "No date column detected (needed to measure recency and churn)."}

    tx = parse_time(df, time_col)
    if tx.empty:
        return {"available": False, "reason": f"Date column '{time_col}' has no parseable dates."}

    snapshot = tx["_dt"].max()
    span = int((snapshot - tx["_dt"].min()).days)

    # Keep the horizon feasible: need a pre-window for features and an
    # outcome-window for labels within the available history.
    horizon = min(horizon_days, max(7, span // 3))
    if span < horizon + 14:
        return {"available": False,
                "reason": f"History too short ({span} days) for a {horizon_days}-day churn window."}

    res = train_and_predict(tx, snapshot, horizon, schema)
    if res is None:
        res = heuristic_predict(tx, snapshot, horizon, schema)

    scored = res["scored"]
    p = scored["churn_probability"].to_numpy(dtype=float)

    risk_distribution = {
        "high": int((p >= HIGH_RISK).sum()),
        "medium": int(((p >= MED_RISK) & (p < HIGH_RISK)).sum()),
        "low": int((p < MED_RISK).sum()),
    }

    has_money = "monetary" in scored.columns
    if has_money:
        money = scored["monetary"].to_numpy(dtype=float)
        revenue_at_risk = float(np.round(money[p >= 0.5].sum(), 2))
        expected_revenue_at_risk = float(np.round((p * money).sum(), 2))
    else:
        revenue_at_risk = None
        expected_revenue_at_risk = None

    # ── Top at-risk customers (with optional human-readable name) ────────────
    name_col = _find_name_column(df, cust)
    name_map = {}
    if name_col:
        name_map = df.dropna(subset=[name_col]).groupby(cust)[name_col].first().to_dict()

    ranked = scored.sort_values("churn_probability", ascending=False).head(top_n)
    at_risk_customers = []
    for customer_id, row in ranked.iterrows():
        prob = float(row["churn_probability"])
        entry = {
            "customer": str(customer_id),
            "name": str(name_map.get(customer_id, "")) if name_map else "",
            "churn_probability": round(prob, 4),
            "risk_tier": _risk_tier(prob),
            "recency_days": int(row.get("recency_days", 0)),
            "frequency": int(row.get("frequency", 0)),
        }
        if has_money:
            entry["monetary"] = round(float(row.get("monetary", 0.0)), 2)
        at_risk_customers.append(entry)

    return {
        "available": True,
        "horizon_days": horizon,
        "requested_horizon_days": horizon_days,
        "snapshot_date": str(snapshot.date()),
        "cutoff_date": str(pd.Timestamp(res["cutoff"]).date()),
        "model": res["metrics"],
        "feature_importance": res["importances"],
        "risk_distribution": risk_distribution,
        "customers_scored": int(len(scored)),
        "revenue_at_risk": revenue_at_risk,
        "expected_revenue_at_risk": expected_revenue_at_risk,
        "at_risk_customers": at_risk_customers,
        "metadata": {
            "customer_column": cust,
            "time_column": time_col,
            "monetary_column": schema["monetary_columns"][0] if schema["monetary_columns"] else None,
            "row_count": int(len(df)),
            "history_days": span,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": "in_memory" if data_df is not None else f"pg:{table_name}",
        },
    }
