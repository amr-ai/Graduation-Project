from __future__ import annotations

import pandas as pd
import pytest

from agents.analytics.kpi_engine import compute_kpis


def test_compute_kpis_basic() -> None:
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=10),
        "revenue": [100, 200, 150, 300, 250, 400, 350, 500, 450, 600],
        "orders": [10, 15, 12, 20, 18, 25, 22, 30, 28, 35],
        "quantity": [20, 30, 25, 40, 35, 50, 45, 60, 55, 70],
    })
    kpis = compute_kpis(df)
    assert kpis["revenue"] == 3300.0
    assert kpis["revenue_column_used"] == "revenue"
    assert isinstance(kpis["aov"], float)


def test_compute_kpis_no_revenue_column() -> None:
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    kpis = compute_kpis(df)
    # Falls back: first numeric column is treated as revenue
    assert kpis.get("revenue") is not None


def test_compute_kpis_with_customers() -> None:
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=6),
        "revenue": [100, 200, 150, 300, 250, 400],
        "customer_id": [1, 2, 1, 3, 2, 1],
    })
    kpis = compute_kpis(df)
    assert kpis["total_customers"] == 3


def test_compute_kpis_growth() -> None:
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=60, freq="D"),
        "revenue": [100 + i for i in range(60)],
    })
    kpis = compute_kpis(df)
    assert "monthly_growth_avg" in kpis
    assert "growth_trend_direction" in kpis


def test_compute_kpis_empty_df() -> None:
    df = pd.DataFrame({"revenue": []})
    kpis = compute_kpis(df)
    assert isinstance(kpis, dict)


def test_new_and_returning_are_a_partition() -> None:
    # C1 has 2 orders (returning), C2 & C3 have 1 order each (one-time).
    df = pd.DataFrame({
        "order_id": ["O1", "O2", "O3", "O4"],
        "customer_id": ["C1", "C1", "C2", "C3"],
        "order_date": pd.date_range("2024-01-01", periods=4),
        "revenue": [100, 200, 150, 300],
    })
    kpis = compute_kpis(df)
    assert kpis["total_customers"] == 3
    assert kpis["returning_customers"] == 1        # C1
    assert kpis["new_customers"] == 2              # C2, C3 (one-time)
    # Non-overlapping partition: the two must add up to the total…
    assert kpis["new_customers"] + kpis["returning_customers"] == kpis["total_customers"]
    # …and the percentages must sum to 100 (the old bug made new_pct ~100 alone).
    assert round(kpis["new_customer_pct"] + kpis["returning_customer_pct"]) == 100
