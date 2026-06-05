"""
KPI Engine — compute business metrics from cleaned e-commerce data.

Resilient to missing columns: every metric falls back gracefully
when its required columns are not detected.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema_intel import build_schema_summary


def _safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0)


def compute_kpis(df: pd.DataFrame) -> dict:
    schema = build_schema_summary(df)
    kpi = {}

    time_col = schema["time_column"]
    monetary_cols = schema["monetary_columns"]
    qty_col = schema["quantity_column"]
    product_col = schema["product_column"]
    customer_col = schema["customer_column"]
    order_col = schema["order_column"]

    # ── Revenue ────────────────────────────────────────────────────────
    revenue_col = monetary_cols[0] if monetary_cols else None
    if revenue_col:
        revenue_series = _safe_numeric(df[revenue_col])
        kpi["revenue"] = round(float(revenue_series.sum()), 2)
        kpi["revenue_column_used"] = revenue_col
        kpi["avg_revenue_per_row"] = round(float(revenue_series.mean()), 2)
    else:
        kpi["revenue"] = 0.0
        kpi["revenue_column_used"] = None

    # ── Multiple monetary columns (if product of price * quantity exists) ──
    if revenue_col and qty_col:
        qty_series = _safe_numeric(df[qty_col])
        computed_rev = _safe_numeric(df[revenue_col]) * qty_series
        kpi["computed_total"] = round(float(computed_rev.sum()), 2)
        kpi["has_price_x_qty"] = True
    else:
        kpi["computed_total"] = None
        kpi["has_price_x_qty"] = False

    # ── Orders ─────────────────────────────────────────────────────────
    if order_col:
        kpi["orders"] = int(df[order_col].nunique())
        kpi["order_column_used"] = order_col
        if revenue_col:
            order_rev = df.groupby(order_col)[revenue_col].sum()
            kpi["aov"] = round(float(order_rev.mean()), 2)
            kpi["aov_median"] = round(float(order_rev.median()), 2)
            kpi["min_order_value"] = round(float(order_rev.min()), 2)
            kpi["max_order_value"] = round(float(order_rev.max()), 2)
    else:
        kpi["orders"] = len(df)
        kpi["order_column_used"] = None
        kpi["aov"] = kpi.get("avg_revenue_per_row", 0.0)

    # ── Items per order ────────────────────────────────────────────────
    if order_col and qty_col:
        items_per_order = df.groupby(order_col)[qty_col].sum()
        kpi["items_per_order_avg"] = round(float(items_per_order.mean()), 2)
        kpi["items_per_order_median"] = round(float(items_per_order.median()), 2)
    else:
        kpi["items_per_order_avg"] = None

    # ── Revenue per customer ───────────────────────────────────────────
    if customer_col and revenue_col:
        rpc = df.groupby(customer_col)[revenue_col].sum()
        kpi["revenue_per_customer_avg"] = round(float(rpc.mean()), 2)
        kpi["revenue_per_customer_median"] = round(float(rpc.median()), 2)
        kpi["total_customers"] = int(df[customer_col].nunique())
    elif customer_col:
        kpi["total_customers"] = int(df[customer_col].nunique())
        kpi["revenue_per_customer_avg"] = None
    else:
        kpi["total_customers"] = None
        kpi["revenue_per_customer_avg"] = None

    # ── Top / bottom products ──────────────────────────────────────────
    if product_col and revenue_col:
        prod_rev = df.groupby(product_col)[revenue_col].sum().sort_values(ascending=False)
        kpi["top_products"] = {
            str(k): round(float(v), 2)
            for k, v in prod_rev.head(10).items()
        }
        kpi["bottom_products"] = {
            str(k): round(float(v), 2)
            for k, v in prod_rev.tail(10).items()
        }
        kpi["product_count"] = int(prod_rev.count())
    elif product_col:
        kpi["product_count"] = int(df[product_col].nunique())
        kpi["top_products"] = {}
        kpi["bottom_products"] = {}
    else:
        kpi["product_count"] = None
        kpi["top_products"] = {}
        kpi["bottom_products"] = {}

    # ── Customer segmentation: new vs returning ────────────────────────
    if customer_col and time_col and order_col:
        try:
            df_time = df.copy()
            df_time[time_col] = pd.to_datetime(df_time[time_col], errors="coerce")
            first_purchase = df_time.groupby(customer_col)[time_col].min().to_dict()
            df_time["is_first_purchase"] = df_time.apply(
                lambda r: r[time_col] == first_purchase.get(r[customer_col]), axis=1
            )
            kpi["new_customers"] = int(df_time[df_time["is_first_purchase"]][customer_col].nunique())
            kpi["returning_customers"] = int(
                df_time[~df_time["is_first_purchase"]][customer_col].nunique()
            )
            if kpi["total_customers"] and kpi["total_customers"] > 0:
                kpi["new_customer_pct"] = round(
                    kpi["new_customers"] / kpi["total_customers"] * 100, 1
                )
                kpi["returning_customer_pct"] = round(
                    kpi["returning_customers"] / kpi["total_customers"] * 100, 1
                )
        except Exception:
            kpi["new_customers"] = None
            kpi["returning_customers"] = None
    else:
        kpi["new_customers"] = None
        kpi["returning_customers"] = None

    # ── Growth metrics (if time exists) ────────────────────────────────
    if time_col and revenue_col:
        try:
            df_t = df.copy()
            df_t[time_col] = pd.to_datetime(df_t[time_col], errors="coerce")
            df_t["_period"] = df_t[time_col].dt.to_period("M")
            monthly_rev = df_t.groupby("_period")[revenue_col].sum().sort_index()
            if len(monthly_rev) >= 2:
                growth = monthly_rev.pct_change().dropna()
                kpi["monthly_growth_avg"] = round(float(growth.mean()), 4)
                kpi["monthly_growth_std"] = round(float(growth.std()), 4)
                kpi["monthly_growth_min"] = round(float(growth.min()), 4)
                kpi["monthly_growth_max"] = round(float(growth.max()), 4)
                kpi["growth_trend_direction"] = (
                    "up" if kpi["monthly_growth_avg"] > 0.01
                    else "down" if kpi["monthly_growth_avg"] < -0.01
                    else "flat"
                )
            else:
                kpi["monthly_growth_avg"] = None
        except Exception:
            kpi["monthly_growth_avg"] = None
    else:
        kpi["monthly_growth_avg"] = None

    # ── Category breakdown ─────────────────────────────────────────────
    cat_cols = schema["category_columns"]
    if cat_cols and revenue_col:
        kpi["category_revenue"] = {}
        for cat in cat_cols:
            cat_rev = df.groupby(cat)[revenue_col].sum().sort_values(ascending=False)
            kpi["category_revenue"][cat] = {
                str(k): round(float(v), 2)
                for k, v in cat_rev.head(20).items()
            }

    kpi["_schema"] = schema
    return kpi
