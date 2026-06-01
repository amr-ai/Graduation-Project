"""
Category & Product Performance Analysis — Marketing Agent module.

Analyses sales performance per category:
  - Revenue, quantity sold, profit margin per category
  - Top / bottom products within each category
  - Monthly trends with growth / decline / seasonality detection
  - Structured JSON + natural-language insights

Uses the schema intelligence layer to auto-detect column roles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from agents.analytics.schema_intel import build_schema_summary


# ── Helpers ────────────────────────────────────────────────────────────

def _safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0)


def _trend_slope(values: np.ndarray) -> float:
    """Simple linear regression slope over an array of values."""
    x = np.arange(len(values))
    if len(x) < 2 or np.isclose(np.var(x), 0):
        return 0.0
    return float(np.polyfit(x, values, 1)[0])


def _classify_trend(slope: float, cv: float) -> str:
    """Classify trend direction from slope and coefficient of variation."""
    if abs(slope) < 0.01:
        return "stable"
    return "growth" if slope > 0 else "decline"


def _detect_cost_column(df: pd.DataFrame) -> str | None:
    """Try to find a cost / COGS column by name heuristic."""
    cost_keywords = [
        "cost", "cogs", "expense", "purchase_price", "buy_price",
        "wholesale", "base_cost",
    ]
    for col in df.columns:
        lowered = col.lower().replace("_", "").replace("-", "").replace(" ", "")
        for kw in cost_keywords:
            pat = kw.replace("_", "")
            if pat in lowered:
                if pd.api.types.is_numeric_dtype(df[col]):
                    return col
    return None


# ── Core computation ───────────────────────────────────────────────────

def compute_category_performance(df: pd.DataFrame) -> dict:
    """
    Compute per-category and per-product performance metrics.

    Returns a structured dict with keys:
      - categories   : per-category detail
      - overall      : aggregate totals
      - insights     : list of insight strings
      - natural_language_summary : human-readable paragraph
    """
    schema = build_schema_summary(df)

    cat_cols = schema["category_columns"]
    monetary_cols = schema["monetary_columns"]
    qty_col = schema["quantity_column"]
    product_col = schema["product_column"]
    time_col = schema["time_column"]

    # Pick primary category and revenue columns
    cat_col = cat_cols[0] if cat_cols else None
    revenue_col = monetary_cols[0] if monetary_cols else None

    if cat_col is None or revenue_col is None:
        return {
            "available": False,
            "reason": (
                "Cannot perform category analysis — "
                f"{'no category column detected' if cat_col is None else ''}"
                f"{' and ' if cat_col is None and revenue_col is None else ''}"
                f"{'no revenue column detected' if revenue_col is None else ''}"
            ),
        }

    # Prepare working copy
    wdf = df.copy()
    wdf[revenue_col] = _safe_numeric(wdf[revenue_col])
    if qty_col:
        wdf[qty_col] = _safe_numeric(wdf[qty_col])

    cost_col = _detect_cost_column(wdf)

    total_revenue = float(wdf[revenue_col].sum())
    total_quantity = float(wdf[qty_col].sum()) if qty_col else None

    # ── Per-category aggregation ───────────────────────────────────────
    categories: dict[str, dict] = {}
    cat_groups = wdf.groupby(cat_col)

    for cat_name, grp in cat_groups:
        cat_name = str(cat_name)
        cat_revenue = float(grp[revenue_col].sum())
        cat_qty = float(grp[qty_col].sum()) if qty_col else None
        rev_share = round(cat_revenue / total_revenue * 100, 2) if total_revenue else 0.0

        # Profit margin (only when cost column exists)
        margin = None
        if cost_col:
            cat_cost = float(_safe_numeric(grp[cost_col]).sum())
            if cat_revenue > 0:
                margin = round((cat_revenue - cat_cost) / cat_revenue * 100, 2)

        # ── Product breakdown inside category ──────────────────────────
        top_products = []
        bottom_products = []
        if product_col:
            prod_rev = (
                grp.groupby(product_col)[revenue_col]
                .sum()
                .sort_values(ascending=False)
            )
            for p, v in prod_rev.head(5).items():
                entry = {"product": str(p), "revenue": round(float(v), 2)}
                if qty_col:
                    pq = grp.loc[grp[product_col] == p, qty_col].sum()
                    entry["quantity"] = int(pq)
                top_products.append(entry)
            for p, v in prod_rev.tail(5).items():
                entry = {"product": str(p), "revenue": round(float(v), 2)}
                if qty_col:
                    pq = grp.loc[grp[product_col] == p, qty_col].sum()
                    entry["quantity"] = int(pq)
                bottom_products.append(entry)

        # ── Monthly trend for category ─────────────────────────────────
        monthly_trend: dict = {}
        trend_direction = "stable"
        seasonality_score = 0.0

        if time_col:
            try:
                grp_ts = grp.copy()
                grp_ts[time_col] = pd.to_datetime(grp_ts[time_col], errors="coerce")
                grp_ts = grp_ts.dropna(subset=[time_col])
                grp_ts["_period"] = grp_ts[time_col].dt.to_period("M")
                monthly = grp_ts.groupby("_period")[revenue_col].sum().sort_index()

                monthly_trend = {
                    str(k): round(float(v), 2) for k, v in monthly.items()
                }

                if len(monthly) >= 2:
                    vals = monthly.values.astype(float)
                    slope = _trend_slope(vals)
                    mean_val = float(np.mean(vals))
                    std_val = float(np.std(vals))
                    cv = std_val / mean_val if mean_val > 0 else 0.0
                    seasonality_score = round(cv, 4)
                    trend_direction = _classify_trend(slope, cv)

                    # Growth rates
                    growth_rates = monthly.pct_change().dropna()
                    monthly_trend["_avg_growth_rate"] = round(
                        float(growth_rates.mean()), 4
                    )
            except Exception:
                pass

        categories[cat_name] = {
            "revenue": round(cat_revenue, 2),
            "quantity": int(cat_qty) if cat_qty is not None else None,
            "margin_pct": margin,
            "revenue_share_pct": rev_share,
            "top_products": top_products,
            "bottom_products": bottom_products,
            "monthly_trend": monthly_trend,
            "trend_direction": trend_direction,
            "seasonality_score": seasonality_score,
        }

    # ── Rank categories ────────────────────────────────────────────────
    ranked = sorted(categories.items(), key=lambda x: x[1]["revenue"], reverse=True)
    for rank, (name, _) in enumerate(ranked, 1):
        categories[name]["rank"] = rank

    best_cat = ranked[0][0] if ranked else None
    worst_cat = ranked[-1][0] if ranked else None

    # ── Generate insights ──────────────────────────────────────────────
    insights = _generate_category_insights(categories, total_revenue, total_quantity)
    nl_summary = _build_natural_language_summary(
        categories, total_revenue, best_cat, worst_cat
    )

    return {
        "available": True,
        "category_column": cat_col,
        "revenue_column": revenue_col,
        "cost_column_detected": cost_col,
        "categories": categories,
        "overall": {
            "total_revenue": round(total_revenue, 2),
            "total_quantity": int(total_quantity) if total_quantity is not None else None,
            "category_count": len(categories),
            "best_category": best_cat,
            "worst_category": worst_cat,
        },
        "insights": insights,
        "natural_language_summary": nl_summary,
    }


# ── Insight generation ─────────────────────────────────────────────────

def _generate_category_insights(
    categories: dict, total_revenue: float, total_quantity: float | None
) -> list[str]:
    """Produce actionable insight strings by comparing categories."""
    insights: list[str] = []
    if not categories:
        return insights

    items = list(categories.items())

    # 1. Revenue concentration
    top_cat, top_data = items[0] if items else (None, {})
    if top_data:
        share = top_data.get("revenue_share_pct", 0)
        if share > 50:
            insights.append(
                f"{top_cat} dominates with {share}% of total revenue — "
                f"high dependency risk on a single category."
            )
        elif share > 30:
            insights.append(
                f"{top_cat} leads with {share}% of total revenue."
            )

    # 2. Margin comparison (only when margins are available)
    cats_with_margin = [
        (n, d) for n, d in items if d.get("margin_pct") is not None
    ]
    if len(cats_with_margin) >= 2:
        by_margin = sorted(cats_with_margin, key=lambda x: x[1]["margin_pct"])
        lowest_name, lowest = by_margin[0]
        highest_name, highest = by_margin[-1]
        insights.append(
            f"{highest_name} has the highest margin ({highest['margin_pct']}%) "
            f"while {lowest_name} has the lowest ({lowest['margin_pct']}%)."
        )
        # Revenue vs margin mismatch
        revenue_leader = max(items, key=lambda x: x[1]["revenue"])[0]
        if revenue_leader == lowest_name:
            insights.append(
                f"{revenue_leader} generates the most revenue but has the "
                f"lowest margin — consider pricing or cost optimization."
            )

    # 3. Trend-based insights
    growing = [n for n, d in items if d.get("trend_direction") == "growth"]
    declining = [n for n, d in items if d.get("trend_direction") == "decline"]
    if growing:
        insights.append(
            f"Growing categories: {', '.join(growing)} — consider increasing "
            f"inventory and marketing spend."
        )
    if declining:
        insights.append(
            f"Declining categories: {', '.join(declining)} — investigate root "
            f"causes and consider promotional campaigns."
        )

    # 4. Seasonality
    seasonal = [
        (n, d["seasonality_score"])
        for n, d in items
        if d.get("seasonality_score", 0) > 0.3
    ]
    if seasonal:
        names = ", ".join(f"{n} (CV={s:.2f})" for n, s in seasonal)
        insights.append(
            f"High seasonality detected in: {names} — "
            f"plan inventory and campaigns accordingly."
        )

    # 5. Underperforming products
    for name, data in items:
        bottoms = data.get("bottom_products", [])
        if bottoms:
            worst = bottoms[0]
            insights.append(
                f"In {name}, '{worst['product']}' is the lowest performer "
                f"(${worst['revenue']:,.2f} revenue) — review or discontinue."
            )

    return insights


def _build_natural_language_summary(
    categories: dict,
    total_revenue: float,
    best_cat: str | None,
    worst_cat: str | None,
) -> str:
    """Build a concise paragraph summarising category performance."""
    if not categories:
        return "No category data available for analysis."

    n = len(categories)
    parts = [
        f"Across {n} categories, total revenue is ${total_revenue:,.2f}."
    ]

    if best_cat and best_cat in categories:
        best = categories[best_cat]
        parts.append(
            f" {best_cat} is the top performer with ${best['revenue']:,.2f} "
            f"({best['revenue_share_pct']}% share)."
        )

    if worst_cat and worst_cat in categories and worst_cat != best_cat:
        worst = categories[worst_cat]
        parts.append(
            f" {worst_cat} ranks last at ${worst['revenue']:,.2f} "
            f"({worst['revenue_share_pct']}% share)."
        )

    # Margin note
    cats_with_margin = {
        n: d["margin_pct"]
        for n, d in categories.items()
        if d.get("margin_pct") is not None
    }
    if cats_with_margin:
        best_m = max(cats_with_margin, key=cats_with_margin.get)  # type: ignore[arg-type]
        worst_m = min(cats_with_margin, key=cats_with_margin.get)  # type: ignore[arg-type]
        if best_m != worst_m:
            parts.append(
                f" {best_m} has the best margin ({cats_with_margin[best_m]}%) "
                f"compared to {worst_m} ({cats_with_margin[worst_m]}%)."
            )

    # Trend note
    growing = [n for n, d in categories.items() if d.get("trend_direction") == "growth"]
    declining = [n for n, d in categories.items() if d.get("trend_direction") == "decline"]
    if growing:
        parts.append(f" {', '.join(growing)} {'shows' if len(growing) == 1 else 'show'} growth trends.")
    if declining:
        parts.append(f" {', '.join(declining)} {'is' if len(declining) == 1 else 'are'} declining.")

    return "".join(parts)
