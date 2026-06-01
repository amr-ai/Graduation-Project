"""
Customer Segmentation — Marketing Agent module.

Segments customers using RFM (Recency, Frequency, Monetary) analysis:
  - Computes R, F, M scores per customer (quartile-based, 1–4)
  - Classifies into VIP / Regular / At-Risk / New segments
  - Generates per-segment statistics and recommended marketing actions
  - Outputs structured JSON + human-readable summary

Uses the schema intelligence layer to auto-detect column roles.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from agents.analytics.schema_intel import build_schema_summary


# ── Helpers ────────────────────────────────────────────────────────────

def _safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0)


def _quartile_score(series: pd.Series, ascending: bool = True) -> pd.Series:
    """
    Assign quartile scores 1–4.

    Parameters
    ----------
    series : numeric Series
    ascending : bool
        If True, lower values get score 1, higher get 4.
        If False (e.g. Recency where lower = better), reversed.
    """
    try:
        labels = [1, 2, 3, 4] if ascending else [4, 3, 2, 1]
        return pd.qcut(series.rank(method="first"), q=4, labels=labels).astype(int)
    except (ValueError, TypeError):
        # Fallback for low-cardinality data
        med = series.median()
        if ascending:
            return (series > med).astype(int) * 2 + 1
        else:
            return (series <= med).astype(int) * 2 + 1


# ── Segment classification rules ──────────────────────────────────────

_SEGMENT_ACTIONS = {
    "VIP Customers": (
        "Loyalty rewards program, exclusive early access to new products, "
        "personalized thank-you messages, and dedicated account manager."
    ),
    "Regular Customers": (
        "Engagement campaigns, cross-sell recommendations, "
        "referral incentives, and periodic discount offers."
    ),
    "At-Risk Customers": (
        "Win-back campaign with special discount, personalized re-engagement "
        "emails, satisfaction survey, and limited-time comeback offer."
    ),
    "New Customers": (
        "Welcome email series, first-purchase discount on next order, "
        "product education content, and onboarding guide."
    ),
}


def _classify_segment(row: pd.Series) -> str:
    """
    Classify a customer into a segment based on RFM scores.

    Rules
    -----
    - **New Customers**: frequency == 1 AND recency score >= 3 (recent)
    - **VIP Customers**: frequency score >= 3 AND monetary score >= 3
      AND recency score >= 2
    - **At-Risk Customers**: recency score <= 2 (not recent) AND
      (frequency score >= 2 OR monetary score >= 2)
    - **Regular Customers**: everything else
    """
    r_score = row["r_score"]
    f_score = row["f_score"]
    m_score = row["m_score"]
    frequency = row["frequency"]

    # New = single purchase, recent
    if frequency == 1 and r_score >= 3:
        return "New Customers"

    # VIP = high frequency + high monetary + reasonably recent
    if f_score >= 3 and m_score >= 3 and r_score >= 2:
        return "VIP Customers"

    # At-Risk = not recent, but historically engaged
    if r_score <= 2 and (f_score >= 2 or m_score >= 2):
        return "At-Risk Customers"

    return "Regular Customers"


# ── Core computation ───────────────────────────────────────────────────

def compute_customer_segmentation(df: pd.DataFrame) -> dict:
    """
    Perform RFM-based customer segmentation.

    Returns a structured dict with keys:
      - segments                : per-segment detail
      - rfm_summary             : aggregate RFM averages
      - total_customers         : int
      - insights                : list of insight strings
      - natural_language_summary: human-readable paragraph
    """
    schema = build_schema_summary(df)

    customer_col = schema["customer_column"]
    time_col = schema["time_column"]
    monetary_cols = schema["monetary_columns"]
    order_col = schema["order_column"]
    revenue_col = monetary_cols[0] if monetary_cols else None

    if customer_col is None:
        return {
            "available": False,
            "reason": "No customer identifier column detected.",
        }
    if revenue_col is None:
        return {
            "available": False,
            "reason": "No monetary / revenue column detected.",
        }

    wdf = df.copy()
    wdf[revenue_col] = _safe_numeric(wdf[revenue_col])

    # Parse time column
    if time_col:
        wdf[time_col] = pd.to_datetime(wdf[time_col], errors="coerce")
        wdf = wdf.dropna(subset=[time_col])
        reference_date = wdf[time_col].max() + timedelta(days=1)
    else:
        reference_date = None

    # ── Build RFM table ────────────────────────────────────────────────
    agg_dict: dict = {}

    # Monetary — total spend per customer
    agg_dict["monetary"] = (revenue_col, "sum")

    # Frequency — number of distinct orders (or rows if no order column)
    if order_col:
        agg_dict["frequency"] = (order_col, "nunique")
    else:
        agg_dict["frequency"] = (revenue_col, "count")

    # Recency — days since last purchase
    if time_col:
        agg_dict["last_purchase"] = (time_col, "max")

    rfm = wdf.groupby(customer_col).agg(**agg_dict).reset_index()

    if time_col and reference_date is not None:
        rfm["recency"] = (reference_date - rfm["last_purchase"]).dt.days
    else:
        # Without time column, assign neutral recency
        rfm["recency"] = 0

    # ── Score ──────────────────────────────────────────────────────────
    # Recency: lower = better → ascending=False (low recency → high score)
    rfm["r_score"] = _quartile_score(rfm["recency"], ascending=False)
    # Frequency: higher = better
    rfm["f_score"] = _quartile_score(rfm["frequency"], ascending=True)
    # Monetary: higher = better
    rfm["m_score"] = _quartile_score(rfm["monetary"], ascending=True)

    rfm["rfm_score"] = rfm["r_score"] + rfm["f_score"] + rfm["m_score"]

    # ── Classify ───────────────────────────────────────────────────────
    rfm["segment"] = rfm.apply(_classify_segment, axis=1)

    # ── Aggregate per segment ──────────────────────────────────────────
    total_customers = len(rfm)
    segments: dict[str, dict] = {}

    for seg_name in ["VIP Customers", "Regular Customers", "At-Risk Customers", "New Customers"]:
        seg_df = rfm[rfm["segment"] == seg_name]
        count = len(seg_df)
        if count == 0:
            segments[seg_name] = {
                "count": 0,
                "pct": 0.0,
                "avg_order_value": 0.0,
                "avg_frequency": 0.0,
                "avg_recency_days": 0.0,
                "avg_monetary": 0.0,
                "avg_rfm_score": 0.0,
                "recommended_action": _SEGMENT_ACTIONS[seg_name],
            }
            continue

        avg_monetary = float(seg_df["monetary"].mean())
        avg_frequency = float(seg_df["frequency"].mean())
        avg_recency = float(seg_df["recency"].mean())
        avg_rfm = float(seg_df["rfm_score"].mean())

        # AOV = total monetary / total frequency
        total_seg_monetary = float(seg_df["monetary"].sum())
        total_seg_frequency = float(seg_df["frequency"].sum())
        aov = total_seg_monetary / total_seg_frequency if total_seg_frequency > 0 else 0.0

        segments[seg_name] = {
            "count": count,
            "pct": round(count / total_customers * 100, 1),
            "avg_order_value": round(aov, 2),
            "avg_frequency": round(avg_frequency, 2),
            "avg_recency_days": round(avg_recency, 1),
            "avg_monetary": round(avg_monetary, 2),
            "avg_rfm_score": round(avg_rfm, 2),
            "recommended_action": _SEGMENT_ACTIONS[seg_name],
        }

    # ── RFM summary ────────────────────────────────────────────────────
    rfm_summary = {
        "avg_recency": round(float(rfm["recency"].mean()), 1),
        "avg_frequency": round(float(rfm["frequency"].mean()), 2),
        "avg_monetary": round(float(rfm["monetary"].mean()), 2),
        "median_recency": round(float(rfm["recency"].median()), 1),
        "median_frequency": round(float(rfm["frequency"].median()), 2),
        "median_monetary": round(float(rfm["monetary"].median()), 2),
    }

    # ── Insights ───────────────────────────────────────────────────────
    insights = _generate_segmentation_insights(segments, total_customers, rfm_summary)
    nl_summary = _build_segmentation_summary(segments, total_customers, rfm_summary)

    return {
        "available": True,
        "customer_column": customer_col,
        "revenue_column": revenue_col,
        "time_column": time_col,
        "segments": segments,
        "rfm_summary": rfm_summary,
        "total_customers": total_customers,
        "insights": insights,
        "natural_language_summary": nl_summary,
    }


# ── Insight generation ─────────────────────────────────────────────────

def _generate_segmentation_insights(
    segments: dict, total: int, rfm_summary: dict
) -> list[str]:
    """Produce actionable insight strings from segment data."""
    insights: list[str] = []
    if total == 0:
        return insights

    vip = segments.get("VIP Customers", {})
    at_risk = segments.get("At-Risk Customers", {})
    new = segments.get("New Customers", {})
    regular = segments.get("Regular Customers", {})

    # VIP concentration
    if vip.get("count", 0) > 0:
        vip_pct = vip["pct"]
        vip_value = vip["avg_monetary"]
        insights.append(
            f"VIP Customers represent {vip_pct}% of customers "
            f"with an average spend of ${vip_value:,.2f} — "
            f"prioritize retention with loyalty rewards."
        )

    # At-risk alert
    if at_risk.get("count", 0) > 0:
        risk_pct = at_risk["pct"]
        risk_recency = at_risk["avg_recency_days"]
        insights.append(
            f"{risk_pct}% of customers are At-Risk (avg {risk_recency:.0f} days "
            f"since last purchase) — launch win-back campaigns immediately."
        )

    # New customer opportunity
    if new.get("count", 0) > 0:
        new_pct = new["pct"]
        new_aov = new["avg_order_value"]
        insights.append(
            f"New Customers make up {new_pct}% with an AOV of ${new_aov:,.2f} — "
            f"focus on onboarding and second-purchase incentives."
        )

    # Regular engagement
    if regular.get("count", 0) > 0:
        reg_pct = regular["pct"]
        reg_freq = regular["avg_frequency"]
        insights.append(
            f"Regular Customers ({reg_pct}%) purchase on average "
            f"{reg_freq:.1f} times — cross-sell and upsell to move them to VIP."
        )

    # VIP vs At-Risk ratio
    if vip.get("count", 0) > 0 and at_risk.get("count", 0) > 0:
        ratio = at_risk["count"] / vip["count"]
        if ratio > 2:
            insights.append(
                f"Warning: At-Risk customers outnumber VIPs by {ratio:.1f}x — "
                f"retention should be the top priority."
            )

    return insights


def _build_segmentation_summary(
    segments: dict, total: int, rfm_summary: dict
) -> str:
    """Build a concise paragraph summarising customer segmentation."""
    if total == 0:
        return "No customer data available for segmentation."

    parts = [f"Out of {total} customers:"]

    for seg_name in ["VIP Customers", "Regular Customers", "At-Risk Customers", "New Customers"]:
        seg = segments.get(seg_name, {})
        count = seg.get("count", 0)
        pct = seg.get("pct", 0)
        if count > 0:
            parts.append(
                f" {count} ({pct}%) are {seg_name}"
                f" (avg spend ${seg.get('avg_monetary', 0):,.2f},"
                f" avg frequency {seg.get('avg_frequency', 0):.1f})."
            )

    parts.append(
        f" Overall average recency is {rfm_summary['avg_recency']:.0f} days,"
        f" frequency is {rfm_summary['avg_frequency']:.1f} orders,"
        f" and monetary value is ${rfm_summary['avg_monetary']:,.2f}."
    )

    return "".join(parts)
