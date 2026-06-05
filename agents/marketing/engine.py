"""
Marketing Analytics Engine — main orchestrator.

Runs the deterministic marketing analytics pipeline:
  1. Schema intelligence (column-role detection)
  2. Core business KPIs (reused from the analytics engine)
  3. RFM customer segmentation
  4. Marketing-specific KPIs (repeat rate, CLV proxy, churn-risk base)
  5. Channel / dimension performance (region, category, payment, ...)

Returns a structured payload consumable by the marketing strategy / campaign
LLM agent.  No LLM runs here — everything is computed from the data.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from agents.analytics.kpi_engine import compute_kpis
from agents.analytics.schema_intel import build_schema_summary
from tools.db_tools import load_df_from_pg

from .segmentation import compute_rfm

# Segments considered "at risk" / lapsing for the churn-base metric.
_CHURN_RISK_SEGMENTS = {
    "At Risk", "Can't Lose Them", "About to Sleep", "Hibernating", "Lost",
}
# Segments worth acquiring-style nurturing.
_GROWTH_SEGMENTS = {"New Customers", "Promising", "Potential Loyalists"}


def _safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0)


def _is_text_like(s: pd.Series) -> bool:
    """True for object / string / categorical columns (pandas 2.x and 3.x)."""
    return (
        s.dtype == object
        or pd.api.types.is_string_dtype(s)
        or isinstance(s.dtype, pd.CategoricalDtype)
    )


def compute_marketing_kpis(df: pd.DataFrame, kpi: dict, rfm: dict) -> dict:
    """Derive marketing-oriented KPIs from the core KPIs and RFM output."""
    out: dict = {}

    total_customers = kpi.get("total_customers")
    returning = kpi.get("returning_customers")
    new = kpi.get("new_customers")

    # ── Repeat-purchase / loyalty ──────────────────────────────────────────
    if total_customers and returning is not None:
        out["repeat_purchase_rate"] = round(returning / total_customers * 100, 1)
        out["one_time_buyer_rate"] = round((total_customers - returning) / total_customers * 100, 1)
    else:
        out["repeat_purchase_rate"] = None
        out["one_time_buyer_rate"] = None

    out["new_customers"] = new
    out["returning_customers"] = returning
    out["new_customer_pct"] = kpi.get("new_customer_pct")
    out["returning_customer_pct"] = kpi.get("returning_customer_pct")

    # ── CLV proxy (avg revenue per customer) ───────────────────────────────
    out["avg_customer_value"] = kpi.get("revenue_per_customer_avg")
    out["median_customer_value"] = kpi.get("revenue_per_customer_median")
    out["aov"] = kpi.get("aov")
    out["aov_median"] = kpi.get("aov_median")

    # ── Churn-risk customer base (from RFM) ────────────────────────────────
    if rfm.get("available"):
        segs = rfm["segments"]
        churn_count = sum(segs[s]["count"] for s in segs if s in _CHURN_RISK_SEGMENTS)
        churn_rev = sum(segs[s]["revenue"] for s in segs if s in _CHURN_RISK_SEGMENTS)
        growth_count = sum(segs[s]["count"] for s in segs if s in _GROWTH_SEGMENTS)
        total = rfm["total_customers"] or 1
        out["churn_risk_customers"] = churn_count
        out["churn_risk_pct"] = round(churn_count / total * 100, 1)
        out["revenue_at_risk"] = round(churn_rev, 2)
        out["growth_segment_customers"] = growth_count
        high_value = {"Champions", "Loyal Customers"}
        out["high_value_customers"] = sum(
            segs[s]["count"] for s in segs if s in high_value
        )
    else:
        out["churn_risk_customers"] = None
        out["churn_risk_pct"] = None

    return out


def compute_channel_performance(df: pd.DataFrame, schema: dict, top_n: int = 8) -> dict:
    """
    Revenue and order breakdown across categorical "channel" dimensions
    (region, category, payment method, marketing channel, ...).
    """
    monetary_cols = schema["monetary_columns"]
    revenue_col = monetary_cols[0] if monetary_cols else None

    # Columns that are not useful as marketing "channel" dimensions.
    exclude = set(filter(None, [
        schema.get("customer_column"),
        schema.get("order_column"),
        schema.get("product_column"),
        schema.get("time_column"),
    ]))

    # Candidate dimensions: detected category columns + any low-cardinality
    # text column or one whose name looks like a channel/segment dimension.
    _CHANNEL_KEYWORDS = (
        "region", "channel", "country", "city", "state", "source", "medium",
        "segment", "category", "payment", "department", "type",
    )
    dims: list[str] = list(schema["category_columns"])
    for col in df.columns:
        if col in dims or col in exclude:
            continue
        s = df[col]
        if not _is_text_like(s):
            continue
        nunique = int(s.nunique(dropna=True))
        keyword_hit = any(k in col.lower() for k in _CHANNEL_KEYWORDS)
        if keyword_hit or (2 <= nunique <= 30):
            dims.append(col)

    result: dict[str, dict] = {}
    for dim in dims:
        if dim not in df.columns:
            continue
        try:
            if revenue_col:
                rev = df.groupby(dim)[revenue_col].apply(lambda s: float(_safe_numeric(s).sum()))
                rev = rev.sort_values(ascending=False)
                total = float(rev.sum()) or 1.0
                breakdown = {
                    str(k): {
                        "revenue": round(float(v), 2),
                        "revenue_pct": round(float(v) / total * 100, 1),
                    }
                    for k, v in rev.head(top_n).items()
                }
            else:
                counts = df[dim].value_counts().head(top_n)
                total = int(counts.sum()) or 1
                breakdown = {
                    str(k): {
                        "count": int(v),
                        "share_pct": round(int(v) / total * 100, 1),
                    }
                    for k, v in counts.items()
                }
            if breakdown:
                result[dim] = breakdown
        except Exception:
            continue
    return result


def run_marketing_analytics(
    data_df: pd.DataFrame | None = None,
    table_name: str = "",
) -> dict:
    """
    Execute the full marketing analytics pipeline.

    Parameters
    ----------
    data_df : DataFrame or None
        In-memory cleaned data.  When *None* loads from *table_name*.
    table_name : str
        PostgreSQL table name (used only when *data_df* is None).

    Returns
    -------
    dict
        ``{schema, kpi, rfm, marketing_kpis, channels, metadata}``
    """
    df = data_df if data_df is not None else load_df_from_pg(table_name)

    schema = build_schema_summary(df)
    kpi = compute_kpis(df)
    # The raw schema dict embedded in kpi is verbose; drop it from the payload.
    kpi.pop("_schema", None)

    rfm = compute_rfm(df)
    marketing_kpis = compute_marketing_kpis(df, kpi, rfm)
    channels = compute_channel_performance(df, schema)

    return {
        "schema": schema,
        "kpi": kpi,
        "rfm": rfm,
        "marketing_kpis": marketing_kpis,
        "channels": channels,
        "metadata": {
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": "in_memory" if data_df is not None else f"pg:{table_name}",
        },
    }
