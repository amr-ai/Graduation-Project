"""
Schema Intelligence Layer — heuristically detect column roles from a
DataFrame without any user input.

Detection targets:
  - time / date column
  - revenue / price / monetary columns
  - quantity column
  - product identifier
  - customer identifier
  - transaction / order identifier
  - category / segment columns
"""

from __future__ import annotations

import re

import pandas as pd

_TIME_KEYWORDS = [
    "date", "time", "timestamp", "datetime", "day", "month", "year",
    "order_date", "created_at", "updated_at", "transaction_date",
    "ship_date", "invoice_date", "period",
]
_REVENUE_KEYWORDS = [
    "revenue", "sales", "total", "amount", "price", "spend", "value",
    "gmv", "net", "gross", "profit", "income", "cost", "fee", "payment",
    "total_spent", "unit_price", "line_total", "subtotal",
]
_QUANTITY_KEYWORDS = [
    "quantity", "qty", "count", "num", "number", "units", "volume",
    "items", "pcs",
]
_PRODUCT_KEYWORDS = [
    "product", "item", "sku", "stock_code", "catalog", "article",
    "good", "merchandise", "variant",
]
_CUSTOMER_KEYWORDS = [
    "customer", "client", "user", "member", "buyer", "account",
    "contact", "email", "phone", "loyalty",
]
_ORDER_KEYWORDS = [
    "order", "transaction", "invoice", "receipt", "ticket", "reference",
    "cart", "checkout", "purchase",
]
_CATEGORY_KEYWORDS = [
    "category", "segment", "group", "class", "type", "department",
    "division", "family", "subcategory", "channel",
]


def _is_text_like(series: pd.Series) -> bool:
    """True for object / string / categorical columns (pandas 2.x and 3.x).

    pandas 3.0 stores plain text as a dedicated ``string`` dtype rather than
    ``object``, so a bare ``dtype == object`` check misses text columns.
    """
    dtype = series.dtype
    return (
        dtype == object
        or pd.api.types.is_string_dtype(series)
        or isinstance(dtype, pd.CategoricalDtype)
    )


def _col_match(name: str, keywords: list[str]) -> bool:
    lowered = name.lower().replace("_", "").replace("-", "").replace(" ", "")
    for kw in keywords:
        pat = kw.lower().replace("_", "")
        if pat in lowered:
            return True
    return False


def _score_column(name: str, keywords: list[str]) -> int:
    lowered = name.lower().replace("_", "").replace("-", "").replace(" ", "")
    score = 0
    for kw in keywords:
        pat = kw.lower().replace("_", "")
        if pat in lowered:
            score += 1
            if lowered == pat or lowered.startswith(pat) or lowered.endswith(pat):
                score += 2
    return score


def detect_time_column(df: pd.DataFrame) -> str | None:
    candidates = []
    for col in df.columns:
        score = _score_column(col, _TIME_KEYWORDS)
        if score > 0:
            candidates.append((score, col))
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            candidates.append((3, col))

    if not candidates:
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                candidates.append((1, col))
            elif df[col].dtype == "object":
                sample = df[col].dropna().head(20)
                if sample.empty:
                    continue
                try:
                    pd.to_datetime(sample)
                    candidates.append((1, col))
                except (ValueError, TypeError):
                    pass

    if not candidates:
        return None
    candidates.sort(key=lambda x: -x[0])
    return candidates[0][1]


# Column names that represent *gross revenue* (the line/order total) and should
# outrank per-unit or non-revenue money columns (unit_price, cost, profit) when
# choosing the primary revenue column. Boost only — nothing is excluded.
_STRONG_REVENUE = [
    "revenue", "sales", "gmv", "total", "amount", "grand_total",
    "line_total", "subtotal", "net_total", "order_value", "total_spent",
]


def detect_monetary_columns(df: pd.DataFrame) -> list[str]:
    candidates = []
    for col in df.columns:
        score = _score_column(col, _REVENUE_KEYWORDS)
        if score > 0 and pd.api.types.is_numeric_dtype(df[col]):
            # Prefer true revenue/total columns over unit_price / cost / profit.
            if _col_match(col, _STRONG_REVENUE):
                score += 6
            candidates.append((score, col))
    if not candidates:
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]) and col.lower() not in ("id", "index"):
                candidates.append((0, col))
    candidates.sort(key=lambda x: -x[0])
    return [c[1] for c in candidates]


def detect_quantity_column(df: pd.DataFrame) -> str | None:
    candidates = []
    for col in df.columns:
        score = _score_column(col, _QUANTITY_KEYWORDS)
        if score > 0 and pd.api.types.is_numeric_dtype(df[col]):
            candidates.append((score, col))
    if not candidates:
        for col in df.columns:
            if pd.api.types.is_integer_dtype(df[col]) and col.lower() not in ("id", "index"):
                candidates.append((0, col))
    candidates.sort(key=lambda x: -x[0])
    return candidates[0][1] if candidates else None


def detect_product_column(df: pd.DataFrame) -> str | None:
    candidates = []
    for col in df.columns:
        score = _score_column(col, _PRODUCT_KEYWORDS)
        if score > 0:
            candidates.append((score, col))
    candidates.sort(key=lambda x: -x[0])
    return candidates[0][1] if candidates else None


def detect_customer_column(df: pd.DataFrame) -> str | None:
    candidates = []
    for col in df.columns:
        score = _score_column(col, _CUSTOMER_KEYWORDS)
        if score > 0:
            candidates.append((score, col))
    candidates.sort(key=lambda x: -x[0])
    return candidates[0][1] if candidates else None


def detect_order_column(df: pd.DataFrame) -> str | None:
    candidates = []
    for col in df.columns:
        score = _score_column(col, _ORDER_KEYWORDS)
        if score > 0:
            candidates.append((score, col))
    candidates.sort(key=lambda x: -x[0])
    return candidates[0][1] if candidates else None


def detect_category_columns(df: pd.DataFrame) -> list[str]:
    candidates = []
    for col in df.columns:
        score = _score_column(col, _CATEGORY_KEYWORDS)
        if score > 0 and _is_text_like(df[col]):
            candidates.append((score, col))
    candidates.sort(key=lambda x: -x[0])
    return [c[1] for c in candidates]


def build_schema_summary(df: pd.DataFrame) -> dict:
    return {
        "time_column": detect_time_column(df),
        "monetary_columns": detect_monetary_columns(df),
        "quantity_column": detect_quantity_column(df),
        "product_column": detect_product_column(df),
        "customer_column": detect_customer_column(df),
        "order_column": detect_order_column(df),
        "category_columns": detect_category_columns(df),
        "row_count": len(df),
        "column_count": len(df.columns),
    }
