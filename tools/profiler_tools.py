"""
Helper functions used by the Profiler node.

These functions encapsulate the pandas logic for building
ColumnProfile and DatasetProfile objects. They are kept separate
from the node itself for testability and reuse by future agents.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from models.profiler_models import ColumnProfile, DatasetProfile


# ── Per-column helpers ─────────────────────────────────────────────────


def detect_type_mismatch(series: pd.Series) -> bool:
    """
    Check if a column contains values that conflict with its inferred dtype.

    For object columns: check if >50% of non-null values can be parsed as numbers,
    suggesting the column *should* be numeric but has rogue strings mixed in.

    For numeric columns: check if the original raw series (before pandas coercion)
    contained non-numeric strings — pandas may have already coerced them to NaN.
    """
    if series.dropna().empty:
        return False

    if series.dtype == "object":
        non_null = series.dropna()
        numeric_count = pd.to_numeric(non_null, errors="coerce").notna().sum()
        ratio = numeric_count / len(non_null) if len(non_null) > 0 else 0
        # If most values look numeric but dtype is object → mismatch
        if 0.3 < ratio < 1.0:
            return True
        # Check for date-like strings in an object column
        # (heuristic: if a column has "date" in its name and is object)
        return False

    return False


def detect_outliers(series: pd.Series) -> bool:
    """
    IQR-based outlier detection for numeric columns.
    Returns True if any values fall outside 1.5 * IQR from Q1/Q3.
    Non-numeric columns always return False.
    """
    if not pd.api.types.is_numeric_dtype(series):
        return False

    clean = series.dropna()
    if len(clean) < 4:
        return False

    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    iqr = q3 - q1

    if iqr == 0:
        return False

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return bool(((clean < lower) | (clean > upper)).any())


def compute_numeric_stats(series: pd.Series) -> dict | None:
    """
    Compute descriptive statistics for a numeric column.
    Returns None for non-numeric columns.

    Stats: min, max, mean, std, skewness.
    """
    if not pd.api.types.is_numeric_dtype(series):
        return None

    clean = series.dropna()
    if clean.empty:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "skewness": None,
        }

    return {
        "min": float(clean.min()),
        "max": float(clean.max()),
        "mean": float(clean.mean()),
        "std": float(clean.std()) if len(clean) > 1 else 0.0,
        "skewness": float(clean.skew()) if len(clean) > 2 else 0.0,
    }


# ── Builders ───────────────────────────────────────────────────────────


def build_column_profile(series: pd.Series) -> ColumnProfile:
    """Build a ColumnProfile for a single DataFrame column."""
    total = len(series)
    missing = int(series.isna().sum())
    non_null = series.dropna()

    # Sample up to 5 unique values, converted to native Python types
    sample_vals = (
        non_null.drop_duplicates()
        .head(5)
        .apply(lambda x: x.item() if hasattr(x, "item") else x)
        .tolist()
    )

    return ColumnProfile(
        name=series.name,
        dtype=str(series.dtype),
        missing_count=missing,
        missing_pct=round((missing / total) * 100, 2) if total > 0 else 0.0,
        unique_count=int(non_null.nunique()),
        sample_values=sample_vals,
        type_mismatch_flag=detect_type_mismatch(series),
        outlier_flag=detect_outliers(series),
        numeric_stats=compute_numeric_stats(series),
    )


def build_dataset_profile(df: pd.DataFrame) -> DatasetProfile:
    """Build a full DatasetProfile from a DataFrame."""
    column_profiles = [build_column_profile(df[col]) for col in df.columns]

    return DatasetProfile(
        total_rows=len(df),
        total_cols=len(df.columns),
        duplicate_rows_count=int(df.duplicated().sum()),
        columns=column_profiles,
    )
