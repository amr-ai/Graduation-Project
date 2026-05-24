"""
Node 5 - Validator (no LLM).

Compares the raw DataFrame against the cleaned DataFrame to produce a
validation report. Flags any regressions (e.g., nulls increasing) and
determines whether the cleaning passed or failed.
"""

from __future__ import annotations

import pandas as pd

from core.state import GraphState


def validator_node(state: GraphState) -> dict:
    """
    Validate the cleaned DataFrame against the original.

    Returns:
        dict with 'validation_report' containing comparison metrics
        and a 'passed' boolean flag.
    """
    print("\n-- Validator: comparing raw vs. cleaned data ...")

    raw_df = state["raw_df"]
    execution_result = state["execution_result"]

    # Handle case where execution failed and we were sent here with a failure flag
    if not execution_result.get("success", False) or execution_result.get("clean_df") is None:
        print("   [X] No clean DataFrame available - execution failed")
        return {
            "validation_report": {
                "passed": False,
                "reason": "Code execution failed after max retries",
                "error": execution_result.get("error", "Unknown error"),
                "rows_before": len(raw_df),
                "rows_after": 0,
                "nulls_before": int(raw_df.isna().sum().sum()),
                "nulls_after": None,
                "duplicates_before": int(raw_df.duplicated().sum()),
                "duplicates_after": None,
                "column_details": {},
                "regression_warnings": [],
            }
        }

    clean_df = execution_result["clean_df"]

    # ── Row-level comparison ───────────────────────────────────────────
    rows_before = len(raw_df)
    rows_after = len(clean_df)

    # ── Duplicate comparison ───────────────────────────────────────────
    duplicates_before = int(raw_df.duplicated().sum())
    duplicates_after = int(clean_df.duplicated().sum())

    # ── Null comparison per column ─────────────────────────────────────
    nulls_before_total = int(raw_df.isna().sum().sum())
    nulls_after_total = int(clean_df.isna().sum().sum())

    column_details = {}
    regression_warnings = []

    # Compare columns that exist in both DataFrames
    common_cols = set(raw_df.columns) & set(clean_df.columns)
    for col in sorted(common_cols):
        nb = int(raw_df[col].isna().sum())
        na = int(clean_df[col].isna().sum())
        column_details[col] = {
            "nulls_before": nb,
            "nulls_after": na,
            "nulls_reduced": nb - na,
        }
        if na > nb:
            warning = f"Column '{col}': nulls INCREASED from {nb} to {na}"
            regression_warnings.append(warning)
            print(f"   [!] REGRESSION: {warning}")

    # Flag columns that were dropped
    dropped_cols = set(raw_df.columns) - set(clean_df.columns)
    if dropped_cols:
        print(f"   [i] Columns removed: {', '.join(sorted(dropped_cols))}")

    # Flag new columns
    new_cols = set(clean_df.columns) - set(raw_df.columns)
    if new_cols:
        print(f"   [i] New columns added: {', '.join(sorted(new_cols))}")

    # ── Overall verdict ────────────────────────────────────────────────
    passed = (
        len(regression_warnings) == 0
        and nulls_after_total <= nulls_before_total
        and rows_after > 0
    )

    report = {
        "passed": passed,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_removed": rows_before - rows_after,
        "nulls_before": nulls_before_total,
        "nulls_after": nulls_after_total,
        "nulls_reduced": nulls_before_total - nulls_after_total,
        "duplicates_before": duplicates_before,
        "duplicates_after": duplicates_after,
        "column_details": column_details,
        "regression_warnings": regression_warnings,
        "dropped_columns": sorted(dropped_cols) if dropped_cols else [],
        "new_columns": sorted(new_cols) if new_cols else [],
    }

    status = "PASSED" if passed else "FAILED"
    print(f"\n   Validation {status}")
    print(f"   Rows: {rows_before} -> {rows_after} ({rows_before - rows_after} removed)")
    print(f"   Nulls: {nulls_before_total} -> {nulls_after_total} ({nulls_before_total - nulls_after_total} fixed)")
    print(f"   Duplicates: {duplicates_before} -> {duplicates_after}")

    return {"validation_report": report}
