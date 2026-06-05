"""
Node 1 - Data Profiler (no LLM).

Builds a structured DatasetProfile from the raw DataFrame using pure
pandas analysis. The profile becomes the shared context that every
downstream LLM node uses to understand the data.
"""

from __future__ import annotations

from core.state import GraphState
from tools.profiler_tools import build_dataset_profile, replace_placeholders_with_nan


def profiler_node(state: GraphState) -> dict:
    """
    Standardise placeholders to NaN, profile the DataFrame, and return
    the cleaned raw_df + metadata.
    """
    raw_df = replace_placeholders_with_nan(state["raw_df"])

    print("-- Profiler: analysing dataset ...")
    profile = build_dataset_profile(raw_df)
    print(f"   -> {profile.total_rows} rows, {profile.total_cols} cols, "
          f"{profile.duplicate_rows_count} duplicates detected")

    for col in profile.columns:
        flags = [f"inferred: {col.inferred_type}"]
        if col.missing_count > 0:
            label = "missing"
            if col.placeholder_count:
                label += f" ({col.placeholder_count} placeholders)"
            flags.append(f"{col.missing_count} {label} ({col.missing_pct}%)")
        if col.type_mismatch_flag:
            flags.append("type mismatch")
        if col.outlier_flag:
            flags.append("outliers")
        print(f"   {col.name}: {', '.join(flags)}")

    for rel in profile.relationships:
        print(f"   [~] {rel['expression']}  ({rel['match_pct']}% match)")

    return {"raw_df": raw_df, "metadata": profile.model_dump()}
