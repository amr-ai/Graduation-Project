"""
Node 1 - Data Profiler (no LLM).

Builds a structured DatasetProfile from the raw DataFrame using pure
pandas analysis. The profile becomes the shared context that every
downstream LLM node uses to understand the data.
"""

from __future__ import annotations

from core.state import GraphState
from tools.profiler_tools import build_dataset_profile


def profiler_node(state: GraphState) -> dict:
    """
    Analyse the raw DataFrame and produce a validated DatasetProfile.

    Returns:
        dict with 'metadata' key containing DatasetProfile.model_dump().
    """
    raw_df = state["raw_df"]

    print("-- Profiler: analysing dataset ...")
    profile = build_dataset_profile(raw_df)
    print(f"   -> {profile.total_rows} rows, {profile.total_cols} cols, "
          f"{profile.duplicate_rows_count} duplicates detected")

    for col in profile.columns:
        flags = []
        if col.missing_count > 0:
            flags.append(f"{col.missing_count} missing ({col.missing_pct}%)")
        if col.type_mismatch_flag:
            flags.append("type mismatch")
        if col.outlier_flag:
            flags.append("outliers")
        if flags:
            print(f"   [!] {col.name}: {', '.join(flags)}")

    return {"metadata": profile.model_dump()}
