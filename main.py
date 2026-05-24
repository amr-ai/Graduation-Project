"""
Smart Analyst — Data Cleaning Pipeline (Phase 1, Stage 1)

Entry point that loads raw CSV data, runs it through the LangGraph
cleaning pipeline, and prints the results.

Usage:
    python main.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from core.state import GraphState
from graphs.cleaning_graph import build_cleaning_graph


def main() -> None:
    # ── Load environment ───────────────────────────────────────────────
    load_dotenv()

    if not os.environ.get("GROQ_API_KEY") or os.environ["GROQ_API_KEY"] == "your_groq_api_key_here":
        print("ERROR: Please set a valid GROQ_API_KEY in smart_analyst/.env")
        sys.exit(1)

    # ── Load data ──────────────────────────────────────────────────────
    data_path = Path(__file__).resolve().parent / "data" / "sample.csv"
    if not data_path.exists():
        print(f"ERROR: Sample data not found at {data_path}")
        print("Run: python generate_sample_data.py")
        sys.exit(1)

    print("=" * 60)
    print("  SMART ANALYST - Data Cleaning Pipeline")
    print("=" * 60)

    raw_df = pd.read_csv(str(data_path))
    print(f"\nLoaded {len(raw_df)} rows x {len(raw_df.columns)} cols from {data_path.name}")

    # ── Initialize state ───────────────────────────────────────────────
    initial_state: GraphState = {
        "raw_df": raw_df,
        "file_path": str(data_path),
        "model_name": "llama-3.3-70b-versatile",
        "metadata": {},
        "cleaning_plan": "",
        "generated_code": "",
        "execution_result": {},
        "validation_report": {},
        "transformation_log": [],
        "retry_count": 0,
        "last_error": "",
        "messages": [],
    }

    # ── Build and run the pipeline ─────────────────────────────────────
    print("\nStarting cleaning pipeline...\n")

    graph = build_cleaning_graph()
    final_state = graph.invoke(initial_state)

    # ── Print results ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PIPELINE RESULTS")
    print("=" * 60)

    # Cleaning Plan
    print("\n--- CLEANING PLAN ---")
    print(final_state.get("cleaning_plan", "No plan generated"))

    # Transformation Log
    print("\n--- TRANSFORMATION LOG ---")
    log = final_state.get("transformation_log", [])
    if log:
        for i, entry in enumerate(log, 1):
            print(f"  {i}. {entry}")
    else:
        print("  No transformations logged.")

    # Validation Report
    print("\n--- VALIDATION REPORT ---")
    report = final_state.get("validation_report", {})
    if report:
        # Print without the DataFrame to keep output clean
        printable = {k: v for k, v in report.items() if k != "clean_df"}
        print(json.dumps(printable, indent=2, default=str))
    else:
        print("  No validation report available.")

    # Summary
    print("\n" + "=" * 60)
    passed = report.get("passed", False)
    status = "PASSED" if passed else "FAILED"
    print(f"  Pipeline complete - Validation: {status}")
    print(f"  Retries used: {final_state.get('retry_count', 0)}")
    print("=" * 60)

    # Save cleaned data to output folder
    clean_df = final_state.get("execution_result", {}).get("clean_df")
    if clean_df is not None:
        output_dir = Path(__file__).resolve().parent / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "cleaned_data.csv"
        clean_df.to_csv(output_path, index=False)
        print(f"\n[INFO] Saved cleaned data ({len(clean_df)} rows) to: {output_path}")


if __name__ == "__main__":
    main()
