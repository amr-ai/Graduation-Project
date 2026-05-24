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
from graphs.planner_graph import build_planner_graph
from graphs.cleaning_graph import build_cleaning_graph


def main() -> None:
    load_dotenv()

    if not os.environ.get("GROQ_API_KEY") or os.environ["GROQ_API_KEY"] == "your_groq_api_key_here":
        print("ERROR: Please set a valid GROQ_API_KEY in smart_analyst/.env")
        sys.exit(1)

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

    # ── Stage 1: Planning ──────────────────────────────────────────────
    print("\n--- STAGE 1: Planning ---\n")

    planner_graph = build_planner_graph()
    plan_state = planner_graph.invoke({
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
    })

    plan = plan_state.get("cleaning_plan", "")
    metadata = plan_state.get("metadata", {})
    raw_df = plan_state.get("raw_df", raw_df)  # use placeholder-sanitised version

    # ── Human review ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  HUMAN REVIEW — Cleaning Plan Generated")
    print("=" * 60)
    print("\n" + plan)

    print("\n" + "-" * 60)
    print("Options:")
    print("  [enter]  Approve plan as-is and continue")
    print("  e        Edit the plan")
    print("  q        Quit")
    choice = input("\nChoice: ").strip().lower()

    if choice == "q":
        print("\nPipeline cancelled by user.")
        sys.exit(0)

    edited_plan = plan
    if choice == "e":
        print("\nPaste the edited plan below. Finish with Ctrl+Z on a new line:\n")
        lines = []
        try:
            while True:
                line = input()
                lines.append(line)
        except EOFError:
            pass
        pasted = "\n".join(lines).strip()
        if pasted:
            edited_plan = pasted

    # ── Stage 2: Cleaning ──────────────────────────────────────────────
    print("\n--- STAGE 2: Cleaning ---\n")

    cleaning_graph = build_cleaning_graph()
    final_state = cleaning_graph.invoke({
        "raw_df": raw_df,
        "file_path": str(data_path),
        "model_name": "llama-3.3-70b-versatile",
        "metadata": metadata,
        "cleaning_plan": edited_plan,
        "generated_code": "",
        "execution_result": {},
        "validation_report": {},
        "transformation_log": [],
        "retry_count": 0,
        "last_error": "",
        "messages": [],
    })

    # ── Print results ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PIPELINE RESULTS")
    print("=" * 60)

    print("\n--- CLEANING PLAN ---")
    print(final_state.get("cleaning_plan", "No plan generated"))

    print("\n--- TRANSFORMATION LOG ---")
    log = final_state.get("transformation_log", [])
    if log:
        for i, entry in enumerate(log, 1):
            print(f"  {i}. {entry}")
    else:
        print("  No transformations logged.")

    print("\n--- VALIDATION REPORT ---")
    report = final_state.get("validation_report", {})
    if report:
        printable = {k: v for k, v in report.items() if k != "clean_df"}
        print(json.dumps(printable, indent=2, default=str))
    else:
        print("  No validation report available.")

    print("\n" + "=" * 60)
    passed = report.get("passed", False)
    status = "PASSED" if passed else "FAILED"
    print(f"  Pipeline complete - Validation: {status}")
    print(f"  Retries used: {final_state.get('retry_count', 0)}")
    print("=" * 60)

    clean_df = final_state.get("execution_result", {}).get("clean_df")
    if clean_df is not None:
        output_dir = Path(__file__).resolve().parent / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "cleaned_data.csv"
        clean_df.to_csv(output_path, index=False)
        print(f"\n[INFO] Saved cleaned data ({len(clean_df)} rows) to: {output_path}")


if __name__ == "__main__":
    main()
