"""
Node 4 - Code Executor (no LLM).

Runs the LLM-generated clean_data(df) function inside a sandboxed
exec() environment with restricted globals. Captures stdout for the
transformation log and full tracebacks on failure.
"""

from __future__ import annotations

import io
import sys
import traceback

import pandas as pd

from core.state import GraphState

# Restricted globals for the sandboxed exec environment.
# Only pandas and a handful of safe builtins are exposed.
SAFE_BUILTINS = {
    "print": print,   # will be redirected to StringIO
    "range": range,
    "len": len,
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    "round": round,
    "min": min,
    "max": max,
    "abs": abs,
    "sum": sum,
    "sorted": sorted,
    "enumerate": enumerate,
    "zip": zip,
    "isinstance": isinstance,
    "hasattr": hasattr,
    "getattr": getattr,
    "type": type,
    "filter": filter,
    "map": map,
    "any": any,
    "all": all,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "Exception": Exception,
}


def executor_node(state: GraphState) -> dict:
    """
    Execute the generated clean_data(df) in a sandboxed environment.

    Returns:
        dict with 'execution_result', and on failure also
        'retry_count' and 'last_error'.
    """
    code = state["generated_code"]
    raw_df = state["raw_df"]
    retry_count = state.get("retry_count", 0)

    print("\n-- Executor: running generated code ...")

    # Capture stdout from the generated code's print() calls
    captured_output = io.StringIO()

    # Build a custom print that writes to our buffer
    def sandboxed_print(*args, **kwargs):
        kwargs["file"] = captured_output
        print(*args, **kwargs)

    safe_globals = {
        "pd": pd,
        "__builtins__": {**SAFE_BUILTINS, "print": sandboxed_print},
    }

    try:
        # Execute the function definition
        exec(code, safe_globals)

        # Verify clean_data was defined
        if "clean_data" not in safe_globals:
            raise RuntimeError(
                "The generated code did not define a 'clean_data' function. "
                "Expected: def clean_data(df): ..."
            )

        # Call clean_data with a copy of the raw DataFrame
        clean_df = safe_globals["clean_data"](raw_df.copy())
        original_rows = len(raw_df)

        # Validate return type
        if not isinstance(clean_df, pd.DataFrame):
            raise RuntimeError(
                f"clean_data() returned {type(clean_df).__name__} instead of DataFrame"
            )

        # Guard against total row loss — treat 0 rows as failure
        if len(clean_df) == 0:
            raise RuntimeError(
                f"clean_data() returned 0 rows (from {original_rows} input rows). "
                "Dropping all rows is NEVER acceptable. "
                "Use imputation (fillna) instead of dropna, or use subset with dropna."
            )

        # Guard against catastrophic row loss (>50% rows dropped)
        if original_rows > 0 and len(clean_df) < original_rows * 0.5:
            print(f"   [WARN] clean_data() dropped {original_rows - len(clean_df)} of {original_rows} rows")

        # Extract transformation log from captured stdout
        log_text = captured_output.getvalue()
        transformation_log = [
            line.strip() for line in log_text.strip().split("\n") if line.strip()
        ]

        print(f"   [OK] Success - {len(clean_df)} rows, {len(transformation_log)} transformations logged")

        return {
            "execution_result": {
                "success": True,
                "clean_df": clean_df,
                "error": "",
            },
            "transformation_log": transformation_log,
        }

    except Exception:
        error_tb = traceback.format_exc()
        print(f"   [X] Execution failed (retry {retry_count + 1})")
        print(f"   Error: {error_tb.splitlines()[-1]}")

        return {
            "execution_result": {
                "success": False,
                "clean_df": None,
                "error": error_tb,
            },
            "retry_count": retry_count + 1,
            "last_error": error_tb,
        }
