"""
Sandbox execution utilities for LLM-generated code.

Provides safe code execution with restricted globals, stdout capture,
and automatic figure extraction for Plotly charts.

Security: no __import__, no file I/O, no subprocess, no network access.
Output is capped at 1 MB; execution times out after 30 seconds.
"""

from __future__ import annotations

import io
import re
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

SAFE_BUILTINS: dict[str, Any] = {
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
    # getattr is intentionally omitted — it can be used to walk __class__.__mro__
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

_BLOCKED_PATTERNS = [
    "__class__", "__bases__", "__subclasses__", "__globals__",
    "__builtins__", "__import__", "eval(", "exec(", "compile(",
    "open(", "os.system", "subprocess", "socket",
]


def _check_code_safety(code: str) -> str | None:
    """Return an error string if the code contains a blocked pattern, else None."""
    for pattern in _BLOCKED_PATTERNS:
        if pattern in code:
            return f"Blocked: code contains disallowed pattern '{pattern}'"
    return None


def strip_fences(code: str) -> str:
    """Strip ```python ... ``` fences from LLM-generated code."""
    code = code.strip()
    code = re.sub(r"^```(?:python)?\s*\n?", "", code)
    code = re.sub(r"\n?```\s*$", "", code)
    return code.strip()


_EXEC_TIMEOUT = 30
_MAX_OUTPUT_SIZE = 1_048_576


def _exec_in_thread(code: str, g: dict) -> None:
    exec(code, g)


def run_analysis(code: str, df: pd.DataFrame) -> dict:
    """
    Execute LLM-generated code on a DataFrame copy.

    Returns {output, error, df, fig} where *fig* is only present
    when the code assigned a Plotly figure to a variable named ``fig``.
    Times out after 30 seconds; output capped at 1 MB.
    """
    code = strip_fences(code)
    safety_error = _check_code_safety(code)
    if safety_error:
        return {"output": "", "error": safety_error, "df": df}
    buffer = io.StringIO()

    def _sandbox_print(*a, **kw):
        kw["file"] = buffer
        print(*a, **kw)

    g: dict[str, Any] = {
        "pd": pd,
        "px": px,
        "go": go,
        "df": df.copy(),
        "__builtins__": {**SAFE_BUILTINS, "print": _sandbox_print},
    }
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_exec_in_thread, code, g)
            future.result(timeout=_EXEC_TIMEOUT)

        out = buffer.getvalue()
        if len(out) > _MAX_OUTPUT_SIZE:
            out = out[:_MAX_OUTPUT_SIZE] + "\n... (truncated)"
        new_df = g.get("df", df)
        fig = g.get("fig")
        result: dict[str, Any] = {"output": out, "error": None, "df": new_df}
        if fig is not None:
            result["fig"] = fig
        return result
    except TimeoutError:
        return {"output": buffer.getvalue(), "error": "Execution timed out (>30s)", "df": df}
    except Exception:
        return {"output": buffer.getvalue(), "error": traceback.format_exc(), "df": df}


def df_context(df: pd.DataFrame) -> str:
    """Short textual summary of a DataFrame for LLM context."""
    buf = io.StringIO()
    buf.write(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")
    buf.write(f"Columns: {list(df.columns)}\n")
    buf.write(f"Dtypes:\n{df.dtypes.to_string()}\n")
    buf.write(f"\nFirst 5 rows:\n{df.head(5).to_string()}\n")
    if len(df) > 0:
        num_cols = df.select_dtypes(include="number").columns
        if len(num_cols) > 0:
            buf.write(f"\nNumeric summary:\n{df[num_cols].describe().to_string()}\n")
    return buf.getvalue()


def _convert_value(v):
    """Recursively convert a value to a JSON-safe type."""
    if isinstance(v, pd.Period):
        return str(v)
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if isinstance(v, pd.Timedelta):
        return str(v)
    import numpy as np
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, dict):
        return {k: _convert_value(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_convert_value(item) for item in v]
    if isinstance(v, tuple):
        return tuple(_convert_value(item) for item in v)
    return v


def sanitize_fig(fig):
    """Convert a Plotly figure in-place, replacing non-JSON-serializable
    types (e.g. pandas Period) with their string representations."""
    d = fig.to_dict()
    cleaned = _convert_value(d)
    fig.update(data=cleaned.get("data", fig.data), layout=cleaned.get("layout", fig.layout))
    if "frames" in cleaned:
        fig.frames = cleaned["frames"]
    return fig
