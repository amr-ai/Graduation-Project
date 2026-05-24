"""
Dashboard auto-generation for the visualization agent.

Analyses data for KPIs first, then generates 5-7 diverse Plotly charts
with automatic retry — the LLM receives its own error message and
regenerates fixed code on failure.
"""

from __future__ import annotations

import io
import os
import re
import traceback

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from groq import Groq

from tools.db_tools import get_schema_info, load_df_from_pg
from tools.sandbox import SAFE_BUILTINS, df_context


def _resolve_table(table_name: str) -> str:
    """Try the given table name; fall back to lowercase if not found."""
    try:
        get_schema_info(table_name)
        return table_name
    except Exception:
        lowered = table_name.lower()
        if lowered != table_name:
            try:
                get_schema_info(lowered)
                return lowered
            except Exception:
                pass
        return table_name


def generate_dashboard(
    data_df: pd.DataFrame | None, table_name: str
) -> list[dict]:
    """
    Auto-generate 5-7 diverse charts based on discovered KPIs.

    Parameters
    ----------
    data_df : DataFrame or None
        In-memory data.  When *None* the function loads from *table_name*.
    table_name : str
        PostgreSQL table name (used only when *data_df* is None).

    Returns
    -------
    list[dict]
        Each entry has keys ``query`` (KPI / chart title), ``fig`` (Plotly
        Figure), and ``code`` (the full generated Python block).
    """
    if data_df is not None:
        schema = df_context(data_df)
        src_df = data_df
    else:
        table_name = _resolve_table(table_name)
        schema = get_schema_info(table_name)
        src_df = load_df_from_pg(table_name)

    client = Groq(api_key=os.environ.get("GROQ_API_KEY") or "")
    model = st.session_state.get("model", "llama-3.3-70b-versatile")

    base_prompt = (
        "You are a data visualization expert. Your job has two steps.\n\n"
        "STEP 1 — Identify KPIs\n"
        "Analyse the data schema below and identify the most important "
        "key performance indicators (KPIs) and relationships. Consider:\n"
        "- Which columns are metrics (numeric aggregations like totals, averages, counts)\n"
        "- Which columns are dimensions (categories, dates, segments to slice by)\n"
        "- What trends, distributions, or comparisons would be most insightful\n\n"
        "STEP 2 — Build charts\n"
        "Create 5 to 7 diverse charts that visualise the KPIs you identified. "
        "Choose chart types based on the analysis goal:\n"
        "- Categorical comparison -> Bar chart\n"
        "- Time series / trend     -> Line chart\n"
        "- Percentage / share      -> Pie or Donut chart\n"
        "- Correlation (2 vars)    -> Scatter plot\n"
        "- Distribution            -> Histogram\n"
        "- Correlation (many vars) -> Heatmap (annotated)\n"
        "- Statistical spread      -> Box plot\n\n"
        "Assign each chart to fig1, fig2, fig3, fig4, fig5, fig6, fig7. "
        "Every figure MUST have a descriptive title that mentions the KPI being shown.\n\n"
        f"Data schema:\n{schema}\n\n"
        "IMPORTANT:\n"
        "- Do NOT use import statements. "
        "pandas as pd, plotly.express as px, and plotly.graph_objects as go are already loaded.\n"
        "- The data is in a DataFrame called `df`.\n"
        "- Use EXACT column names as shown in the schema (case-sensitive, spaces matter).\n"
        "- For px.pie(), use names= for the category column and values= for the numeric column.\n"
        "- Wrap ALL code in a single ```python ... ``` block."
    )

    messages = [{"role": "system", "content": base_prompt}]

    for attempt in range(3):
        reply = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=4096,
        )
        raw = reply.choices[0].message.content.strip()

        code_match = re.search(r"```python\n(.*?)```", raw, re.DOTALL)
        code = code_match.group(1).strip() if code_match else raw

        buffer = io.StringIO()

        def _sandbox_print(*a, **kw):
            kw["file"] = buffer
            print(*a, **kw)

        g = {
            "pd": pd,
            "px": px,
            "go": go,
            "df": src_df.copy(),
            "__builtins__": {
                **SAFE_BUILTINS, "print": _sandbox_print, "__import__": __import__,
            },
        }

        try:
            exec(code, g)
        except Exception:
            tb = traceback.format_exc()
            if attempt < 2:
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": (
                        "The code above failed with this error:\n"
                        f"```\n{tb}\n```\n"
                        "Fix the issue and regenerate ALL charts. "
                        "Ensure column names match exactly and all imports are omitted."
                    ),
                })
                continue
            st.error(f"Dashboard generation failed after 3 attempts:\n```\n{tb}\n```")
            return []

        charts = []
        i = 1
        while g.get(f"fig{i}") is not None:
            fig = g[f"fig{i}"]
            title = fig.layout.title.text if fig.layout.title else f"Chart {i}"
            charts.append({"query": title, "fig": fig, "code": code})
            i += 1

        if not charts and g.get("fig"):
            fig = g["fig"]
            title = fig.layout.title.text if fig.layout.title else "Chart"
            charts.append({"query": title, "fig": fig, "code": code})

        return charts

    return []
