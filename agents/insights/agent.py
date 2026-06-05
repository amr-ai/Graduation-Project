"""
Insights Agent — generates a business report using a template selected
based on the data profile and business context.

The old hardcoded single prompt has been replaced by a template system in
``prompts/insights/``.  The template selector picks the best template for
the audience (default: non-technical plain-language report).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from groq import Groq

from agents.constants import DEFAULT_INSIGHTS_MODEL
from tools.db_tools import get_schema_info
from tools.sandbox import df_context

from .template_selector import select_template, describe_template

_PROMPT_DIR = Path(__file__).resolve().parent.parent.parent / "prompts" / "insights"

_TEMPLATES = {
    "non_technical": "non_technical.md",
    "executive": "executive.md",
    "detailed": "detailed.md",
}


def _load_template(stem: str) -> str:
    """Load a template file from the insights prompts directory."""
    path = _PROMPT_DIR / _TEMPLATES.get(stem, "non_technical.md")
    return path.read_text(encoding="utf-8")


def generate_insights(
    data_df: pd.DataFrame | None,
    table_name: str,
    business_context: str = "",
    analytics_payload: dict | None = None,
) -> str:
    """
    Generate a structured business insights report from the data.

    The template is auto-selected by ``template_selector.select_template()``
    based on the business context and analytics payload.

    Parameters
    ----------
    data_df : DataFrame or None
        In-memory cleaned data.  When *None* loads from *table_name*.
    table_name : str
        PostgreSQL table name (used only when *data_df* is None).
    business_context : str, optional
        Industry, goals, and other context provided by the user.
    analytics_payload : dict, optional
        Pre-computed KPIs, time series, and quality data from the
        analytics engine.

    Returns
    -------
    str
        Markdown-formatted business report.
    """
    if data_df is not None:
        schema = df_context(data_df)
    else:
        schema = get_schema_info(table_name)

    # ── Auto-select the best template ──────────────────────────────────
    template_stem = select_template(
        business_context=business_context,
        analytics_payload=analytics_payload,
    )
    system_prompt = _load_template(template_stem)

    # Store the choice in session state so the UI can show it
    st.session_state["selected_template"] = template_stem
    st.session_state["selected_template_label"] = describe_template(template_stem)

    client = Groq(api_key=os.environ.get("GROQ_API_KEY") or "")
    model = st.session_state.get("insights_model", DEFAULT_INSIGHTS_MODEL)

    parts = [
        "Here is the cleaned dataset I need you to analyse:\n\n",
        f"{schema}\n\n",
    ]

    if business_context:
        parts.append(
            f"Business context provided by the user:\n{business_context}\n\n"
        )

    if analytics_payload:
        parts.append(
            "Pre-computed analytics payload (use this as the foundation for your analysis):\n"
            f"```json\n{json.dumps(analytics_payload, indent=2, default=str)}\n```\n\n"
        )

    parts.append(
        "Generate the report following the required structure. "
        "If any business context is missing (industry, goals, etc.), "
        "mention it at the start so the user can provide it."
    )

    user_prompt = "".join(parts)

    reply = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=4096,
    )

    return reply.choices[0].message.content.strip()
