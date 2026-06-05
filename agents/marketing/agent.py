"""
Marketing Agent — LLM layer.

Turns the deterministic marketing analytics payload into:
  1. A markdown marketing strategy report   -> generate_marketing_strategy()
  2. A structured campaign plan (JSON)       -> generate_campaign_plan()
  3. Ready-to-ship ad copy (JSON)            -> generate_ad_copy()

All three are grounded in the pre-computed RFM segments / KPIs so the model
does not invent numbers.  Prompts live in ``prompts/marketing/``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from groq import Groq

from agents.constants import DEFAULT_MARKETING_MODEL
from tools.db_tools import get_schema_info
from tools.sandbox import df_context

_PROMPT_DIR = Path(__file__).resolve().parent.parent.parent / "prompts" / "marketing"


def _load_prompt(stem: str) -> str:
    return (_PROMPT_DIR / f"{stem}.md").read_text(encoding="utf-8")


def _client() -> Groq:
    return Groq(api_key=os.environ.get("GROQ_API_KEY") or "")


def _model() -> str:
    return st.session_state.get("marketing_model", DEFAULT_MARKETING_MODEL)


def _slim_forecast(forecast_results: dict | None) -> list[dict]:
    """Extract just the headline forecast numbers for the prompt context."""
    if not forecast_results:
        return []
    out = []
    for o in forecast_results.get("forecast_outputs", [])[:6]:
        out.append({
            "metric": o.get("metric"),
            "forecasted_value": o.get("forecasted_value"),
            "change_percent": o.get("change_percent"),
            "forecast_horizon": o.get("forecast_horizon"),
            "business_impact": o.get("business_impact"),
        })
    return out


def _context_block(
    business_context: str,
    objective: str,
    budget: str,
    channels: list[str] | None,
    brand_voice: str,
) -> str:
    parts = []
    if objective:
        parts.append(f"Primary objective: {objective}")
    if budget:
        parts.append(f"Marketing budget: {budget}")
    if channels:
        parts.append(f"Allowed channels: {', '.join(channels)}")
    if brand_voice:
        parts.append(f"Brand voice: {brand_voice}")
    if business_context:
        parts.append(f"Business context: {business_context}")
    return "\n".join(parts)


# ── 1. Strategy report (markdown) ──────────────────────────────────────────

def generate_marketing_strategy(
    data_df: pd.DataFrame | None,
    table_name: str,
    marketing_payload: dict,
    business_context: str = "",
    objective: str = "",
    budget: str = "",
    channels: list[str] | None = None,
    brand_voice: str = "",
    forecast_results: dict | None = None,
) -> str:
    """Generate a markdown marketing strategy report grounded in the analytics."""
    if data_df is not None:
        schema = df_context(data_df)
    else:
        schema = get_schema_info(table_name)

    system_prompt = _load_prompt("strategy")
    context = _context_block(business_context, objective, budget, channels, brand_voice)
    forecast = _slim_forecast(forecast_results)

    parts = [
        "Dataset overview:\n",
        f"{schema}\n\n",
        "Pre-computed marketing analytics (use as the foundation; do not invent numbers):\n",
        f"```json\n{json.dumps(marketing_payload, indent=2, default=str)}\n```\n\n",
    ]
    if forecast:
        parts.append(
            "Sales forecast highlights from the Forecasting agent:\n"
            f"```json\n{json.dumps(forecast, indent=2, default=str)}\n```\n\n"
        )
    if context:
        parts.append(f"User-provided context:\n{context}\n\n")
    parts.append("Write the marketing strategy following the required structure.")

    reply = _client().chat.completions.create(
        model=_model(),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "".join(parts)},
        ],
        temperature=0.4,
        max_tokens=4096,
    )
    return reply.choices[0].message.content.strip()


# ── 2. Campaign plan (structured JSON) ─────────────────────────────────────

def generate_campaign_plan(
    marketing_payload: dict,
    business_context: str = "",
    objective: str = "",
    budget: str = "",
    channels: list[str] | None = None,
    brand_voice: str = "",
    forecast_results: dict | None = None,
) -> dict:
    """Generate a structured campaign plan (dict with a ``campaigns`` list)."""
    system_prompt = _load_prompt("campaigns")
    context = _context_block(business_context, objective, budget, channels, brand_voice)

    # Only the segment + KPI parts of the payload are needed for planning.
    planning_ctx = {
        "rfm": marketing_payload.get("rfm", {}),
        "marketing_kpis": marketing_payload.get("marketing_kpis", {}),
        "channels": marketing_payload.get("channels", {}),
        "forecast": _slim_forecast(forecast_results),
        "user_context": context,
    }

    try:
        reply = _client().chat.completions.create(
            model=_model(),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(planning_ctx, indent=2, default=str)},
            ],
            temperature=0.4,
            max_tokens=2048,
            response_format={"type": "json_object"},
        )
        data = json.loads(reply.choices[0].message.content.strip())
        if "campaigns" not in data:
            data = {"campaigns": data if isinstance(data, list) else [], "summary": ""}
        return data
    except Exception as exc:
        return {"campaigns": [], "summary": "", "error": str(exc)}


# ── 3. Ad copy (structured JSON) ───────────────────────────────────────────

def generate_ad_copy(
    segment: str,
    channel: str,
    platform: str,
    offer: str,
    brand_voice: str = "",
    segment_stats: dict | None = None,
    playbook: str = "",
) -> dict:
    """Generate ready-to-ship copy for a segment + channel + platform."""
    system_prompt = _load_prompt("ad_copy")

    brief = {
        "target_segment": segment,
        "segment_playbook": playbook,
        "segment_stats": segment_stats or {},
        "channel": channel,
        "platform": platform,
        "offer": offer,
        "brand_voice": brand_voice or "friendly, trustworthy, concise",
    }

    try:
        reply = _client().chat.completions.create(
            model=_model(),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(brief, indent=2, default=str)},
            ],
            temperature=0.7,
            max_tokens=1536,
            response_format={"type": "json_object"},
        )
        return json.loads(reply.choices[0].message.content.strip())
    except Exception as exc:
        return {"error": str(exc)}
