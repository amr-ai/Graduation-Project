"""
Step 8 — Business interpretation via LLM only (no forecasting math).
"""

from __future__ import annotations

import json
import os

from groq import Groq

from agents.forecasting.forecast_models import ForecastState

_SYSTEM = """You are a senior e-commerce analyst. You receive structured forecast JSON.
Write business language only — do NOT invent numbers not in the JSON.
Return valid JSON:
{
  "business_summary": "2-3 sentences for executives",
  "risks": ["2-4 items"],
  "opportunities": ["2-4 items"],
  "recommended_actions": ["2-4 concrete actions"]
}
"""


def _interpret_one(record: dict, business_context: str, client: "Groq", model: str) -> dict:
    """Enrich a single forecast record with LLM narrative. Returns the enriched record."""
    ctx = {k: record.get(k) for k in (
        "metric", "current_value", "forecasted_value", "change_percent",
        "confidence_score", "selected_model", "evaluation", "key_drivers",
        "business_impact", "trend", "horizons",
    )}
    if business_context:
        ctx["business_context"] = business_context
    try:
        reply = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": json.dumps(ctx, indent=2)},
            ],
            temperature=0.3,
            max_tokens=1024,
            response_format={"type": "json_object"},
        )
        llm = json.loads(reply.choices[0].message.content.strip())
        record = {**record,
                  "business_summary": llm.get("business_summary", ""),
                  "risks": llm.get("risks", []),
                  "opportunities": llm.get("opportunities", []),
                  "recommended_actions": llm.get("recommended_actions", [])}
    except Exception:
        pass
    return record


def _fallback(record: dict) -> dict:
    m = record.get("metric", "Metric")
    chg = record.get("change_percent", 0)
    return {
        **record,
        "business_summary": (
            f"{m.replace('_', ' ').title()} is projected to change by {chg:+.1f}% "
            f"over {record.get('forecast_horizon', '30_days')}. "
            f"Model: {record.get('selected_model', 'n/a')}."
        ),
        "risks": record.get("risks") or ["Market shifts may alter actual results"],
        "opportunities": record.get("opportunities") or ["Use forecast for inventory and campaign planning"],
        "recommended_actions": record.get("recommended_actions") or ["Compare actuals to forecast weekly"],
    }


def interpret_node(state: ForecastState) -> dict:
    """Enrich all forecast outputs with LLM business narrative (batch)."""
    outputs = list(state.get("forecast_outputs", []))
    if not outputs:
        return {}

    api_key = os.environ.get("GROQ_API_KEY")
    from agents.constants import DEFAULT_FORECAST_MODEL
    model = state.get("model") or DEFAULT_FORECAST_MODEL
    business_context = state.get("business_context", "")

    client = Groq(api_key=api_key) if api_key else None
    enriched: list[dict] = []
    interpretations: list[dict] = []

    for record in outputs:
        record = dict(record)
        if client:
            record = _interpret_one(record, business_context, client, model)
        if not record.get("business_summary"):
            record = _fallback(record)
        enriched.append(record)
        interpretations.append({
            "metric": record.get("metric", ""),
            "executive_summary": record.get("business_summary", ""),
            "risks": record.get("risks", []),
            "opportunities": record.get("opportunities", []),
            "recommended_actions": record.get("recommended_actions", []),
            "confidence_statement": (
                f"Confidence {record.get('confidence_score', 0):.0%} based on backtest error."
            ),
        })

    return {"forecast_outputs": enriched, "interpretations": interpretations}
