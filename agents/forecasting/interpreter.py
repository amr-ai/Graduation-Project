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


def interpret_node(state: ForecastState) -> dict:
    """Enrich the latest forecast output with LLM business narrative."""
    idx = state["current_target_index"]
    outputs = list(state.get("forecast_outputs", []))
    interpretations = list(state.get("interpretations", []))

    if idx >= len(outputs):
        return {}

    record = dict(outputs[idx])
    api_key = os.environ.get("GROQ_API_KEY")
    from agents.constants import DEFAULT_FORECAST_MODEL
    model = DEFAULT_FORECAST_MODEL

    if api_key:
        try:
            client = Groq(api_key=api_key)
            ctx = {k: record.get(k) for k in (
                "metric", "current_value", "forecasted_value", "change_percent",
                "confidence_score", "selected_model", "evaluation", "key_drivers",
                "business_impact", "trend", "horizons",
            )}
            if state.get("business_context"):
                ctx["business_context"] = state["business_context"]

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
            record["business_summary"] = llm.get("business_summary", "")
            record["risks"] = llm.get("risks", [])
            record["opportunities"] = llm.get("opportunities", [])
            record["recommended_actions"] = llm.get("recommended_actions", [])
        except Exception:
            pass

    if not record.get("business_summary"):
        m = record.get("metric", "Metric")
        chg = record.get("change_percent", 0)
        record["business_summary"] = (
            f"{m.replace('_', ' ').title()} is projected to change by {chg:+.1f}% "
            f"over {record.get('forecast_horizon', '30_days')}. "
            f"Model: {record.get('selected_model', 'n/a')}."
        )
        record.setdefault("risks", ["Market shifts may alter actual results"])
        record.setdefault("opportunities", ["Use forecast for inventory and campaign planning"])
        record.setdefault("recommended_actions", ["Compare actuals to forecast weekly"])

    outputs[idx] = record
    interpretations.append({
        "executive_summary": record.get("business_summary", ""),
        "risks": record.get("risks", []),
        "opportunities": record.get("opportunities", []),
        "recommended_actions": record.get("recommended_actions", []),
        "confidence_statement": (
            f"Confidence {record.get('confidence_score', 0):.0%} based on backtest error."
        ),
    })

    return {"forecast_outputs": outputs, "interpretations": interpretations}
