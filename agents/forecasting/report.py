"""
Deterministic assembly of a forecast narrative.

The forecasting pipeline returns structured, per-metric output (selected model,
back-test error, horizon values, and an LLM business interpretation).  This
module stitches those pieces into a single consulting-style markdown report so
the forecast can be grounded, presented and exported like every other Smart
Analyst report — without an extra LLM call.
"""

from __future__ import annotations

from typing import Any


def _fmt(value: Any, spec: str = ",.0f") -> str:
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return "—"


def _metric_title(metric: str) -> str:
    return (metric or "Metric").replace("_", " ").title()


def build_forecast_report_md(results: dict) -> str:
    """Build a markdown forecast report from a pipeline result dict."""
    outputs = results.get("forecast_outputs", []) or []
    if not outputs:
        return ""

    lines: list[str] = ["# Sales & Demand Forecast", ""]

    # ── Executive summary ──────────────────────────────────────────────
    lines.append("## Executive Summary")
    for out in outputs:
        name = _metric_title(out.get("metric", ""))
        chg = out.get("change_percent")
        horizon = out.get("forecast_horizon", "the forecast horizon")
        model = out.get("selected_model", "n/a")
        if chg is not None:
            lines.append(
                f"- **{name}** is projected to move **{chg:+.1f}%** over "
                f"{horizon} (model: {model})."
            )
        else:
            lines.append(f"- **{name}** — forecast produced with {model}.")
    lines.append("")

    # ── Per-metric detail ──────────────────────────────────────────────
    for out in outputs:
        name = _metric_title(out.get("metric", ""))
        lines.append(f"## {name}")

        current = out.get("current_value")
        forecast = out.get("forecasted_value")
        chg = out.get("change_percent")
        conf = out.get("confidence_score")
        ev = out.get("evaluation", {}) or {}
        skill = out.get("skill_score")

        detail = []
        if current is not None:
            detail.append(f"- **Current value:** {_fmt(current)}")
        if forecast is not None:
            chg_txt = f" ({chg:+.1f}%)" if chg is not None else ""
            detail.append(f"- **Forecast:** {_fmt(forecast)}{chg_txt}")
        detail.append(f"- **Selected model:** {out.get('selected_model', 'n/a')}")
        if out.get("granularity"):
            detail.append(f"- **Granularity:** {out['granularity']}")
        reason = out.get("model_selection_reason")
        if reason:
            detail.append(f"- **Why this model:** {reason}")
        bt = (
            f"- **Back-test (out-of-sample):** MAE {ev.get('mae', '—')} · "
            f"RMSE {ev.get('rmse', '—')} · MAPE {ev.get('mape', '—')}%"
        )
        if skill is not None:
            bt += f" · skill vs naive {skill * 100:+.0f}%"
        detail.append(bt)
        if conf is not None:
            detail.append(f"- **Confidence:** {conf:.0%}")
        impact = out.get("business_impact")
        if impact:
            detail.append(f"- **Business impact:** {impact}")
        lines.extend(detail)
        lines.append("")

        horizons = out.get("horizons")
        if isinstance(horizons, dict) and horizons:
            hz = " · ".join(f"{k}: {_fmt(v)}" for k, v in horizons.items())
            lines.append(f"**Horizon forecasts (mean):** {hz}")
            lines.append("")

        summary = out.get("business_summary")
        if summary:
            lines.append(f"**Summary.** {summary}")
            lines.append("")

        for label, key in (("Risks", "risks"),
                           ("Opportunities", "opportunities"),
                           ("Recommended actions", "recommended_actions")):
            items = out.get(key) or []
            if items:
                lines.append(f"**{label}**")
                lines.extend(f"- {item}" for item in items)
                lines.append("")

        lines.append("---")
        lines.append("")

    # Drop the trailing separator.
    while lines and lines[-1] in ("", "---"):
        lines.pop()

    return "\n".join(lines)
