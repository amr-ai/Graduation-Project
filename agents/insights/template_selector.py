"""
Template Selector — analyzes analytics payload + business context to
pick the best insight template for the audience.

Detection strategy:
  1. Check business context for explicit tone/audience keywords.
  2. Default to non_technical for users who didn't specify.
  3. Fall through to detailed only when explicitly requested.

Returns the template file stem (e.g. "non_technical", "executive", "detailed").
"""

from __future__ import annotations

import re

KEYWORD_MAP: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(board|executive|c[eé]o|cfo|leadership|director)\b", re.IGNORECASE), "executive"),
    (re.compile(r"\b(brief|concise|summary|high.level|tl;dr|snapshot)\b", re.IGNORECASE), "executive"),
    (re.compile(r"\b(detailed|technical|comprehensive|in.depth|full|deep.dive)\b", re.IGNORECASE), "detailed"),
    (re.compile(r"\b(auditor|analyst|data.team|expert)\b", re.IGNORECASE), "detailed"),
]

# Industry segments that map better to non-technical storytelling
_STORYTELLING_INDUSTRIES = {
    "fashion", "grocery", "home", "beauty", "sports", "books",
    "retail", "restaurant", "cafe", "bakery", "food",
    "general retail", "other",
}


def select_template(
    business_context: str = "",
    analytics_payload: dict | None = None,
) -> str:
    """
    Pick the best insight template.

    Parameters
    ----------
    business_context : str
        Free-text business context from the user (industry, goals, etc.).
    analytics_payload : dict or None
        Output of ``run_analytics()`` — used for heuristic fallback when
        there is no explicit audience signal.

    Returns
    -------
    str
        One of ``"non_technical"``, ``"executive"``, ``"detailed"``.
    """
    # 1. Explicit keyword match in business context
    for pattern, template in KEYWORD_MAP:
        if pattern.search(business_context):
            return template

    # 2. Industry heuristic: if industry is a "storytelling-friendly" one,
    #    stick with non_technical (the default).
    industry = _extract_industry(business_context)
    if industry and industry.lower() in _STORYTELLING_INDUSTRIES:
        return "non_technical"

    # 3. Data complexity heuristic (only when we have the analytics payload)
    if analytics_payload:
        col_count = (analytics_payload.get("metadata") or {}).get("column_count", 0)
        kpi = analytics_payload.get("kpi", {})

        # Very wide datasets with lots of KPIs → detailed might be needed
        if col_count > 30 and kpi.get("revenue") is not None:
            return "detailed"

        # Very simple datasets (few columns, few KPIs) → non_technical works well
        if col_count <= 15:
            return "non_technical"

    # 4. Safe default
    return "non_technical"


def _extract_industry(context: str) -> str:
    """Try to extract a named industry from the business context string."""
    for line in context.splitlines():
        line = line.strip()
        if line.lower().startswith("industry:"):
            return line.split(":", 1)[1].strip()
    return ""


def describe_template(template: str) -> str:
    """Return a human-readable label for the selected template."""
    descriptions = {
        "non_technical": "Plain-language story for non-technical readers",
        "executive": "Concise board-level executive briefing",
        "detailed": "Comprehensive technical BI report (10 sections)",
    }
    return descriptions.get(template, template)
