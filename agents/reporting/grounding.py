"""
Report grounding / faithfulness check.

Every LLM-written report in Smart Analyst is *grounded*: the deterministic
engines (KPIs, RFM, forecasts, churn) compute the numbers and the model only
narrates them.  This module verifies that promise by extracting every material
figure from a generated report and checking it against the numeric values that
actually appear in the deterministic payload(s).

It is intentionally a *heuristic* guard, not a proof — it catches the common
failure mode where a model quietly invents or mis-states a figure.  The result
is surfaced in the UI ("14 of 14 figures traced to your data") and embedded in
the exported report, giving the output a measurable trust signal.

Pure standard-library; no Streamlit / pandas import so it stays unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import floor, log10

import re

# ── Number extraction ──────────────────────────────────────────────────────

# Matches an optional currency symbol, a number (with thousands separators or
# plain), an optional magnitude suffix (K/M/B) and an optional trailing percent.
_NUM_RE = re.compile(
    r"(?P<cur>[$€£])?\s*"
    r"(?P<num>-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?)"
    r"\s*(?P<suffix>[KkMmBb])?"
    r"\s*(?P<pct>%)?"
)

_SUFFIX = {"k": 1e3, "m": 1e6, "b": 1e9}

# Plain numbers below this magnitude are treated as structural (list counts,
# ordinals, "next 30 days", horizons) rather than data claims, so they are not
# counted for/against grounding.  Currency and percentages are always counted.
_PLAIN_MIN_MAGNITUDE = 1000.0


@dataclass
class NumericClaim:
    """A single numeric figure asserted in the report text."""

    raw: str
    value: float
    kind: str  # "currency" | "percent" | "number"
    verified: bool = False


@dataclass
class GroundingResult:
    """Outcome of checking a report's figures against the source payload."""

    claims: list[NumericClaim] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.claims)

    @property
    def verified_count(self) -> int:
        return sum(1 for c in self.claims if c.verified)

    @property
    def unverified(self) -> list[NumericClaim]:
        return [c for c in self.claims if not c.verified]

    @property
    def coverage(self) -> float:
        """Share of material figures traced back to the data (0.0–1.0)."""
        return 1.0 if self.total == 0 else self.verified_count / self.total

    @property
    def status(self) -> str:
        """One of ``"none"``, ``"clean"``, ``"partial"``, ``"weak"``."""
        if self.total == 0:
            return "none"
        if self.verified_count == self.total:
            return "clean"
        return "partial" if self.coverage >= 0.8 else "weak"

    @property
    def label(self) -> str:
        """Human-readable one-line summary for the UI / export footer."""
        if self.total == 0:
            return "No specific figures to verify in this report."
        if self.status == "clean":
            return (
                f"All {self.total} figures in this report trace back to your data."
            )
        return (
            f"{self.verified_count} of {self.total} figures traced to your data "
            f"({len(self.unverified)} could not be matched)."
        )


# ── Helpers ─────────────────────────────────────────────────────────────────

def _to_float(token: str) -> float | None:
    try:
        return float(token.replace(",", ""))
    except (TypeError, ValueError):
        return None


def _sig_round(x: float, sig: int) -> float:
    """Round *x* to *sig* significant figures (handles the way models round)."""
    if x == 0:
        return 0.0
    return round(x, -int(floor(log10(abs(x)))) + (sig - 1))


def _close(a: float, b: float, rel: float, abs_tol: float) -> bool:
    return abs(a - b) <= max(abs_tol, rel * max(abs(a), abs(b)))


def _matches(claim: float, known: float, rel: float = 0.02, abs_tol: float = 0.5) -> bool:
    """True if *claim* equals *known* within tolerance or a sensible rounding."""
    if _close(claim, known, rel, abs_tol):
        return True
    # Accept aggressive-but-honest rounding, e.g. 12,345 quoted as "$12,000".
    for sig in (1, 2, 3):
        if abs(claim - _sig_round(known, sig)) <= 1e-9:
            return True
    return False


def collect_known_values(payload: object, _budget: int = 20000) -> list[float]:
    """Recursively pull every numeric leaf out of a payload (dict/list/scalar)."""
    out: list[float] = []

    def walk(node: object) -> None:
        if len(out) >= _budget:
            return
        if isinstance(node, bool):
            return  # bool is an int subclass — never a data figure
        if isinstance(node, (int, float)):
            out.append(float(node))
        elif isinstance(node, str):
            v = _to_float(node.strip())
            if v is not None:
                out.append(v)
        elif isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, (list, tuple, set)):
            for v in node:
                walk(v)

    walk(payload)
    return out


def extract_claims(text: str) -> list[NumericClaim]:
    """Extract the material numeric figures asserted in *text* (deduplicated)."""
    claims: dict[tuple[str, float], NumericClaim] = {}

    for m in _NUM_RE.finditer(text or ""):
        base = _to_float(m.group("num"))
        if base is None:
            continue
        value = base
        suffix = (m.group("suffix") or "").lower()
        if suffix:
            value *= _SUFFIX[suffix]

        if m.group("cur"):
            kind = "currency"
        elif m.group("pct"):
            kind = "percent"
        else:
            kind = "number"

        # Skip structural / non-data numbers to avoid noisy false flags.
        if kind == "number":
            if abs(value) < _PLAIN_MIN_MAGNITUDE:
                continue
            if value == int(value) and 1900 <= value <= 2100:
                continue  # calendar year

        key = (kind, round(value, 4))
        if key not in claims:
            claims[key] = NumericClaim(raw=m.group(0).strip(), value=value, kind=kind)

    return list(claims.values())


def _is_verified(claim: NumericClaim, known: list[float]) -> bool:
    v = claim.value
    if claim.kind == "percent":
        # Payload may store a percent either as 23.4 or as the fraction 0.234.
        for k in known:
            if _matches(v, k) or _matches(v, k * 100.0):
                return True
        return False
    for k in known:
        if _matches(v, k):
            return True
    return False


def check_grounding(report_text: str, *payloads: object) -> GroundingResult:
    """
    Verify a report's figures against one or more deterministic payloads.

    Parameters
    ----------
    report_text : str
        The generated report (markdown / plain text).
    *payloads : object
        Any number of analytics payloads (dicts, lists, scalars).  Their
        numeric leaves are unioned into the reference set.

    Returns
    -------
    GroundingResult
    """
    known: list[float] = []
    for p in payloads:
        if p is not None:
            known.extend(collect_known_values(p))

    result = GroundingResult()
    for claim in extract_claims(report_text):
        claim.verified = bool(known) and _is_verified(claim, known)
        result.claims.append(claim)
    return result
