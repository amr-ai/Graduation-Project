"""
Node 2 — Cleaning Planner (LLM via Groq).

Reads the DatasetProfile JSON and produces a numbered, human-readable
cleaning plan. The plan is precise enough for the Coder agent to
implement directly, yet understandable by a non-technical business owner.
"""

from __future__ import annotations

import os
from pathlib import Path

from groq import Groq

from core.state import GraphState
from models.profiler_models import DatasetProfile

# Resolve prompt path relative to this file's location
_PROMPT_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"


def _load_prompt(filename: str) -> str:
    """Load a system prompt from the prompts/ folder."""
    path = _PROMPT_DIR / filename
    return path.read_text(encoding="utf-8")


def planner_node(state: GraphState) -> dict:
    """
    Generate a structured cleaning plan from the dataset profile.

    Returns:
        dict with 'cleaning_plan' key containing the numbered plan text.
    """
    print("\n-- Planner: generating cleaning plan ...")

    system_prompt = _load_prompt("planner_prompt.txt")

    # Reconstruct the DatasetProfile to get validated JSON
    profile = DatasetProfile(**state["metadata"])
    user_message = (
        "Here is the DatasetProfile for the dataset that needs cleaning:\n\n"
        f"{profile.model_dump_json(indent=2)}"
    )

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    from agents.constants import DEFAULT_PLANNER_MODEL
    model = state.get("planner_model", DEFAULT_PLANNER_MODEL)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
        max_tokens=2048,
    )

    plan = response.choices[0].message.content.strip()
    print(f"   -> Plan generated ({plan.count(chr(10)) + 1} lines)")

    return {"cleaning_plan": plan}
