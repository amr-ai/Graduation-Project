"""
Node 3 — Code Generator (LLM via Groq).

Generates a self-contained `clean_data(df)` Python function based on
the cleaning plan and dataset profile. On retry, switches to the repair
prompt and includes the previous code + error traceback so the LLM can
self-correct.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from groq import Groq

from core.state import GraphState
from models.profiler_models import DatasetProfile

_PROMPT_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"


def _load_prompt(filename: str) -> str:
    """Load a system prompt from the prompts/ folder."""
    path = _PROMPT_DIR / filename
    return path.read_text(encoding="utf-8")


def _strip_markdown_fences(code: str) -> str:
    """Remove ```python ... ``` fences if the LLM wrapped its output."""
    code = code.strip()
    # Remove opening fence
    code = re.sub(r"^```(?:python)?\s*\n?", "", code)
    # Remove closing fence
    code = re.sub(r"\n?```\s*$", "", code)
    return code.strip()


def coder_node(state: GraphState) -> dict:
    """
    Generate (or repair) the clean_data(df) function.

    Returns:
        dict with 'generated_code' key containing the Python function string.
    """
    retry_count = state.get("retry_count", 0)

    if retry_count == 0:
        print("\n-- Coder: generating cleaning code ...")
        system_prompt = _load_prompt("coder_prompt.txt")

        profile = DatasetProfile(**state["metadata"])
        user_message = (
            "## Cleaning Plan\n\n"
            f"{state['cleaning_plan']}\n\n"
            "## Dataset Profile\n\n"
            f"{profile.model_dump_json(indent=2)}"
        )
    else:
        print(f"\n-- Coder: repairing code (attempt {retry_count + 1}) ...")
        system_prompt = _load_prompt("repair_prompt.txt")

        user_message = (
            "## Previous Code (FAILED)\n\n"
            f"```python\n{state['generated_code']}\n```\n\n"
            "## Error Traceback\n\n"
            f"```\n{state['last_error']}\n```\n\n"
            "## Cleaning Plan\n\n"
            f"{state['cleaning_plan']}"
        )

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    from agents.constants import DEFAULT_CODER_MODEL
    model = state.get("coder_model", DEFAULT_CODER_MODEL)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,
        max_tokens=4096,
    )

    raw_code = response.choices[0].message.content.strip()
    code = _strip_markdown_fences(raw_code)

    print(f"   -> Generated {code.count(chr(10)) + 1} lines of code")

    return {"generated_code": code}
