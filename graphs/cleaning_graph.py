"""
LangGraph StateGraph for the cleaning stage only.

Coder → Executor → (retry?) → Validator → END

Expects raw_df, metadata, and cleaning_plan to already be present in state.
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END

from core.state import GraphState
from agents.cleaning.coder import coder_node
from agents.cleaning.executor import executor_node
from agents.cleaning.validator import validator_node


# ── Routing logic ──────────────────────────────────────────────────────

MAX_RETRIES = 2


def _route_after_executor(state: GraphState) -> str:
    result = state.get("execution_result", {})
    retry_count = state.get("retry_count", 0)

    if result.get("success", False):
        return "validator"
    elif retry_count < MAX_RETRIES:
        return "coder"
    else:
        return "validator"


# ── Graph builder ──────────────────────────────────────────────────────


def build_cleaning_graph():
    """
    Construct and compile the cleaning-only StateGraph.

    Returns:
        A compiled LangGraph runnable.
    """
    graph = StateGraph(GraphState)

    graph.add_node("coder", coder_node)
    graph.add_node("executor", executor_node)
    graph.add_node("validator", validator_node)

    graph.set_entry_point("coder")
    graph.add_edge("coder", "executor")

    graph.add_conditional_edges(
        "executor",
        _route_after_executor,
        {
            "coder": "coder",
            "validator": "validator",
        },
    )

    graph.add_edge("validator", END)

    return graph.compile()
