"""
LangGraph StateGraph for the data cleaning pipeline.

This module contains ONLY graph wiring — no business logic.
Node implementations live in agents/cleaning/*.py.

Pipeline:
    Profiler → Planner → Coder → Executor → (retry?) → Validator
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END

from core.state import GraphState
from agents.cleaning.profiler import profiler_node
from agents.cleaning.planner import planner_node
from agents.cleaning.coder import coder_node
from agents.cleaning.executor import executor_node
from agents.cleaning.validator import validator_node


# ── Routing logic ──────────────────────────────────────────────────────

MAX_RETRIES = 2


def _route_after_executor(state: GraphState) -> str:
    """
    Decide the next node after execution:
      - success → validator
      - failure + retries left → coder (retry loop)
      - failure + max retries → validator (with failure flag)
    """
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
    Construct and compile the cleaning pipeline StateGraph.

    Returns:
        A compiled LangGraph runnable.
    """
    graph = StateGraph(GraphState)

    # Register nodes
    graph.add_node("profiler", profiler_node)
    graph.add_node("planner", planner_node)
    graph.add_node("coder", coder_node)
    graph.add_node("executor", executor_node)
    graph.add_node("validator", validator_node)

    # Wire edges
    graph.set_entry_point("profiler")
    graph.add_edge("profiler", "planner")
    graph.add_edge("planner", "coder")
    graph.add_edge("coder", "executor")

    # Conditional edge: retry loop or proceed to validation
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
