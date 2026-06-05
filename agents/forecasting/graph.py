from __future__ import annotations

from langgraph.graph import StateGraph, END

from agents.forecasting.forecast_models import ForecastState
from agents.forecasting.nodes import forecast_node, prepare_node, validate_node


def build_graph() -> StateGraph:
    builder = StateGraph(ForecastState)

    builder.add_node("validate", validate_node)
    builder.add_node("prepare", prepare_node)
    builder.add_node("forecast", forecast_node)

    builder.set_entry_point("validate")
    builder.add_edge("validate", "prepare")
    builder.add_edge("prepare", "forecast")
    builder.add_edge("forecast", END)

    return builder.compile()


forecast_graph = build_graph()
