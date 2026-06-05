"""
Structured forecast output schema for downstream Strategy / Marketing agents.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class ForecastEvaluation(BaseModel):
    mae: Optional[float] = None
    rmse: Optional[float] = None
    mape: Optional[float] = None


class ForecastOutput(BaseModel):
    metric: str
    current_value: float
    forecasted_value: float
    change_percent: float
    confidence_score: float
    forecast_horizon: str
    selected_model: str
    model_selection_reason: str = ""
    evaluation: ForecastEvaluation = Field(default_factory=ForecastEvaluation)
    key_drivers: list[str] = Field(default_factory=list)
    business_impact: str = "medium"
    business_summary: str = ""
    trend: str = "stable"
    risks: list[str] = Field(default_factory=list)
    opportunities: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    horizons: dict[str, float] = Field(default_factory=dict)
    # Back-test leaderboard ({model, mae, rmse, mape, ...}) and relative skill
    # vs. the naive baseline (0.42 = 42% lower error). Populated by the engine.
    cv_results: list[dict[str, Any]] = Field(default_factory=list)
    skill_score: Optional[float] = None

    def to_agent_dict(self) -> dict[str, Any]:
        return self.model_dump()
