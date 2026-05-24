from __future__ import annotations

from typing import Any, Optional, TypedDict


class VizState(TypedDict):
    user_query: str
    schema_info: str
    table_name: str
    existing_chart_count: int
    generated_code: str
    execution_result: dict
    retry_count: int
    last_error: str
    df: Any
