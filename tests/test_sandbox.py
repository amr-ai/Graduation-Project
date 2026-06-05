from __future__ import annotations

import pandas as pd
import pytest

from tools.sandbox import run_analysis, strip_fences, df_context


class TestStripFences:
    def test_strips_python_fence(self) -> None:
        code = "```python\nprint('hello')\n```"
        assert strip_fences(code) == "print('hello')"

    def test_strips_plain_fence(self) -> None:
        code = "```\nprint('hello')\n```"
        assert strip_fences(code) == "print('hello')"

    def test_no_fence(self) -> None:
        code = "print('hello')"
        assert strip_fences(code) == "print('hello')"


class TestRunAnalysis:
    def test_simple_code(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = run_analysis("print(df['x'].sum())", df)
        assert result["error"] is None
        assert "6" in result["output"]

    def test_code_error(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = run_analysis("print(undefined_var)", df)
        assert result["error"] is not None

    def test_fig_capture(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        code = "fig = px.line(df, x='x', y='y')"
        result = run_analysis(code, df)
        assert result["error"] is None
        assert result.get("fig") is not None

    def test_dataframe_not_mutated(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = run_analysis("df['x'] = df['x'] * 0", df)
        assert result["error"] is None
        assert result["df"]["x"].tolist() == [0, 0, 0]

    def test_output_capped(self) -> None:
        df = pd.DataFrame({"x": [1]})
        code = "print('A' * 2_000_000)"
        result = run_analysis(code, df)
        assert result["error"] is None
        assert len(result["output"]) < 2_500_000


class TestDfContext:
    def test_basic_context(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]})
        ctx = df_context(df)
        assert "3 rows" in ctx
        assert "x" in ctx
        assert "y" in ctx

    def test_empty_df(self) -> None:
        df = pd.DataFrame()
        ctx = df_context(df)
        assert "0 rows" in ctx
