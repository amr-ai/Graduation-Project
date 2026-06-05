---
name: professionalization-roadmap
description: The agreed 4-feature professionalization plan for Smart Analyst and its status
metadata:
  type: project
---

Agreed 2026-06-05 to build four upgrades (user selected all four; priority =
analytical value). Status:

1. **DONE** — Real forecast backtesting + honest model selection. Added
   `agents/forecasting/tools/models.py` (Naive, Seasonal Naive, Drift, Moving
   Average, Linear Trend, Holt-Winters[optional], Prophet) and `backtest.py`
   (rolling-origin CV, `select_model`, `skill_vs_naive`). `ForecastEngine.run`
   now back-tests candidates and reports out-of-sample metrics; `model_override`
   is wired through pipeline→state→nodes; UI shows a leaderboard + skill score.
   Replaced the previous fake "Auto" (always Prophet, in-sample metrics).
   Tests: `tests/test_forecast_backtest.py`. Also fixed a pandas-3 bug where
   text columns (now `string` dtype, not `object`) weren't detected in
   `schema_intel.py` / `quality.py`.

1b. **DONE** — Churn Prediction agent (`agents/churn/`): schema-aware feature
   engineering (`features.py`), windowed labelling + GradientBoosting with a
   recency-heuristic fallback (`model.py`), orchestrator payload with risk tiers
   + revenue-at-risk + top at-risk customers (`engine.py`), LLM retention-plan
   layer (`agent.py` + `prompts/churn/strategy.md`), PG persistence
   (`storage.py`), UI `pages/6_Churn.py` (registered in nav), `DEFAULT_CHURN_MODEL`
   + sidebar selector. Tests: `tests/test_churn.py`. On Bloom & Bean: AUC 0.926,
   recency is the top driver, surfaces high-value lapsing customers. Avoids
   circular labelling via observation/outcome window split. Also added a
   realistic test dataset generator `generate_business_data.py` →
   `data/bloom_and_bean_sales*.csv`, and boosted revenue-column detection in
   `schema_intel.py` (prefer total/revenue over unit_price/cost).

2. **TODO** — Harden the code sandbox. `tools/sandbox.py` + `agents/cleaning/
   executor.py` use `exec()` with restricted builtins, which is trivially
   escapable (e.g. `().__class__.__bases__[0].__subclasses__()`). Move to
   subprocess isolation (timeout/memory/no-net) or whitelisted tool-calling.

3. **TODO** — Supervisor router agent: one NL entry point that classifies intent
   and dispatches to clean/viz/forecast/marketing (LangGraph supervisor pattern).

4. **TODO** — Token/cost tracking + tracing across all Groq/LangGraph calls
   (wrapper that logs `usage`; per-run cost panel; optional Langfuse/LangSmith).

Other noted-but-unscheduled gaps: orphaned forecasting code (`interpret_node`,
`discover_metrics_node`, `store_node` not wired into the graph; the forecast
graph is validate→prepare→forecast→END so LLM business narrative is currently
absent); broad `except: pass` blocks; graphs rebuilt every Streamlit rerun
(wrap with `@st.cache_resource`). See [[smart-analyst-project]].
