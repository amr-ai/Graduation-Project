# Smart Analyst — Multi-Agent Business Intelligence Platform

Smart Analyst is an AI-powered, **multi-agent data analytics platform** for small and
medium businesses (with an e-commerce / retail focus). A non-technical business owner
uploads a raw CSV and the system **cleans it, analyses it, visualises it, forecasts it,
builds marketing strategy from it, and predicts customer churn** — each step driven by a
specialised agent with a human-in-the-loop checkpoint.

It is built on **LangGraph** (agent orchestration), **Streamlit** (multipage UI),
**Groq** (LLM inference), and the **pandas / scikit-learn / Prophet** scientific stack.

> This README is the single source of truth for the project's architecture and is written
> so it can be handed to any developer or AI assistant to understand and extend the system.

---

## Table of Contents
1. [What it does](#what-it-does)
2. [Core design principles](#core-design-principles)
3. [Tech stack](#tech-stack)
4. [System architecture](#system-architecture)
5. [The six agents](#the-six-agents)
6. [End-to-end workflow & data flow](#end-to-end-workflow--data-flow)
7. [Project structure](#project-structure)
8. [Shared/core components](#sharedcore-components)
9. [Setup & installation](#setup--installation)
10. [Running the project](#running-the-project)
11. [Configuration](#configuration)
12. [Datasets](#datasets)
13. [How to add a new agent (worked example)](#how-to-add-a-new-agent-worked-example)
14. [Testing](#testing)
15. [Design notes & known limitations](#design-notes--known-limitations)
16. [Recent enhancements](#recent-enhancements)

---

## What it does

| Stage | Agent | Output |
|-------|-------|--------|
| 1 | **Data Cleaning** | A profiled, validated, cleaned dataset (with a human-approved plan) |
| 2 | **Analytics** (engine) | Deterministic KPIs, time-series intelligence, data-quality report |
| 3 | **Visualization** | Chat-to-chart + an auto-generated multi-chart dashboard |
| 4 | **Business Insights** | A written, audience-aware business report |
| 5 | **Forecasting** | Multi-model, back-tested sales/metric forecasts with confidence |
| 6 | **Marketing** | RFM customer segments → strategy, campaigns, and ad copy |
| 7 | **Churn Prediction** | Per-customer churn probability + retention strategy |

The platform is intentionally **domain-aware** (e-commerce): it auto-detects revenue,
quantity, customer, order, product, date, and category columns so it works on most sales
exports without configuration.

---

## Core design principles

1. **Deterministic math first, LLM narrates second.** All numbers (KPIs, RFM, forecasts,
   churn probabilities) are computed in pure pandas / scikit-learn / Prophet. The LLM is
   given those computed numbers and asked only to *explain* them. This prevents the model
   from hallucinating figures — a critical property for a business tool.
2. **Human-in-the-loop (HITL).** The cleaning plan is reviewed and editable before any code
   runs; results are inspected before they flow downstream.
3. **Self-correcting code generation.** When the cleaning Coder produces code that fails, the
   error traceback is fed back to the LLM (a repair prompt) and it retries.
4. **Per-agent model selection.** Each agent can use a different LLM, chosen in the sidebar.
5. **Graceful degradation.** Optional dependencies (PostgreSQL, statsmodels) and optional
   columns are all handled with safe fallbacks.
6. **Separation of concerns.** Each agent is a self-contained package: deterministic engine,
   optional LLM layer, optional persistence, and a UI page.
7. **Verifiable output.** Every generated report is passed through a **grounding check**
   (`agents/reporting/grounding.py`) that traces each figure in the narrative back to a
   computed value and flags anything it cannot match — a measurable guard against
   hallucinated numbers, surfaced as a badge in the UI and in the export.
8. **Portable deliverables.** Insights, Marketing, Churn and Forecast reports export to a
   styled, self-contained, print-ready HTML document (open → *Save as PDF*) and to raw
   Markdown, so the output is board-ready — with zero extra dependencies.

---

## Tech stack

| Concern | Technology |
|---------|------------|
| Agent orchestration | **LangGraph** (`StateGraph`, conditional edges, retries) |
| UI | **Streamlit** multipage app (`st.navigation`) |
| LLM inference | **Groq** API (`groq` SDK) — Llama 3.x, GPT-OSS, Qwen, Mixtral |
| Data | **pandas**, **numpy** |
| ML / stats | **scikit-learn** (churn), **Prophet** + **statsmodels** (forecasting), **scipy** |
| Charts | **Plotly** |
| Persistence (optional) | **PostgreSQL** via **SQLAlchemy** |
| Config | **python-dotenv** (`.env`) |
| Tests | **pytest** |
| Lint/format | **ruff** |

Environment used in development: **Python 3.14**, pandas 3.x, numpy 2.x.

---

## System architecture

```
                          ┌──────────────────────────────────────────────┐
                          │              streamlit_app.py                  │
                          │   Sidebar: Groq key + per-agent model picker   │
                          │   Navigation across 6 pages, shared session    │
                          └───────────────────────┬──────────────────────┘
                                                  │ st.session_state
        ┌──────────────┬───────────────┬─────────┴───────┬───────────────┬───────────────┐
        ▼              ▼               ▼                 ▼               ▼               ▼
   1_Data_Cleaning  2_Visualization 3_Business_Insights 4_Forecasting  5_Marketing    6_Churn
        │              │               │                 │               │               │
        ▼              ▼               ▼                 ▼               ▼               ▼
  cleaning graph   viz graph +     analytics engine   forecast       marketing       churn
  (LangGraph)      dashboard       + insights LLM      pipeline       engine + LLM    engine + LLM
        │                                              (LangGraph)
        ▼
  clean_df ──► (in-memory session state) ──► every downstream agent
        │
        └────► (optional) PostgreSQL table  ──► every downstream agent

   Shared services:  core/state.py (GraphState) · tools/sandbox.py (safe exec) ·
                     tools/db_tools.py (Postgres) · tools/profiler_tools.py ·
                     agents/analytics/* (schema intelligence, KPIs, time-series, quality)
```

**Three independent LangGraph state machines** exist:
- **Cleaning** (`core/state.py: GraphState`) — profiler → planner (planning graph), then
  coder → executor → validator (cleaning graph).
- **Forecasting** (`agents/forecasting/forecast_models.py: ForecastState`) — validate →
  prepare → forecast.
- **Visualization** (`agents/visualization/viz_models.py: VizState`) — retrieve_schema →
  coder → executor.

The Analytics, Insights, Marketing, and Churn agents are plain Python pipelines (no graph)
that call deterministic engines and then an LLM layer.

---

## The six agents

### 1. Data Cleaning  (`agents/cleaning/`, `graphs/`, `pages/1_Data_Cleaning.py`)
A 5-node LangGraph pipeline split into two stages around a human checkpoint.

| Node | File | LLM? | Role |
|------|------|------|------|
| Profiler | `profiler.py` | No | Standardises placeholders → NaN; builds a `DatasetProfile` (per-column types, missing %, outliers, type mismatches, arithmetic relationships) |
| Planner | `planner.py` | Yes | Turns the profile into a numbered, human-readable cleaning plan |
| *(human review)* | `pages/1_Data_Cleaning.py` | — | User edits/approves the plan |
| Coder | `coder.py` | Yes | Generates a self-contained `clean_data(df)` function; on retry switches to a repair prompt with the previous code + traceback |
| Executor | `executor.py` | No | Runs the code in a restricted sandbox; guards against 0-row / >50% row loss; captures a transformation log |
| Validator | `validator.py` | No | Compares raw vs. clean (nulls, duplicates, regressions) and emits a pass/fail report |

Graphs: `graphs/planner_graph.py` (profiler→planner) and `graphs/cleaning_graph.py`
(coder→executor→[retry up to 2]→validator). Prompts live in `prompts/`.

### 2. Analytics  (`agents/analytics/`) — deterministic engine, no UI page of its own
The analytical backbone reused by Insights, Marketing, and Churn.
- `schema_intel.py` — heuristically detects column roles (time, monetary, quantity,
  product, customer, order, category) from names + dtypes. **`build_schema_summary(df)`** is
  the key entry point.
- `kpi_engine.py` — revenue, orders, AOV, revenue-per-customer, top/bottom products,
  new-vs-returning customers, monthly growth, category breakdowns.
- `timeseries.py` — daily/weekly aggregation, rolling means, trend slope, anomaly detection
  (z-score + IQR), spikes/drops, structural breaks.
- `quality.py` — data-quality checks (missing, duplicates, type consistency).
- `engine.py` — `run_analytics(df)` orchestrates all of the above into one payload.

### 3. Visualization  (`agents/visualization/`, `pages/2_Visualization.py`)
- `builder.py` — `build_executive_dashboard(df)`: a **deterministic** board-quality dashboard
  (KPI cards + curated charts — revenue trend, category & product leaders, customer mix,
  monthly/day-of-week seasonality, order-value distribution) computed straight from the
  schema/KPI engine, so it is always correct and consistent. Uses a corrected revenue series
  (price × quantity when only a unit price exists) and non-overlapping customer segmentation.
- `theme.py` — the registered `smart_analyst` Plotly template (palette, typography, gridlines)
  + `style_figure()` so **every** chart, deterministic or AI-written, shares one house style,
  plus number-formatting helpers.
- `export.py` — `build_dashboard_html()`: bundles the whole dashboard into one self-contained,
  interactive, print-to-PDF HTML file.
- `viz_graph.py` — chat-to-chart: user asks in natural language → LLM writes Plotly code →
  executed in the sandbox → figure rendered (themed). Retries on error.
- `dashboard.py` — `generate_dashboard()` asks the LLM to identify KPIs and emit 5–7 diverse
  charts in one pass, with professional-quality rules + automatic error-repair retries.
- The page renders a KPI band + responsive grid canvas with **live slicers** (date range +
  category) that re-filter and rebuild the dashboard instantly.
- `viz_models.py` — `VizState` TypedDict.

### 4. Business Insights  (`agents/insights/`, `pages/3_Business_Insights.py`)
- `template_selector.py` — picks the best report template (`non_technical`, `executive`,
  `detailed`) from business context + analytics payload.
- `agent.py` — `generate_insights()` feeds the chosen template + the deterministic analytics
  payload to the LLM. Templates live in `prompts/insights/`.

### 5. Forecasting  (`agents/forecasting/`, `pages/4_Forecasting.py`)
A LangGraph pipeline with a **real, back-tested model selector** and honest confidence.
- `pipeline.py` — `ForecastPipeline.run(..., granularity=...)` builds state and invokes the
  graph; persists results if configured.
- `graph.py` / `nodes.py` — validate → prepare → forecast → interpret. `forecast_node`
  forecasts each target metric in parallel, at the chosen **granularity** (daily/weekly/
  monthly): additive metrics (revenue, quantity) are summed per period, rates averaged, and
  day-based horizons are converted to periods.
- `validation.py` / `metric_discovery.py` — date detection, frequency, history sufficiency,
  and metric ranking that prioritises the true revenue column.
- `tools/models.py` — model library: Naive, Seasonal Naive, Drift, Moving Average,
  Linear Trend, Holt-Winters, **Theta** (M3 winner), **ETS**, **ARIMA** (statsmodels),
  Prophet.
- `tools/backtest.py` — **rolling-origin cross-validation**, `select_model` (ranks by
  out-of-sample MAPE), `skill_vs_naive`; seasonal series are now reliably back-testable.
- `tools/engines.py` — `ForecastEngine.run(model, series, horizon)`: when `model="Auto"`,
  back-tests all candidates and picks the winner; reports honest out-of-sample metrics,
  a selection reason, the leaderboard, and a skill score.
- Confidence = a principled, bounded transform of the back-test error rewarded for beating
  the naive baseline; it is **`None` ("not back-tested")** rather than a fabricated number
  when a series can't be cross-validated.
- `interpreter.py` — LLM business narrative (wired into the graph). `report.py` assembles the
  exportable, grounded forecast report.
- `schema.py` — `ForecastOutput` / `ForecastEvaluation` pydantic models.
- `storage.py` — PostgreSQL persistence of runs/forecasts/evaluations/history.

### 6. Marketing  (`agents/marketing/`, `pages/5_Marketing.py`)
- `segmentation.py` — **RFM** (Recency/Frequency/Monetary) segmentation into named segments
  (Champions … Lost) with per-segment stats and playbooks. Pure pandas.
- `engine.py` — `run_marketing_analytics()` = schema + KPIs + RFM + marketing KPIs
  (repeat rate, CLV proxy, churn-risk base) + channel/dimension performance.
- `agent.py` — LLM layer: strategy report (markdown), campaign plan (JSON), ad copy (JSON).
- `storage.py` — PostgreSQL persistence. Prompts in `prompts/marketing/`.

### 7. Churn Prediction  (`agents/churn/`, `pages/6_Churn.py`)
Supervised churn modelling on transaction data.
- `features.py` — schema-aware per-customer features (recency, frequency, monetary, tenure,
  AOV, inter-purchase gap, product/category diversity, discount usage), all measured as-of a
  cutoff date.
- `model.py` — **windowed labelling** (features as-of `cutoff = snapshot − horizon`; label =
  "did NOT purchase in the next horizon") + `GradientBoostingClassifier` with **probability
  calibration** (so a 0.7 really means ~70% risk), and a recency heuristic fallback for
  small/single-class data.
- `engine.py` — `run_churn_analysis()` → payload: model metrics (AUC, precision, recall),
  feature importance, risk tiers, revenue-at-risk **and expected (probability-weighted)
  revenue-at-risk**, and the top at-risk customers **ranked by expected value at risk
  (churn probability × spend)** with a per-customer `expected_loss`.
- `agent.py` + `prompts/churn/strategy.md` — LLM retention strategy grounded in the payload.
- `storage.py` — PostgreSQL persistence.

---

## End-to-end workflow & data flow

```
Upload CSV (Data Cleaning page)
        │
        ▼
[planner graph]  profiler → planner  ──►  cleaning plan + DatasetProfile
        │
        ▼  (HUMAN reviews / edits / approves the plan)
        │
[cleaning graph] coder → executor → (retry≤2) → validator
        │
        ▼
   clean_df  ─────────────────────────────────────────────┐
        │                                                  │
        ├─► st.session_state.clean_df  (always, in-memory) │
        └─► PostgreSQL table  clean_<filename>  (if configured)
                                                           │
        ┌──────────────────────────────────────────────────┘
        ▼            ▼              ▼              ▼              ▼
  Visualization  Insights      Forecasting    Marketing       Churn
  (reads clean_df or the Postgres table; each runs its deterministic
   engine first, then optionally an LLM narration step)
```

Key mechanism: the cleaned dataset lives in `st.session_state.clean_df` (and optionally a
Postgres table). Every downstream page calls a small `_resolve_dataframe()` helper that
returns the in-memory frame or loads it from Postgres. **You must run Data Cleaning first**;
the other agents are disabled until cleaned data exists.

---

## Project structure

```
.
├── main.py                     # CLI entry point (cleaning pipeline only, with HITL)
├── streamlit_app.py            # Multipage app: sidebar (key + per-agent models) + nav
├── generate_sample_data.py     # Generates data/sample.csv (dirty electronics retail)
├── generate_business_data.py   # Generates data/bloom_and_bean_sales*.csv (rich, realistic)
├── requirements.txt
├── pyproject.toml              # ruff + pytest config
├── .env.example                # GROQ_API_KEY + optional PG_* vars
│
├── core/
│   └── state.py                # GraphState TypedDict (cleaning pipeline state)
├── graphs/
│   ├── planner_graph.py        # profiler → planner
│   └── cleaning_graph.py       # coder → executor → (retry) → validator
├── models/
│   └── profiler_models.py      # ColumnProfile / DatasetProfile (pydantic)
├── tools/
│   ├── sandbox.py              # restricted exec() for LLM code; figure/df capture
│   ├── db_tools.py             # PostgreSQL connect/store/load/schema (SQLAlchemy)
│   └── profiler_tools.py       # pandas logic for building DatasetProfile
│
├── agents/
│   ├── constants.py            # AVAILABLE_MODELS + DEFAULT_*_MODEL + DEFAULT_HORIZONS
│   ├── cleaning/               # profiler, planner, coder, executor, validator
│   ├── analytics/              # schema_intel, kpi_engine, timeseries, quality, engine
│   ├── visualization/          # builder (deterministic dashboard), theme, export,
│   │                           #   viz_graph, dashboard (AI charts), viz_models
│   ├── insights/               # agent, template_selector
│   ├── forecasting/            # pipeline, graph, nodes, validation, metric_discovery,
│   │   │                       #   interpreter, report, schema, storage, forecast_models
│   │   └── tools/              # models, backtest, engines, evaluation
│   ├── marketing/              # segmentation, engine, agent, storage
│   ├── churn/                  # features, model, engine, agent, storage
│   └── reporting/             # grounding (faithfulness check), export (HTML/PDF), render
│
├── prompts/
│   ├── planner_prompt.txt  coder_prompt.txt  repair_prompt.txt
│   ├── insights/   non_technical.md  executive.md  detailed.md
│   ├── marketing/  strategy.md  campaigns.md  ad_copy.md
│   └── churn/      strategy.md
│
├── pages/
│   ├── 1_Data_Cleaning.py  2_Visualization.py  3_Business_Insights.py
│   └── 4_Forecasting.py    5_Marketing.py      6_Churn.py
│
├── data/                       # sample CSVs + Bloom & Bean dataset + data dictionary
└── tests/                      # pytest suite (analytics, forecasting, churn, marketing, …)
```

---

## Shared/core components

- **`core/state.py: GraphState`** — the typed state passed between cleaning-pipeline nodes
  (raw_df, planner_model, coder_model, metadata, cleaning_plan, generated_code,
  execution_result, validation_report, transformation_log, retry_count, last_error).
- **`tools/sandbox.py`** — `run_analysis(code, df)` executes LLM-generated Python with a
  restricted `__builtins__`, a 30s timeout, 1 MB output cap, and automatic capture of a
  Plotly `fig`, a modified `df`, and stdout. Also `df_context()` (LLM-friendly schema text)
  and `sanitize_fig()` (JSON-safe Plotly).
- **`tools/db_tools.py`** — `is_pg_configured()`, `store_df_to_pg()`, `load_df_from_pg()`,
  `get_schema_info()`; identifiers validated against SQL-injection.
- **`agents/analytics/schema_intel.py`** — the column-role detector that makes the whole
  platform "configuration-free".
- **`agents/reporting/`** — the shared report layer used by every narrative-producing agent:
  - `grounding.py` — `check_grounding(report, *payloads)` extracts the material figures
    (currency, percentages, large numbers) from a report and verifies each against the
    deterministic payload, returning a coverage score and the list of unmatched figures.
    Handles fraction-vs-percent (`0.083` ↔ `8.3%`), magnitude suffixes (`$1.2M`), and
    sensible rounding; ignores structural counts and years. Pure stdlib, unit-tested.
  - `export.py` — `build_html_report(...)` and a dependency-free `markdown_to_html(...)`
    that produce the styled, print-ready HTML deliverable.
  - `render.py` — `render_report(...)`, the single Streamlit presentation used by all four
    report pages (header band, KPI strip, grounding badge, HTML + Markdown downloads).

---

## Setup & installation

```bash
# 1. (recommended) create a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 2. install dependencies
pip install -r requirements.txt

# 3. configure secrets
cp .env.example .env        # then edit .env and add your Groq API key
```

`.env` keys:
```
GROQ_API_KEY=your_groq_api_key_here      # required (get one at console.groq.com)
PG_HOST=localhost                        # optional — only for PostgreSQL persistence
PG_PORT=5432
PG_DBNAME=smart_analyst
PG_USER=postgres
PG_PASSWORD=postgres
```

> Note: `prophet` (and optionally `statsmodels`) are scientific packages; on some systems the
> first install compiles native code and can take a few minutes.

---

## Running the project

```bash
# Launch the full multipage web app (primary way to use the project)
streamlit run streamlit_app.py
python -m streamlit run streamlit_app.py #for mac

# Run the cleaning pipeline only, from the terminal (uses data/sample.csv)
python main.py

# Generate test datasets
python generate_sample_data.py        # -> data/sample.csv (dirty electronics retail)
python generate_business_data.py      # -> data/bloom_and_bean_sales*.csv (rich, realistic)

# Run the test suite
python -m pytest -q
```

**Typical session:** `streamlit run streamlit_app.py` → paste Groq key in the sidebar →
**Data Cleaning** → upload `data/bloom_and_bean_sales.csv` → review/approve plan →
explore **Visualization / Insights / Forecasting / Marketing / Churn**.

---

## Configuration

- **Groq API key** — entered in the sidebar (or `.env`).
- **Per-agent models** — sidebar → *Per-Agent Settings*. Each agent has its own selector and
  a session-state key (`planner_model`, `coder_model`, `viz_model`, `dashboard_model`,
  `insights_model`, `forecast_model`, `marketing_model`, `churn_model`). Available models and
  defaults live in `agents/constants.py`.
- **PostgreSQL** — set `PG_*` env vars to enable persistence; everything works without it.

---

## Datasets

| File | Description |
|------|-------------|
| `data/sample.csv` | Dirty electronics-retail sample (from `generate_sample_data.py`) |
| `data/bloom_and_bean_sales.csv` | **Recommended test file** — realistic, dirty coffee-store sales (24 months, repeat customers, trend + seasonality, profit/cost). Upload this to Data Cleaning. |
| `data/bloom_and_bean_sales_clean.csv` | Clean reference version |
| `data/README_bloom_and_bean.md` | Data dictionary + which patterns each agent should find |
| `data/cafe_sales.csv`, `data/dirty_cafe_sales.csv` | Additional test files |

The **Bloom & Bean** generator is deliberately designed so every agent has something to find:
duplicates/outliers/bad-dates for cleaning, KPIs + ~74% repeat rate for analytics, anomalies
(Black Friday spikes, a supply-outage dip) for time-series, a forecastable daily series, a
full RFM segment spread for marketing, and a strong recency signal for churn.

---

## How to add a new agent (worked example)

The **Churn agent** is the cleanest template to copy. To add an agent `foo`:

1. **Create the package** `agents/foo/`:
   - `features.py` / `engine.py` — the **deterministic** computation. Reuse
     `agents.analytics.schema_intel.build_schema_summary(df)` to find column roles. Return a
     JSON-serialisable `payload` dict (this is what the LLM and UI consume).
   - `model.py` — any ML (scikit-learn etc.), if applicable.
   - `agent.py` — the **LLM layer**. Load a prompt from `prompts/foo/`, pass the deterministic
     payload as grounded context, call Groq with `st.session_state.get("foo_model", DEFAULT_FOO_MODEL)`.
   - `storage.py` — optional PostgreSQL persistence (mirror `agents/churn/storage.py`).
2. **Add the prompt** `prompts/foo/strategy.md` (instruct the model to use only payload numbers).
3. **Register the model**: add `DEFAULT_FOO_MODEL` to `agents/constants.py`.
4. **Wire the UI**:
   - Add a selector in `streamlit_app.py` (sidebar expander) with `key="foo_model"`.
   - Add `st.Page("pages/7_Foo.py", title="Foo")` to the `pages` list.
   - Add any session defaults to the `_DEFAULTS` dict.
   - Create `pages/7_Foo.py` (copy `pages/6_Churn.py`: `_data_available()` /
     `_resolve_dataframe()` → run engine → render metrics/charts → optional LLM narration).
5. **Add tests** in `tests/test_foo.py` (test the deterministic engine on synthetic data;
   no network/LLM in tests).

Conventions to follow: deterministic-first; degrade gracefully on missing columns; never let
the LLM invent numbers; keep the engine pure (no Streamlit imports) so it's unit-testable.

---

## Testing

```bash
python -m pytest -q              # whole suite
python -m pytest tests/test_churn.py -q
```

Tests cover the deterministic engines (schema intelligence, KPIs, time-series, quality,
forecasting back-test/validation, marketing, churn) and the sandbox. They are
network-free and LLM-free by design (the LLM layers are thin and exercised manually in the UI).

---

## Design notes & known limitations

- **Sandbox is restricted, not bulletproof.** `tools/sandbox.py` / `agents/cleaning/executor.py`
  run LLM-generated code with a shared restricted `__builtins__` allow-list plus a blocked-pattern
  pre-scan (`__class__`, `__subclasses__`, `__import__`, `open(`, `subprocess`, …) and `getattr`
  removed. This blocks casual misuse and the common escape vectors, but is still not a true
  security boundary (Python sandbox escapes exist). For fully untrusted input, move execution to a
  subprocess/container or a whitelisted operation set. *(Planned hardening.)*
- **Forecasting LLM narrative is wired into the graph.** The graph is
  validate→prepare→forecast→interpret→END; `agents/forecasting/interpreter.py` enriches every
  forecast output with a grounded `business_summary` + risks/opportunities/actions, falling back
  to a deterministic summary when no API key is set.
- **Single LLM provider.** Everything uses Groq; a provider-abstraction layer would allow
  fallback/other providers.
- **State is session-scoped.** Analyses live in `st.session_state` (and optionally Postgres);
  there is no per-user auth or saved-project concept yet.
- **No token/cost tracking** across LLM calls yet.

See `FEATURE_ROADMAP.md` for the prioritised plan addressing these and adding new value.

---
## Recent enhancements

- **Churn & Marketing correctness/quality** — fixed a shared KPI bug where `new_customers`
  equalled *total* (so `new_customer_pct` was always ~100%); new vs returning is now a correct
  non-overlapping partition (one-time vs repeat), which also corrects the Marketing report.
  Churn now **ranks at-risk customers by expected value at risk** (probability × spend) with a
  per-customer `expected_loss` and an **expected-revenue-at-risk** KPI, and **calibrates** the
  churn probabilities so scores are trustworthy. Tests updated in `tests/test_churn.py`,
  `tests/test_kpi_engine.py`.
- **Forecasting accuracy & honesty** — additive metrics are now summed (not averaged) per
  period; a **granularity control** (daily/weekly/monthly) lets volatile daily series be
  forecast where the back-test error is meaningful (on Bloom & Bean, weekly revenue moved from
  a daily 90% MAPE / 37% confidence to Seasonal Naive at 29% MAPE / **70% confidence, +68%
  skill vs naive**); three professional models added (**Theta, ETS, ARIMA**); the confidence
  score is now a principled transform of back-test error and reports **"not back-tested"**
  instead of a fake number; metric ranking prioritises real revenue. Tests:
  `tests/test_forecast_upgrades.py`.
- **Executive Visualization dashboard** — the Visualization agent now leads with a
  deterministic, themed, board-quality dashboard (KPI band + curated charts, one house
  Plotly theme, live date/category slicers, one-click interactive-HTML export) instead of a
  tab strip of ad-hoc LLM charts; AI charts remain for anything custom, matched to the theme.
  New: `agents/visualization/{theme,builder,export}.py`.
- **Professional reporting layer (`agents/reporting/`)** — all four narrative agents
  (Insights, Marketing, Churn, Forecasting) now render through one shared presentation with
  a run-metadata header, a KPI strip, a **grounding/faithfulness badge** that verifies every
  figure against the computed data, and one-click export to styled **HTML (print-to-PDF)** and
  Markdown. The forecast narrative is assembled deterministically from the pipeline output
  (`agents/forecasting/report.py`). No new dependencies. Tests: `tests/test_reporting.py`.
- **Forecasting** — replaced the placeholder "Auto" (which always ran Prophet and reported
  in-sample metrics) with a real multi-model library + rolling-origin cross-validation +
  honest out-of-sample selection, a back-test leaderboard, and a skill-vs-naive score.
- **Churn Prediction agent** — new sixth agent (windowed labelling + gradient boosting,
  AUC ≈ 0.93 on the Bloom & Bean data) with a retention-strategy LLM layer.
- **Realistic dataset generator** — `generate_business_data.py` (Bloom & Bean).
- **pandas 3.x compatibility + revenue-column detection fix** in
  `agents/analytics/schema_intel.py` / `quality.py`.
