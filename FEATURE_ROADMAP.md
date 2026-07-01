# Smart Analyst — Feature Roadmap & High-Value Ideas

A strategic catalogue of features and improvements that take Smart Analyst from a strong
graduation project to a genuinely impressive, defensible, near-production platform. Each item
explains **what** it is, **why it impresses** (and adds real business value), the rough
**effort/impact**, and **where it plugs in** to the existing code.

> Read `README.md` first for the architecture. This document is about where to take it next.

---

## Why the foundation is already strong (lead with this in your defense)

- **Genuine multi-agent architecture** on LangGraph with three state machines and a clean
  agent-per-package design.
- **The right AI safety pattern**: deterministic math (KPIs, RFM, forecasts, churn) computed
  in pandas/scikit-learn/Prophet, with the LLM used only to *narrate* — so it can't
  hallucinate numbers. This is exactly how serious AI analytics products are built.
- **Human-in-the-loop**, **self-repairing code generation**, **configuration-free schema
  detection**, and **real back-tested forecasting** with honest out-of-sample metrics.
- A real **churn model** (AUC ≈ 0.93) using a proper observation/outcome window split (not a
  circular label).

These are the talking points that separate this from a "ChatGPT wrapper".

---

## Scoring legend
- **Impact**: business/academic "wow" value — ⭐ (nice) → ⭐⭐⭐⭐⭐ (headline).
- **Effort**: 🔨 (hours) → 🔨🔨🔨🔨 (weeks).

---

## A. Analytical depth — deepest "real value" features

### A1. Predictive Customer Lifetime Value (CLV)  ⭐⭐⭐⭐⭐ · 🔨🔨
A forward-looking CLV per customer (expected number of future orders × expected value), not
today's average-spend proxy. Use the `lifetimes` library (BG/NBD + Gamma-Gamma) on the
transaction history.
- **Why it impresses**: pairs with churn to answer "who is worth saving?" — the single most
  valuable marketing question. Probabilistic, citable models.
- **Plugs in**: new `agents/clv/` mirroring `agents/churn/`; combine with churn risk to rank
  retention spend by *expected value at risk* = churn_probability × predicted_CLV.

### A2. Market-Basket / Association Rules  ⭐⭐⭐⭐ · 🔨🔨
"Customers who buy X also buy Y." Use `mlxtend` Apriori/FP-Growth on multi-line orders
(the Bloom & Bean data already has 1–3 items per order).
- **Why it impresses**: directly actionable cross-sell/bundle recommendations; very visual
  (network graph / heatmap).
- **Plugs in**: new `agents/basket/`; feed rules into the Marketing agent's campaign prompts.

### A3. Cohort & Retention Analysis  ⭐⭐⭐⭐ · 🔨
Group customers by first-purchase month; chart retention curves and revenue retention over
time (the classic triangular cohort heatmap).
- **Why it impresses**: instantly recognisable "real BI"; reveals whether the business keeps
  customers. Deterministic, fast, beautiful.
- **Plugs in**: extend `agents/analytics/` + a Plotly heatmap on a page or the Insights page.

### A4. Driver / "Why" analysis (explainability)  ⭐⭐⭐⭐⭐ · 🔨🔨🔨
"What is driving revenue / churn?" Train a quick model and surface **SHAP** values or
permutation importance, then have the LLM explain the drivers in business language.
- **Why it impresses**: explainable AI is a top examiner theme; turns black-box ML into
  trustworthy insight. You already expose feature importances in churn — SHAP is the next level.
- **Plugs in**: `agents/churn/` (SHAP on the existing model) and a generic driver module.

### A5. Anomaly & Alerting agent  ⭐⭐⭐ · 🔨🔨
You already detect anomalies, spikes, and structural breaks in `agents/analytics/timeseries.py`.
Promote them to ranked, explained **alerts** ("revenue dropped 35% the week of Mar 10 — likely
a supply issue").
- **Why it impresses**: proactive, not just descriptive. Great live demo on the injected
  Black-Friday spike / supply-outage dip in the Bloom & Bean data.

### A6. Profitability / Margin agent  ⭐⭐⭐ · 🔨
The Bloom & Bean data has `cost` and `profit`. Add margin %, profit by
product/category/channel, and "high-revenue, low-margin" flags.
- **Why it impresses**: revenue is vanity, margin is sanity — shows commercial maturity.

### A7. Subscription / MRR analytics  ⭐⭐⭐ · 🔨🔨
The dataset has a `Subscription` category. Compute MRR, active subscribers, and subscription
churn separately from one-off purchases.

---

## B. Agentic AI capabilities — the "AI" headline

### B1. Supervisor / Router agent ("Ask anything")  ⭐⭐⭐⭐⭐ · 🔨🔨🔨
A single natural-language chat box that classifies intent and dispatches to the right
agent (clean / visualise / forecast / segment / churn) — the LangGraph **supervisor pattern**.
- **Why it impresses**: this is *the* modern agentic-AI talking point and unifies the six
  pages into one product. "Show me last quarter's revenue trend and who's about to churn" →
  routed automatically.
- **Plugs in**: a new top-level LangGraph supervisor that owns the sub-agents as tools/nodes.

### B2. Tool-calling instead of raw code-exec  ⭐⭐⭐⭐ · 🔨🔨🔨
Replace "LLM writes arbitrary Python → exec()" with a whitelisted toolset (filter, groupby,
chart(type, x, y), forecast(metric), …) exposed via function-calling.
- **Why it impresses**: safer, auditable, reproducible; the transformation log becomes a
  replayable recipe. Strong engineering/security story.

### B3. LLM-as-judge self-critique loop  ⭐⭐⭐⭐ · 🔨🔨
After generating an Insights/Marketing report, a second LLM pass checks every number against
the deterministic payload (grounding/faithfulness check) and flags unsupported claims.
- **Why it impresses**: a real *evaluation* component — examiners love measurable quality.
- **✅ Shipped (v1, deterministic):** `agents/reporting/grounding.py` already extracts every
  figure from each report and verifies it against the computed payload (no extra LLM call, no
  cost), surfacing a coverage score + a badge listing any unmatched figures. Next step is the
  optional *LLM* judge pass on top for qualitative critique.

### B4. Cross-session memory  ⭐⭐⭐ · 🔨🔨
Remember a business's context/goals across sessions so reports get more tailored over time.

---

## C. Productionization — "this could ship"

### C1. Token & cost tracking + budget guardrails  ⭐⭐⭐⭐ · 🔨🔨
Wrap the Groq client to log `usage` per call; show a per-run cost/token panel.
- **Why it impresses**: cost-awareness signals production thinking; great dashboards for the
  demo. (Currently there is none.)

### C2. Tracing / observability (Langfuse or LangSmith)  ⭐⭐⭐⭐ · 🔨🔨
Trace every LangGraph run and LLM call. Turns the project from "script" into an observable
system — and gives you superb screenshots for the report.

### C3. Harden the sandbox  ⭐⭐⭐⭐⭐ · 🔨🔨🔨
Move LLM-generated code execution to a subprocess/container with CPU/memory/time limits and
no network (or adopt B2 tool-calling). The current restricted-`exec()` is escapable.
- **Why it impresses**: a concrete, demonstrable **security** improvement — examiners probe
  this for any system that runs generated code.

### C4. Provider-agnostic LLM layer  ⭐⭐⭐ · 🔨🔨
An interface over Groq with retry/fallback so you can add other providers or local models
without touching agents.

### C5. Report export (PDF / PPTX) + scheduled email  ⭐⭐⭐⭐ · 🔨🔨
Export the Insights/Marketing/Forecast/Churn reports and email a "weekly business review".
- **Why it impresses**: turns a tool into a *product* a real owner would pay for.
- **✅ Shipped (v1):** every report now exports to a styled, self-contained, print-ready HTML
  document (Save-as-PDF) and to Markdown via `agents/reporting/export.py` — dependency-free.
  Remaining: native PPTX and scheduled email delivery.

### C6. Saved projects / persistence + auth  ⭐⭐⭐ · 🔨🔨🔨
Replace ephemeral `st.session_state` with saved, reloadable analyses; add multi-user auth.

### C7. Connectors beyond CSV  ⭐⭐⭐ · 🔨🔨🔨
Google Sheets / Shopify / a SQL source, and multi-file joins.

### C8. Dockerfile + CI + coverage for the cleaning agents  ⭐⭐ · 🔨🔨
One-command deploy and green CI badge; add tests for the cleaning graph (currently the LLM
agents are the least-tested area).

---

## D. Product & UX polish

- **D1. Onboarding + sample datasets** in-app (one click to load Bloom & Bean). ⭐⭐ · 🔨
- **D2. Geo dashboard** — choropleth from `region` / `country`. ⭐⭐⭐ · 🔨🔨
- **D3. Shareable report links / read-only mode.** ⭐⭐ · 🔨🔨
- **D4. Wire the forecasting LLM narrative** — `interpreter.py` exists but isn't in the graph;
  connecting it makes forecasts explain themselves. ⭐⭐⭐ · 🔨 (quick win)
- **D5. Unified KPI header** across pages (revenue, orders, customers, churn risk). ⭐⭐ · 🔨

---

## Suggested phased roadmap

**Phase 1 — Quick wins / credibility (days)**
- D4 wire forecasting narrative · C1 cost tracking · A6 margin · A3 cohorts.

**Phase 2 — Headline analytical value (1–2 weeks)**
- A1 predictive CLV · A2 market-basket · A4 SHAP driver analysis (build on churn).

**Phase 3 — Agentic & production (2–4 weeks)**
- B1 supervisor router · C3 sandbox hardening (or B2 tool-calling) · C2 tracing.

**Phase 4 — Product (optional, high polish)**
- C5 report export + scheduling · C6 saved projects + auth · C7 connectors.

A pragmatic "impress the supervisor" trio: **A1 (predictive CLV) + B1 (supervisor router) +
C3 (sandbox hardening)** — one deep data-science feature, one agentic-AI feature, one
engineering/security feature. Together they cover every dimension an examiner cares about.

---

## Defense / presentation talking points

1. **Architecture**: "Six specialised agents on LangGraph; deterministic engines compute the
   numbers, the LLM only explains them — so the system never invents figures."
2. **Rigour**: "Forecasts are selected by rolling-origin cross-validation and reported with
   out-of-sample error and a skill-vs-naive score." / "Churn uses an observation/outcome
   window split to avoid leakage; AUC ≈ 0.93."
3. **Safety**: "LLM-generated code runs in a restricted sandbox; here's the hardening plan."
4. **Business value**: "The platform answers the four questions an owner actually has: What
   happened? Why? What's next? What should I do?"
5. **Engineering**: "Configuration-free schema detection, graceful degradation, a test suite,
   and a clean extension pattern — adding a new agent is a documented, repeatable process."

---

## Metrics to showcase in the report

- Forecasting: MAPE, skill-vs-naive, and the model leaderboard per metric.
- Churn: ROC-AUC, precision/recall, base rate, revenue-at-risk, top drivers (feature importance).
- Data cleaning: nulls/duplicates removed, regressions caught, retries used.
- (If C1/C2 added) tokens & cost per run, end-to-end latency.
