"""
Churn Prediction Agent — UI.

Deterministic ML core (windowed labelling + gradient boosting) scores every
current customer's probability of churning over the next horizon. The LLM is
used only for the retention-strategy narrative.
"""

from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import streamlit as st

from agents.churn.agent import generate_retention_plan
from agents.churn.engine import run_churn_analysis
from agents.churn.storage import save_churn_run
from agents.constants import DEFAULT_CHURN_MODEL
from agents.reporting.render import build_meta, render_report
from tools.db_tools import is_pg_configured


def _data_available() -> bool:
    return (
        st.session_state.clean_df is not None
        or (st.session_state.pg_stored and st.session_state.pg_table)
    )


def _resolve_dataframe() -> tuple[pd.DataFrame | None, str]:
    if st.session_state.clean_df is not None:
        return st.session_state.clean_df, ""
    if st.session_state.pg_stored and st.session_state.pg_table:
        try:
            from tools.db_tools import load_df_from_pg
            tbl = st.session_state.pg_table.lower()
            return load_df_from_pg(tbl), tbl
        except Exception as exc:
            st.error(f"PostgreSQL load failed: {exc}")
    return None, ""


st.title("Churn Prediction Agent")
st.caption(
    "Predicts which customers are likely to stop buying — deterministic ML "
    "(observation/outcome window + gradient boosting) · LLM only narrates the plan"
)

if not _data_available():
    st.info("Upload and clean data in **Data Cleaning** first.")
    st.stop()

df, table = _resolve_dataframe()
if df is None:
    st.stop()

# ── Configuration ────────────────────────────────────────────────────────────
st.subheader("Configuration")
c1, c2, c3 = st.columns(3)
with c1:
    horizon = st.selectbox("Churn horizon (days)", [30, 60, 90, 120, 180], index=2,
                           help="A customer is 'churned' if they don't purchase within this window.")
with c2:
    top_n = st.slider("Customers in at-risk table", 20, 500, 200, step=20)
with c3:
    business_context = st.text_area("Business context (optional)", height=80,
                                    placeholder="Industry, goals, constraints…")

if st.button("Run Churn Analysis", type="primary"):
    with st.spinner("Engineering features and training the churn model…"):
        try:
            st.session_state.churn_payload = run_churn_analysis(
                data_df=df, table_name=table, horizon_days=horizon, top_n=top_n
            )
            st.session_state.churn_report = ""
        except Exception as exc:
            st.error(f"Churn analysis failed: {exc}")
            st.stop()

payload = st.session_state.get("churn_payload")
if not payload:
    st.stop()

if not payload.get("available"):
    st.warning(payload.get("reason", "Churn analysis not available for this data."))
    st.stop()

# ── Model quality ────────────────────────────────────────────────────────────
model = payload["model"]
st.divider()
st.subheader("Model")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Model", model.get("name", "—"))
auc = model.get("auc")
m2.metric("ROC-AUC", f"{auc:.3f}" if auc is not None else "—")
m3.metric("Recall", f"{model.get('recall', '—')}")
m4.metric("Base churn rate", f"{model.get('base_churn_rate', 0) * 100:.0f}%"
          if model.get("base_churn_rate") is not None else "—")
st.caption(
    f"Snapshot {payload['snapshot_date']} · features measured as-of "
    f"{payload['cutoff_date']} · horizon {payload['horizon_days']} days · "
    f"{model.get('n_test', '—')} held-out customers."
    + (" · probabilities calibrated" if model.get("calibrated") else "")
)
if model.get("note"):
    st.info(model["note"])

# ── Risk distribution ────────────────────────────────────────────────────────
st.subheader("Risk overview")
rd = payload["risk_distribution"]
r1, r2, r3 = st.columns(3)
r1.metric("High risk (≥70%)", rd["high"])
r2.metric("Medium (40–70%)", rd["medium"])
r3.metric("Low (<40%)", rd["low"])

rar = payload.get("revenue_at_risk")
err = payload.get("expected_revenue_at_risk")
m1, m2 = st.columns(2)
m1.metric("Revenue at risk", f"${rar:,.0f}" if rar is not None else "—",
          help="Historical spend of customers with ≥50% churn probability.")
m2.metric("Expected revenue at risk", f"${err:,.0f}" if err is not None else "—",
          help="Probability-weighted exposure: Σ (churn probability × historical spend) — "
               "the risk-adjusted revenue you can expect to lose.")

colA, colB = st.columns(2)
with colA:
    dist_df = pd.DataFrame(
        {"tier": ["High", "Medium", "Low"],
         "customers": [rd["high"], rd["medium"], rd["low"]]}
    )
    fig = px.bar(dist_df, x="tier", y="customers", color="tier",
                 color_discrete_map={"High": "#d62728", "Medium": "#ff7f0e", "Low": "#2ca02c"},
                 title="Customers by churn-risk tier")
    fig.update_layout(showlegend=False, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

with colB:
    fi = payload.get("feature_importance", [])
    if fi:
        fi_df = pd.DataFrame(fi).sort_values("importance")
        fig2 = px.bar(fi_df, x="importance", y="feature", orientation="h",
                      title="What drives churn (feature importance)")
        fig2.update_layout(template="plotly_white")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.caption("Feature importance unavailable (heuristic fallback).")

# ── At-risk customers ────────────────────────────────────────────────────────
st.subheader("Highest-risk customers")
at_risk = payload.get("at_risk_customers", [])
if at_risk:
    table_df = pd.DataFrame(at_risk)
    if "name" in table_df.columns and not table_df["name"].astype(bool).any():
        table_df = table_df.drop(columns=["name"])
    # Lead with expected loss (value at risk) so the priority is obvious.
    preferred = ["customer", "name", "expected_loss", "monetary", "churn_probability",
                 "risk_tier", "recency_days", "frequency"]
    ordered = [c for c in preferred if c in table_df.columns] + \
              [c for c in table_df.columns if c not in preferred]
    table_df = table_df[ordered]
    if payload.get("at_risk_ranked_by") == "expected_value_at_risk":
        st.caption(
            "Ranked by **expected value at risk** = churn probability × historical spend — "
            "so retention effort targets the customers whose loss would hurt most, not just "
            "the highest-probability ones."
        )
    st.dataframe(table_df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download at-risk customers (CSV)",
        data=table_df.to_csv(index=False).encode("utf-8"),
        file_name=f"churn_at_risk_{payload['snapshot_date']}.csv",
        mime="text/csv",
    )

# ── Optional PostgreSQL persistence ──────────────────────────────────────────
if is_pg_configured():
    if st.button("Save run to PostgreSQL"):
        res = save_churn_run(payload, table_name=table)
        if res.get("status") == "stored":
            st.success(f"Saved churn run `{res['run_id']}` ({res['scores']} scores).")
        else:
            st.info(f"Not saved: {res.get('reason') or res.get('error')}")

# ── LLM retention plan ───────────────────────────────────────────────────────
st.divider()
st.subheader("Retention strategy")
has_key = bool(os.environ.get("GROQ_API_KEY"))
if st.button("Generate retention plan", disabled=not has_key):
    with st.spinner("Writing the retention strategy…"):
        try:
            st.session_state.churn_report = generate_retention_plan(
                payload, data_df=df, table_name=table, business_context=business_context
            )
        except Exception as exc:
            st.error(f"Plan generation failed: {exc}")
if not has_key:
    st.caption("Add your Groq API key in the sidebar to generate the narrative plan.")

if st.session_state.get("churn_report"):
    rd = payload.get("risk_distribution", {})
    kpi_cards: list[tuple[str, str]] = []
    if payload.get("expected_revenue_at_risk") is not None:
        kpi_cards.append(("Expected rev. at risk", f"${payload['expected_revenue_at_risk']:,.0f}"))
    elif payload.get("revenue_at_risk") is not None:
        kpi_cards.append(("Revenue at risk", f"${payload['revenue_at_risk']:,.0f}"))
    if rd.get("high") is not None:
        kpi_cards.append(("High risk", f"{rd['high']:,}"))
    if rd.get("medium") is not None:
        kpi_cards.append(("Medium risk", f"{rd['medium']:,}"))
    if model.get("auc") is not None:
        kpi_cards.append(("ROC-AUC", f"{model['auc']:.3f}"))

    meta = build_meta(
        "Retention Strategist",
        model=st.session_state.get("churn_model", DEFAULT_CHURN_MODEL),
        dataset=(table or "In-memory data"),
        extra={"Horizon": f"{payload.get('horizon_days', '—')} days"},
    )
    render_report(
        "Customer Retention Strategy",
        st.session_state.churn_report,
        meta=meta,
        payloads=[payload],
        kpis=kpi_cards,
        subtitle="Win-back plan grounded in the churn model output",
        filename_stem="retention_strategy",
    )
