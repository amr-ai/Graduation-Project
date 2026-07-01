"""
Marketing Agent page — RFM segmentation + AI marketing strategy, campaigns and
ad copy, with human-in-the-loop.

Pipeline:
  1. User sets objective / budget / channels / brand voice (HITL).
  2. Deterministic marketing analytics engine runs (RFM segments + KPIs).
  3. LLM generates a grounded strategy report.
  4. LLM generates a structured campaign plan (per-segment, budget-split).
  5. LLM writes ready-to-ship ad copy for a chosen segment + channel.
  6. Optional persistence to PostgreSQL.  Follow-up chat to refine.

Consumes upstream artifacts: cleaned data (Data Cleaning), analytics payload
(Business Insights) and forecast results (Forecasting) when available.
"""

from __future__ import annotations

import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st
from groq import Groq

from agents.constants import DEFAULT_MARKETING_MODEL
from agents.marketing.agent import (
    generate_ad_copy,
    generate_campaign_plan,
    generate_marketing_strategy,
)
from agents.marketing.engine import run_marketing_analytics
from agents.marketing.storage import save_marketing_run
from agents.reporting.render import build_meta, render_report
from tools.db_tools import is_pg_configured
from tools.sandbox import df_context


# ── Data resolution helpers ─────────────────────────────────────────────────

def _data_available() -> bool:
    return (
        st.session_state.clean_df is not None
        or (st.session_state.pg_stored and st.session_state.pg_table)
    )


def _get_data() -> tuple[pd.DataFrame | None, str]:
    if st.session_state.pg_stored and st.session_state.pg_table:
        return None, st.session_state.pg_table.lower()
    return st.session_state.clean_df, ""


# ── Rendering helpers ───────────────────────────────────────────────────────

def _render_marketing_kpis(mk: dict) -> None:
    c = st.columns(5)
    c[0].metric("Repeat rate", f"{mk['repeat_purchase_rate']}%" if mk.get("repeat_purchase_rate") is not None else "—")
    c[1].metric("Avg cust. value", f"${mk['avg_customer_value']:,.0f}" if mk.get("avg_customer_value") else "—")
    c[2].metric("AOV", f"${mk['aov']:,.2f}" if mk.get("aov") else "—")
    c[3].metric("Churn-risk", f"{mk['churn_risk_pct']}%" if mk.get("churn_risk_pct") is not None else "—")
    c[4].metric("High-value", mk.get("high_value_customers") if mk.get("high_value_customers") is not None else "—")


def _render_segments(rfm: dict) -> None:
    if not rfm.get("available"):
        st.warning(f"Segmentation unavailable: {rfm.get('reason', 'insufficient columns')}")
        return

    segs = rfm["segments"]
    st.caption(
        f"{rfm['total_customers']:,} customers · "
        f"${rfm['total_revenue']:,.0f} revenue · "
        f"snapshot {rfm.get('snapshot_date', 'n/a')}"
    )

    # Segment chart: revenue share by segment.
    seg_df = pd.DataFrame([
        {"Segment": name, "Customers": s["count"], "Revenue": s["revenue"],
         "Revenue %": s["revenue_pct"]}
        for name, s in segs.items()
    ])
    if not seg_df.empty:
        fig = px.bar(
            seg_df.sort_values("Revenue", ascending=True),
            x="Revenue", y="Segment", orientation="h",
            title="Revenue by RFM Segment", text="Revenue %",
            color="Customers", color_continuous_scale="Blues",
        )
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        st.plotly_chart(fig, use_container_width=True,
                        config={"displaylogo": False})

    # Segment detail table.
    detail = pd.DataFrame([
        {
            "Segment": name,
            "Customers": s["count"],
            "% Base": s["customer_pct"],
            "Revenue": round(s["revenue"], 2),
            "% Rev": s["revenue_pct"],
            "Avg value": s["avg_monetary"],
            "Avg orders": s["avg_frequency"],
            "Recency (d)": s["avg_recency_days"],
            "Play": s["playbook"],
        }
        for name, s in segs.items()
    ])
    st.dataframe(detail, use_container_width=True, hide_index=True)


def _render_campaigns(plan: dict) -> None:
    campaigns = plan.get("campaigns", [])
    if not campaigns:
        st.info("No campaigns generated. " + plan.get("error", ""))
        return
    if plan.get("summary"):
        st.markdown(f"**Plan summary:** {plan['summary']}")

    table = pd.DataFrame([
        {
            "Campaign": c.get("name"),
            "Segment": c.get("target_segment"),
            "Objective": c.get("objective"),
            "Channel": c.get("channel"),
            "Budget %": c.get("budget_allocation_pct"),
            "Priority": c.get("priority"),
            "Primary KPI": c.get("primary_kpi"),
        }
        for c in campaigns
    ])
    st.dataframe(table, use_container_width=True, hide_index=True)

    for c in campaigns:
        with st.expander(f"{c.get('name', 'Campaign')} → {c.get('target_segment', '')}"):
            st.markdown(f"**Offer:** {c.get('offer', '—')}")
            st.markdown(f"**Angle:** {c.get('message_angle', '—')}")
            st.markdown(f"**Expected impact:** {c.get('expected_impact', '—')}")
            st.markdown(
                f"**Channel:** {c.get('channel', '—')} · "
                f"**Budget:** {c.get('budget_allocation_pct', '—')}% · "
                f"**Priority:** {c.get('priority', '—')}"
            )


# ── Page content ────────────────────────────────────────────────────────────

st.title("Marketing Agent")
st.caption(
    "RFM customer segmentation · AI marketing strategy · campaign plan · ad copy. "
    "Grounded in your cleaned data, analytics and forecasts."
)

if not _data_available():
    st.info("No cleaned data available. Run **Data Cleaning** first.")
    st.stop()

data_df, table = _get_data()

# ── HITL Step 1: Marketing brief ────────────────────────────────────────────

with st.expander("Marketing brief (objective, budget, channels)", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        objective = st.selectbox(
            "Primary objective",
            ["Revenue growth", "Customer retention", "Win-back / reactivation",
             "New customer acquisition", "Loyalty & advocacy", "Maximise AOV"],
        )
        budget = st.text_input(
            "Marketing budget (optional)",
            placeholder="E.g. '$5,000 / month' or leave blank",
        )
    with col2:
        channels = st.multiselect(
            "Allowed channels",
            ["Email", "SMS", "Paid social", "Search ads", "Push", "Retargeting",
             "Influencer", "Content / SEO", "Affiliate"],
            default=["Email", "Paid social", "Retargeting"],
        )
        brand_voice = st.text_input(
            "Brand voice",
            placeholder="E.g. 'playful and bold' or 'premium and minimal'",
        )

    business_context = st.text_area(
        "Additional context (industry, season, promotions, audience)",
        placeholder="E.g. 'Fashion brand, Q4 holiday push, targeting Gen-Z women.'",
    )

use_forecast = st.checkbox(
    "Use forecast results from the Forecasting agent (if available)",
    value=bool(st.session_state.get("forecast_results")),
    disabled=not bool(st.session_state.get("forecast_results")),
)
forecast_results = st.session_state.get("forecast_results") if use_forecast else None

# ── HITL Step 2: Run analytics + generate strategy ──────────────────────────

if st.button("Run Marketing Analysis", type="primary", use_container_width=False):
    if not os.environ.get("GROQ_API_KEY"):
        st.error("Set your Groq API key in the sidebar first.")
        st.stop()

    with st.spinner("Computing RFM segments and marketing KPIs …"):
        payload = run_marketing_analytics(data_df, table)
        st.session_state.marketing_payload = payload

    with st.spinner("Generating marketing strategy …"):
        report = generate_marketing_strategy(
            data_df, table, payload,
            business_context=business_context,
            objective=objective,
            budget=budget,
            channels=channels,
            brand_voice=brand_voice,
            forecast_results=forecast_results,
        )
        st.session_state.marketing_report = report

    # Reset downstream artifacts on a fresh run.
    st.session_state.pop("marketing_campaigns", None)
    st.session_state.pop("marketing_messages", None)
    st.rerun()

# ── Display analytics + strategy ────────────────────────────────────────────

payload = st.session_state.get("marketing_payload")
report = st.session_state.get("marketing_report", "")

if payload:
    st.divider()
    st.subheader("Marketing KPIs")
    _render_marketing_kpis(payload.get("marketing_kpis", {}))

    st.subheader("Customer Segments (RFM)")
    _render_segments(payload.get("rfm", {}))

    channels_perf = payload.get("channels", {})
    if channels_perf:
        with st.expander("Channel / dimension performance"):
            for dim, breakdown in channels_perf.items():
                st.markdown(f"**{dim}**")
                st.dataframe(
                    pd.DataFrame(breakdown).T, use_container_width=True
                )

if report:
    st.divider()
    st.subheader("Marketing Strategy")

    mk = (payload or {}).get("marketing_kpis", {})
    rfm = (payload or {}).get("rfm", {})
    kpi_cards: list[tuple[str, str]] = []
    if rfm.get("total_revenue"):
        kpi_cards.append(("Revenue", f"${rfm['total_revenue']:,.0f}"))
    if rfm.get("total_customers"):
        kpi_cards.append(("Customers", f"{rfm['total_customers']:,}"))
    if mk.get("aov"):
        kpi_cards.append(("AOV", f"${mk['aov']:,.2f}"))
    if mk.get("repeat_purchase_rate") is not None:
        kpi_cards.append(("Repeat rate", f"{mk['repeat_purchase_rate']}%"))
    if mk.get("churn_risk_pct") is not None:
        kpi_cards.append(("Churn-risk", f"{mk['churn_risk_pct']}%"))

    meta = build_meta(
        "Marketing Strategist",
        model=st.session_state.get("marketing_model", DEFAULT_MARKETING_MODEL),
        dataset=(table or "In-memory data"),
        extra={"Objective": objective},
    )
    render_report(
        "Marketing Strategy",
        report,
        meta=meta,
        payloads=[payload, forecast_results],
        kpis=kpi_cards,
        subtitle="RFM-driven growth strategy, grounded in your data",
        filename_stem="marketing_strategy",
    )

# ── HITL Step 3: Campaign plan ──────────────────────────────────────────────

if payload and report:
    st.divider()
    st.subheader("Campaign Plan")

    if st.button("Generate Campaign Plan"):
        with st.spinner("Designing campaigns and splitting budget …"):
            plan = generate_campaign_plan(
                payload,
                business_context=business_context,
                objective=objective,
                budget=budget,
                channels=channels,
                brand_voice=brand_voice,
                forecast_results=forecast_results,
            )
            st.session_state.marketing_campaigns = plan
            st.rerun()

    plan = st.session_state.get("marketing_campaigns")
    if plan:
        _render_campaigns(plan)

# ── HITL Step 4: Ad copy generator ──────────────────────────────────────────

if payload and payload.get("rfm", {}).get("available"):
    st.divider()
    st.subheader("Ad Copy Generator")

    rfm = payload["rfm"]
    seg_names = rfm["segment_order"]

    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        copy_segment = st.selectbox("Target segment", seg_names, key="copy_segment")
    with cc2:
        copy_channel = st.selectbox(
            "Channel", ["Email", "SMS", "Paid social", "Push", "Retargeting"],
            key="copy_channel",
        )
    with cc3:
        copy_platform = st.selectbox(
            "Platform", ["Generic", "Instagram", "Facebook", "TikTok",
                         "Google", "Klaviyo / Email"],
            key="copy_platform",
        )
    copy_offer = st.text_input(
        "Offer / hook",
        placeholder="E.g. '15% off your next order' or 'free shipping this week'",
        key="copy_offer",
    )

    if st.button("Write Ad Copy"):
        with st.spinner("Writing copy …"):
            seg_stats = rfm["segments"].get(copy_segment, {})
            copy_out = generate_ad_copy(
                segment=copy_segment,
                channel=copy_channel,
                platform=copy_platform,
                offer=copy_offer,
                brand_voice=brand_voice,
                segment_stats=seg_stats,
                playbook=seg_stats.get("playbook", ""),
            )
            st.session_state.marketing_ad_copy = copy_out
            st.rerun()

    copy_out = st.session_state.get("marketing_ad_copy")
    if copy_out:
        if copy_out.get("error"):
            st.error(f"Copy generation failed: {copy_out['error']}")
        else:
            if copy_out.get("headlines"):
                st.markdown("**Headlines**")
                for h in copy_out["headlines"]:
                    st.markdown(f"- {h}")
            if copy_out.get("primary_text"):
                st.markdown("**Body copy**")
                for b in copy_out["primary_text"]:
                    st.markdown(f"- {b}")
            if copy_out.get("cta"):
                st.markdown(f"**CTAs:** {' · '.join(copy_out['cta'])}")
            cols = st.columns(2)
            if copy_out.get("email_subject"):
                cols[0].markdown(f"**Email subject:** {copy_out['email_subject']}")
            if copy_out.get("email_preview"):
                cols[1].markdown(f"**Preview:** {copy_out['email_preview']}")
            if copy_out.get("sms"):
                st.markdown(f"**SMS:** {copy_out['sms']}")
            if copy_out.get("hashtags"):
                st.markdown(f"**Hashtags:** {' '.join(copy_out['hashtags'])}")
            if copy_out.get("notes"):
                st.caption(copy_out["notes"])

# ── Save to PostgreSQL ──────────────────────────────────────────────────────

if payload and report:
    st.divider()
    if is_pg_configured():
        if st.button("Save marketing run to PostgreSQL"):
            with st.spinner("Saving …"):
                res = save_marketing_run(
                    payload,
                    table_name=table,
                    objective=objective,
                    budget=budget,
                    strategy_report=report,
                    campaign_plan=st.session_state.get("marketing_campaigns"),
                )
            if res.get("status") == "stored":
                st.success(
                    f"Saved `{res['run_id']}` · {res['segments']} segments · "
                    f"{res['campaigns']} campaigns"
                )
            elif res.get("status") == "skipped":
                st.info(res.get("reason", "Storage skipped."))
            else:
                st.error(f"Save failed: {res.get('error')}")
    else:
        st.caption("Configure PostgreSQL (PG_* env vars) to persist marketing runs.")

# ── HITL Step 5: Follow-up chat ─────────────────────────────────────────────

if report:
    st.divider()
    st.subheader("Ask the marketing agent")
    st.caption("Refine the strategy, ask for copy variations, or drill into a segment.")

    messages = st.session_state.setdefault("marketing_messages", [])
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if followup := st.chat_input("Ask about the strategy, segments or campaigns …"):
        messages.append({"role": "user", "content": followup})
        with st.chat_message("user"):
            st.markdown(followup)

        with st.chat_message("assistant"):
            with st.spinner(""):
                try:
                    client = Groq(api_key=os.environ.get("GROQ_API_KEY") or "")
                    schema = df_context(data_df) if data_df is not None else ""
                    sys_prompt = (
                        "You are a senior marketing strategist. The user received a "
                        "marketing strategy grounded in RFM segmentation and KPIs.\n\n"
                        f"## Strategy report\n\n{report}\n\n"
                        "## Marketing analytics (JSON)\n"
                        f"```json\n{json.dumps(payload, indent=2, default=str)}\n```\n\n"
                        "Answer follow-ups using ONLY the report and analytics above. "
                        "Be concrete and reference the relevant numbers. If asked for "
                        "copy, write it ready-to-ship."
                    )
                    user_msg = followup + (f"\n\nData context:\n{schema}" if schema else "")
                    reply = client.chat.completions.create(
                        model=st.session_state.get("marketing_model", DEFAULT_MARKETING_MODEL),
                        messages=[
                            {"role": "system", "content": sys_prompt},
                            *[{"role": m["role"], "content": m["content"]} for m in messages[-6:]],
                            {"role": "user", "content": user_msg},
                        ],
                        temperature=0.5,
                        max_tokens=2048,
                    )
                    answer = reply.choices[0].message.content.strip()
                except Exception as exc:
                    answer = f"Error: {exc}"

            st.markdown(answer)
            messages.append({"role": "assistant", "content": answer})
