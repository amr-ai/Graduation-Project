"""
Marketing Agent page — Category Performance & Customer Segmentation.

Displays:
  - Category performance analysis with metric cards and product details
  - Customer segmentation with RFM-based segments and recommended actions
  - Natural language insights
  - Follow-up chat for drill-down (HITL pattern)
"""

from __future__ import annotations

import json
import os
import re

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from groq import Groq

from agents.marketing.category_performance import compute_category_performance
from agents.marketing.customer_segmentation import compute_customer_segmentation
from tools.sandbox import df_context, run_analysis


# ── Data helpers ───────────────────────────────────────────────────────

def _data_available() -> bool:
    return (
        st.session_state.clean_df is not None
        or (st.session_state.pg_stored and st.session_state.pg_table)
    )


def _get_df():
    """Return the in-memory DataFrame (or load from PG)."""
    if st.session_state.clean_df is not None:
        return st.session_state.clean_df
    if st.session_state.pg_stored and st.session_state.pg_table:
        from tools.db_tools import load_df_from_pg
        return load_df_from_pg(st.session_state.pg_table)
    return None


# ── Page header ────────────────────────────────────────────────────────

st.title("🎯 Marketing Agent")
st.markdown(
    "Category & product performance analysis and customer segmentation "
    "powered by RFM analysis."
)

if not _data_available():
    st.info(
        "No cleaned data found. You can **upload a CSV** below to analyse directly, "
        "or go to **Data Cleaning** first."
    )
    uploaded = st.file_uploader(
        "Upload a CSV file",
        type=["csv"],
        key="marketing_upload",
    )
    if uploaded is not None:
        import pandas as pd
        data_df = pd.read_csv(uploaded)
        st.session_state.clean_df = data_df
        st.success(f"Loaded **{len(data_df)}** rows × **{len(data_df.columns)}** columns.")
        st.rerun()
    else:
        st.stop()

data_df = _get_df()
if data_df is None:
    st.error("Failed to load data.")
    st.stop()


# ── Run analyses ───────────────────────────────────────────────────────

col_run, _ = st.columns([1, 3])
with col_run:
    run_clicked = st.button(
        "Run Marketing Analysis",
        type="primary",
        use_container_width=True,
    )

if run_clicked:
    with st.spinner("Analysing category performance …"):
        cat_result = compute_category_performance(data_df)
        st.session_state.marketing_category = cat_result

    with st.spinner("Segmenting customers …"):
        seg_result = compute_customer_segmentation(data_df)
        st.session_state.marketing_segmentation = seg_result

    st.rerun()


# ═══════════════════════════════════════════════════════════════════════
#  CATEGORY PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════

cat_result = st.session_state.get("marketing_category")

if cat_result and not cat_result.get("available"):
    st.divider()
    st.warning(
        f"⚠️ **Category analysis unavailable:** {cat_result.get('reason', 'Unknown reason')}. "
        "Make sure your data has a category column and a numeric revenue/price column."
    )
elif cat_result and cat_result.get("available"):
    st.divider()
    st.header("📊 Category & Product Performance")

    categories = cat_result["categories"]
    overall = cat_result["overall"]

    # ── Summary cards ──────────────────────────────────────────────────
    cols = st.columns(4)
    with cols[0]:
        st.metric("Total Revenue", f"${overall['total_revenue']:,.0f}")
    with cols[1]:
        qty = overall.get("total_quantity")
        st.metric("Total Quantity", f"{qty:,}" if qty is not None else "—")
    with cols[2]:
        st.metric("Categories", overall["category_count"])
    with cols[3]:
        st.metric("Best Category", overall["best_category"] or "—")

    # ── Category comparison chart ──────────────────────────────────────
    cat_data = [
        {
            "Category": name,
            "Revenue": data["revenue"],
            "Share (%)": data["revenue_share_pct"],
            "Trend": data.get("trend_direction", "—").capitalize(),
        }
        for name, data in categories.items()
    ]
    if cat_data:
        import pandas as pd

        cat_df = pd.DataFrame(cat_data).sort_values("Revenue", ascending=False)

        fig_bar = px.bar(
            cat_df,
            x="Category",
            y="Revenue",
            color="Trend",
            text="Share (%)",
            color_discrete_map={
                "Growth": "#2ecc71",
                "Decline": "#e74c3c",
                "Stable": "#3498db",
            },
            title="Revenue by Category",
        )
        fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_bar.update_layout(yaxis_title="Revenue ($)", xaxis_title="")
        st.plotly_chart(fig_bar, use_container_width=True, config={"displaylogo": False})

    # ── Per-category details ───────────────────────────────────────────
    st.subheader("Category Details")

    for name, data in sorted(
        categories.items(), key=lambda x: x[1]["revenue"], reverse=True
    ):
        with st.expander(
            f"**{name}** — ${data['revenue']:,.2f} "
            f"({data['revenue_share_pct']}% share, "
            f"trend: {data.get('trend_direction', '—')})",
            expanded=False,
        ):
            detail_cols = st.columns(4)
            with detail_cols[0]:
                st.metric("Revenue", f"${data['revenue']:,.2f}")
            with detail_cols[1]:
                qty_val = data.get("quantity")
                st.metric(
                    "Quantity",
                    f"{qty_val:,}" if qty_val is not None else "—",
                )
            with detail_cols[2]:
                margin = data.get("margin_pct")
                st.metric(
                    "Margin",
                    f"{margin}%" if margin is not None else "N/A",
                )
            with detail_cols[3]:
                st.metric("Seasonality", f"{data.get('seasonality_score', 0):.2f}")

            # Top products
            if data.get("top_products"):
                st.caption("**Top Products**")
                import pandas as pd

                tp_df = pd.DataFrame(data["top_products"])
                st.dataframe(tp_df, use_container_width=True, hide_index=True)

            # Bottom products
            if data.get("bottom_products"):
                st.caption("**Bottom Products**")
                bp_df = pd.DataFrame(data["bottom_products"])
                st.dataframe(bp_df, use_container_width=True, hide_index=True)

            # Monthly trend chart
            monthly = {
                k: v
                for k, v in data.get("monthly_trend", {}).items()
                if not k.startswith("_")
            }
            if monthly:
                trend_df = pd.DataFrame(
                    list(monthly.items()), columns=["Month", "Revenue"]
                )
                fig_trend = px.line(
                    trend_df,
                    x="Month",
                    y="Revenue",
                    title=f"{name} — Monthly Revenue Trend",
                    markers=True,
                )
                fig_trend.update_layout(
                    yaxis_title="Revenue ($)", xaxis_title="Month"
                )
                st.plotly_chart(
                    fig_trend, use_container_width=True, config={"displaylogo": False}
                )

    # ── Natural language insights ──────────────────────────────────────
    st.subheader("Insights")
    nl = cat_result.get("natural_language_summary", "")
    if nl:
        st.info(nl)
    for insight in cat_result.get("insights", []):
        st.markdown(f"- {insight}")


# ═══════════════════════════════════════════════════════════════════════
#  CUSTOMER SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════

seg_result = st.session_state.get("marketing_segmentation")

if seg_result and not seg_result.get("available"):
    st.divider()
    st.warning(
        f"⚠️ **Customer segmentation unavailable:** {seg_result.get('reason', 'Unknown reason')}. "
        "Make sure your data has a customer identifier column and a numeric revenue/price column."
    )
elif seg_result and seg_result.get("available"):
    st.divider()
    st.header("👥 Customer Segmentation (RFM)")

    segments = seg_result["segments"]
    rfm_summary = seg_result["rfm_summary"]
    total_customers = seg_result["total_customers"]

    # ── Summary cards ──────────────────────────────────────────────────
    sum_cols = st.columns(4)
    with sum_cols[0]:
        st.metric("Total Customers", total_customers)
    with sum_cols[1]:
        st.metric("Avg Recency", f"{rfm_summary['avg_recency']:.0f} days")
    with sum_cols[2]:
        st.metric("Avg Frequency", f"{rfm_summary['avg_frequency']:.1f}")
    with sum_cols[3]:
        st.metric("Avg Monetary", f"${rfm_summary['avg_monetary']:,.2f}")

    # ── Segment distribution chart ─────────────────────────────────────
    seg_names = []
    seg_counts = []
    seg_colors = {
        "VIP Customers": "#f1c40f",
        "Regular Customers": "#3498db",
        "At-Risk Customers": "#e74c3c",
        "New Customers": "#2ecc71",
    }

    for name in ["VIP Customers", "Regular Customers", "At-Risk Customers", "New Customers"]:
        seg = segments.get(name, {})
        if seg.get("count", 0) > 0:
            seg_names.append(name)
            seg_counts.append(seg["count"])

    if seg_names:
        fig_pie = go.Figure(
            data=[
                go.Pie(
                    labels=seg_names,
                    values=seg_counts,
                    hole=0.45,
                    marker=dict(
                        colors=[seg_colors.get(n, "#95a5a6") for n in seg_names]
                    ),
                    textinfo="label+percent",
                    textposition="outside",
                )
            ]
        )
        fig_pie.update_layout(
            title="Customer Segment Distribution",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig_pie, use_container_width=True, config={"displaylogo": False})

    # ── Per-segment detail cards ───────────────────────────────────────
    st.subheader("Segment Details")

    for seg_name in ["VIP Customers", "Regular Customers", "At-Risk Customers", "New Customers"]:
        seg = segments.get(seg_name, {})
        if seg.get("count", 0) == 0:
            continue

        emoji_map = {
            "VIP Customers": "👑",
            "Regular Customers": "🛒",
            "At-Risk Customers": "⚠️",
            "New Customers": "🆕",
        }
        emoji = emoji_map.get(seg_name, "📌")

        with st.expander(
            f"**{emoji} {seg_name}** — {seg['count']} customers ({seg['pct']}%)",
            expanded=(seg_name in ("VIP Customers", "At-Risk Customers")),
        ):
            det_cols = st.columns(4)
            with det_cols[0]:
                st.metric("Count", seg["count"])
            with det_cols[1]:
                st.metric("AOV", f"${seg['avg_order_value']:,.2f}")
            with det_cols[2]:
                st.metric("Avg Frequency", f"{seg['avg_frequency']:.1f}")
            with det_cols[3]:
                st.metric("Avg Recency", f"{seg['avg_recency_days']:.0f} days")

            st.markdown(f"**💰 Avg Total Spend:** ${seg['avg_monetary']:,.2f}")
            st.markdown(f"**📈 Avg RFM Score:** {seg['avg_rfm_score']:.1f} / 12")

            st.success(f"**Recommended Action:** {seg['recommended_action']}")

    # ── Natural language insights ──────────────────────────────────────
    st.subheader("Insights")
    nl_seg = seg_result.get("natural_language_summary", "")
    if nl_seg:
        st.info(nl_seg)
    for insight in seg_result.get("insights", []):
        st.markdown(f"- {insight}")


# ═══════════════════════════════════════════════════════════════════════
#  JSON EXPORT
# ═══════════════════════════════════════════════════════════════════════

_has_results = (
    (cat_result and cat_result.get("available"))
    or (seg_result and seg_result.get("available"))
)

if _has_results:
    st.divider()
    st.subheader("📥 Export")
    exp_cols = st.columns(2)

    if cat_result and cat_result.get("available"):
        with exp_cols[0]:
            st.download_button(
                "Download Category Report (JSON)",
                data=json.dumps(cat_result, indent=2, default=str),
                file_name="category_performance.json",
                mime="application/json",
                use_container_width=True,
            )

    if seg_result and seg_result.get("available"):
        with exp_cols[1]:
            st.download_button(
                "Download Segmentation Report (JSON)",
                data=json.dumps(seg_result, indent=2, default=str),
                file_name="customer_segmentation.json",
                mime="application/json",
                use_container_width=True,
            )


# ═══════════════════════════════════════════════════════════════════════
#  FOLLOW-UP CHAT (HITL)
# ═══════════════════════════════════════════════════════════════════════

if _has_results:
    st.divider()
    st.subheader("💬 Ask a follow-up")
    st.caption("Drill into category or segment details, request deeper analysis, etc.")

    marketing_messages = st.session_state.setdefault("marketing_messages", [])

    for msg in marketing_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if followup := st.chat_input("Ask about the marketing analysis …"):
        marketing_messages.append({"role": "user", "content": followup})
        with st.chat_message("user"):
            st.markdown(followup)

        with st.chat_message("assistant"):
            with st.spinner(""):
                try:
                    client = Groq(api_key=os.environ.get("GROQ_API_KEY") or "")

                    # Build context
                    context_parts = [
                        "You are a senior marketing analyst. The user ran category "
                        "performance and customer segmentation analysis. Here are the results:\n\n",
                    ]
                    if cat_result and cat_result.get("available"):
                        context_parts.append(
                            "## Category Performance\n```json\n"
                            f"{json.dumps(cat_result, indent=2, default=str)}\n```\n\n"
                        )
                    if seg_result and seg_result.get("available"):
                        context_parts.append(
                            "## Customer Segmentation\n```json\n"
                            f"{json.dumps(seg_result, indent=2, default=str)}\n```\n\n"
                        )
                    context_parts.append(
                        "Answer concisely based on the analysis. If the user asks "
                        "for additional computation, write Python code with `df` to compute it."
                    )

                    sys_prompt = "".join(context_parts)
                    schema = df_context(data_df)
                    user_msg = followup
                    if schema:
                        user_msg += f"\n\nDataFrame context:\n{schema}"

                    reply = client.chat.completions.create(
                        model=st.session_state.get("model", "llama-3.3-70b-versatile"),
                        messages=[
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": user_msg},
                        ],
                        temperature=0.3,
                        max_tokens=2048,
                    )
                    answer = reply.choices[0].message.content.strip()
                except Exception as exc:
                    answer = f"Error: {exc}"

            # Execute any code blocks in the response
            code_blocks = re.findall(r"```python\n(.*?)```", answer, re.DOTALL)
            if code_blocks and data_df is not None:
                for code in code_blocks:
                    result = run_analysis(code, data_df)
                    if result["error"]:
                        answer += f"\n\n**Execution error:**\n```\n{result['error']}\n```"
                    else:
                        if result["output"]:
                            answer += f"\n\n**Output:**\n```\n{result['output']}\n```"
                        if "fig" in result and result["fig"] is not None:
                            st.plotly_chart(
                                result["fig"],
                                use_container_width=True,
                                config={"displaylogo": False},
                            )

            st.markdown(answer)
            marketing_messages.append({"role": "assistant", "content": answer})
