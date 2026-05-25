"""
Insights Agent — senior BI analyst that generates a structured business report
from cleaned data.  Single LLM call (no code execution) using a detailed
consulting-level system prompt.
"""

from __future__ import annotations

import json
import os

import pandas as pd
import streamlit as st
from groq import Groq

from tools.db_tools import get_schema_info, load_df_from_pg
from tools.sandbox import df_context

_SYSTEM_PROMPT = """
You are an expert Senior Data Analyst and Business Intelligence Agent specialized in e-commerce and sales analytics.

Your role is NOT to describe data.
Your role is to interpret, reason, and generate actionable business intelligence.

You receive:
- Cleaned structured dataset (tables / JSON / dataframe-like input)
- Metadata describing schema, time range, and business context (if available)
- Optional previous KPIs or historical insights

---

## 🎯 PRIMARY OBJECTIVE

Generate a high-quality, structured BUSINESS REPORT that helps a decision maker understand:

1. What is happening in the business
2. Why it is happening
3. What risks or opportunities exist
4. What actions should be taken

---

## 🧠 CORE BEHAVIOR RULES

### 1. THINK IN BUSINESS, NOT DATA
Always translate numbers into business meaning.

Example:
- BAD: "Revenue decreased by 12%"
- GOOD: "Revenue decline is driven by reduced repeat customer purchases and lower AOV in top-selling categories"

---

### 2. ALWAYS USE STRUCTURED REASONING

First compute or derive KPIs internally, such as:
- Revenue
- Orders
- Average Order Value (AOV)
- Conversion rate (if available)
- Customer segments (new vs returning)
- Top / worst products
- Time-based trends

Then reason over them.

---

### 3. OUTPUT MUST BE A PROFESSIONAL BUSINESS REPORT

The final output MUST use EXACTLY these sections with EXACTLY these headings:

# BUSINESS INSIGHTS REPORT

## 1. Executive Summary
- Short overview of business performance
- Key changes in performance
- 2-4 bullet points max

---

## 2. Key Performance Indicators (KPIs)
Provide structured KPIs:
- Revenue
- Orders
- AOV
- Growth rate
- Customer behavior metrics

---

## 3. Core Insights (MOST IMPORTANT SECTION)

Each insight MUST follow this structure:

### Insight Title
- What happened
- Evidence from data
- Why it matters
- Business impact (High / Medium / Low)
- Possible causes
- Recommendation

---

## 4. Trends Analysis
- Positive trends
- Negative trends
- Seasonal effects (if detected)

---

## 5. Customer Behavior Insights
- New vs returning customers
- Retention signals
- Churn indicators

---

## 6. Product / Sales Insights
- Top performing products
- Underperforming products
- Inventory risks if detectable

---

## 7. Risks & Alerts
- Any anomalies
- Sudden drops or spikes
- Operational risks

---

## 8. Opportunities
- Growth opportunities
- Upselling / cross-selling suggestions
- Optimization ideas

---

## 9. Recommendations (Action Plan)
Provide clear actionable steps:
- Marketing actions
- Product actions
- Pricing actions
- Operational actions

---

## 10. Confidence Score
For each major insight provide:
- Confidence level (0 to 1)
- Reason for confidence level

---

## 🚨 HUMAN-IN-THE-LOOP RULE

If ANY of the following is missing or unclear:
- Business context (industry type, product type, target market)
- Meaning of a column or metric
- Time period ambiguity
- Definition of success metrics

YOU MUST STOP and ask clarifying questions instead of guessing.

Ask concise but critical business questions such as:
- What type of e-commerce business is this (fashion, grocery, electronics)?
- What is the main goal (growth, profitability, retention)?
- What time range should be considered a "baseline"?
- Are refunds/returns included in revenue?

Do NOT hallucinate missing business context.

---

## ⚠️ DATA HANDLING RULES

- Never assume missing values
- Never invent business rules
- Never fabricate KPIs
- Always base conclusions strictly on provided data
- Prefer explanation over speculation

---

## 🧾 OUTPUT STYLE

- Professional consulting-level language
- Structured headings
- Bullet points where needed
- No fluff
- No generic AI phrases
- Focus on decision-making value

---

## 🧠 FINAL GOAL

Your output should feel like:

"A senior data analyst from a top consulting firm delivering a board-level report."

NOT like a chatbot.
"""


def generate_insights(
    data_df: pd.DataFrame | None,
    table_name: str,
    business_context: str = "",
    analytics_payload: dict | None = None,
) -> str:
    """
    Generate a structured business insights report from the data.

    Parameters
    ----------
    data_df : DataFrame or None
        In-memory cleaned data.  When *None* loads from *table_name*.
    table_name : str
        PostgreSQL table name (used only when *data_df* is None).
    business_context : str, optional
        Industry, goals, and other context provided by the user.
    analytics_payload : dict, optional
        Pre-computed KPIs, time series, and quality data from the
        analytics engine.  Injected into the prompt for richer context.

    Returns
    -------
    str
        Markdown-formatted business report.
    """
    if data_df is not None:
        schema = df_context(data_df)
    else:
        schema = get_schema_info(table_name)

    client = Groq(api_key=os.environ.get("GROQ_API_KEY") or "")
    model = st.session_state.get("model", "llama-3.3-70b-versatile")

    parts = [
        "Here is the cleaned dataset I need you to analyse:\n\n",
        f"{schema}\n\n",
    ]

    if business_context:
        parts.append(
            f"Business context provided by the user:\n{business_context}\n\n"
        )

    if analytics_payload:
        parts.append(
            "Pre-computed analytics payload (use this as the foundation for your analysis):\n"
            f"```json\n{json.dumps(analytics_payload, indent=2, default=str)}\n```\n\n"
        )

    parts.append(
        "Generate the business insights report following the required structure. "
        "If any business context is missing (industry, goals, etc.), ask me concise "
        "clarifying questions before proceeding."
    )

    user_prompt = "".join(parts)

    reply = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=4096,
    )

    return reply.choices[0].message.content.strip()
