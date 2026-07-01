"""
Shared report presentation for Smart Analyst pages.

Every agent that produces an LLM narrative (Insights, Marketing, Churn,
Forecasting) renders it through :func:`render_report` so the output is
consistent and professional:

  * a titled header band with run metadata (agent, model, dataset, timestamp);
  * an optional KPI strip;
  * a **grounding badge** — a trust signal showing how many figures in the
    narrative were traced back to the deterministic payload;
  * the report body;
  * one-click export to styled, print-ready **HTML** and raw **Markdown**.

This is the single place presentation is defined, so improving it lifts every
report at once.
"""

from __future__ import annotations

from datetime import datetime

import streamlit as st

from .export import build_html_report
from .grounding import GroundingResult, check_grounding


def build_meta(
    agent: str,
    *,
    model: str = "",
    dataset: str = "",
    rows: int | None = None,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """Assemble the metadata line shown in the header and the export."""
    meta: dict[str, str] = {"Agent": agent}
    if model:
        meta["Model"] = model
    if dataset:
        meta["Dataset"] = dataset
    if rows is not None:
        meta["Rows"] = f"{rows:,}"
    meta["Generated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    if extra:
        meta.update(extra)
    return meta


def render_grounding_badge(grounding: GroundingResult) -> None:
    """Render the faithfulness badge + a drill-down on any unmatched figures."""
    if grounding.total == 0:
        return
    if grounding.status == "clean":
        st.success(f"✓ Grounded — {grounding.label}", icon="✅")
        return
    st.warning(f"⚠ Grounding — {grounding.label}", icon="⚠️")
    with st.expander(f"Figures not matched to source data ({len(grounding.unverified)})"):
        st.caption(
            "These figures in the narrative could not be matched to a computed "
            "value. They may be reasonable derivations, external benchmarks, or "
            "model errors — worth a quick check before circulating."
        )
        for c in grounding.unverified:
            st.markdown(f"- `{c.raw}`  ·  _{c.kind}_")


def render_report(
    title: str,
    report_md: str,
    *,
    meta: dict[str, str] | None = None,
    payloads: tuple | list = (),
    kpis: list[tuple[str, str]] | None = None,
    subtitle: str = "",
    filename_stem: str = "report",
    show_grounding: bool = True,
) -> GroundingResult | None:
    """
    Render a generated report with header, grounding badge, body and exports.

    Parameters
    ----------
    title, report_md : str
        Report heading and the markdown body from the agent.
    meta : dict, optional
        Metadata line (see :func:`build_meta`).
    payloads : sequence
        Deterministic payload(s) the report should be grounded against.
    kpis : list[(label, value)], optional
        Headline figures shown as a strip and embedded in the HTML export.
    subtitle : str, optional
        One-line descriptor under the title.
    filename_stem : str
        Base name for the downloaded files.
    show_grounding : bool
        Whether to run and display the grounding check.

    Returns
    -------
    GroundingResult or None
    """
    if not report_md:
        return None

    if meta:
        st.caption(" · ".join(f"**{k}:** {v}" for k, v in meta.items()))

    grounding: GroundingResult | None = None
    if show_grounding and payloads:
        grounding = check_grounding(report_md, *payloads)
        render_grounding_badge(grounding)

    st.markdown(report_md)

    html = build_html_report(
        title, report_md,
        subtitle=subtitle, meta=meta, kpis=kpis, grounding=grounding,
    )

    col_html, col_md, _ = st.columns([1, 1, 2])
    with col_html:
        st.download_button(
            "⬇ Report (HTML / PDF)",
            data=html.encode("utf-8"),
            file_name=f"{filename_stem}.html",
            mime="text/html",
            use_container_width=True,
            help="Styled, print-ready report. Open it and Save as PDF for a "
                 "board-ready deliverable.",
            key=f"dl_html_{filename_stem}",
        )
    with col_md:
        st.download_button(
            "⬇ Markdown",
            data=report_md.encode("utf-8"),
            file_name=f"{filename_stem}.md",
            mime="text/markdown",
            use_container_width=True,
            key=f"dl_md_{filename_stem}",
        )

    return grounding
