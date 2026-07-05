"""Streamlit dashboard for Tokenomist.

Run with::

    streamlit run src/tokenomist/dashboard.py

Upload one or more conversation logs (or load the bundled samples) and the
dashboard renders the comparison table plus charts for cost, latency, token
usage, and convergence efficiency. Streamlit is an optional dependency; the
core library and CLI work without it.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict

try:
    import pandas as pd
    import streamlit as st
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "The dashboard needs streamlit and pandas. Install with:\n"
        "    pip install 'tokenomist[dashboard]'"
    ) from exc

from tokenomist.analyzer import analyze_many
from tokenomist.parsers import load_conversations, parse_data
from tokenomist.report import rank_reports

_SAMPLE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "samples"
)


def _reports_dataframe(reports) -> pd.DataFrame:
    rows = [r.summary_dict() for r in reports]
    df = pd.DataFrame(rows)
    return df


def _trace_dataframe(reports) -> pd.DataFrame:
    rows = []
    for rep in reports:
        for tr in rep.trace:
            rows.append(asdict(tr))
    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(page_title="Tokenomist", layout="wide")
    st.title("🏁 Tokenomist")
    st.caption(
        "Compare how different AI agents solve the same task — cost, speed, "
        "accuracy, and reasoning efficiency."
    )

    with st.sidebar:
        st.header("Input")
        uploaded = st.file_uploader(
            "Upload conversation logs (JSON)", type="json", accept_multiple_files=True
        )
        use_samples = st.checkbox("Load bundled samples", value=not uploaded)

    conversations = []
    if uploaded:
        for file in uploaded:
            data = json.load(file)
            conversations.append(parse_data(data, source_path=file.name))
    if use_samples and os.path.isdir(_SAMPLE_DIR):
        conversations.extend(load_conversations([_SAMPLE_DIR]))

    if not conversations:
        st.info("Upload logs or enable the bundled samples to begin.")
        return

    reports = rank_reports(analyze_many(conversations))
    df = _reports_dataframe(reports)

    winner = reports[0]
    cols = st.columns(4)
    cols[0].metric("Agents compared", len(reports))
    cols[1].metric("Most efficient", winner.agent, f"{winner.convergence_efficiency:.3f}")
    priced = [r for r in reports if r.cost_estimate_usd is not None]
    cheapest = min(priced, key=lambda r: r.cost_estimate_usd).agent if priced else "n/a"
    cols[2].metric("Cheapest", cheapest)
    cols[3].metric("Fastest", min(reports, key=lambda r: r.latency_estimate_ms).agent)

    st.subheader("Comparison")
    st.dataframe(df, use_container_width=True)

    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.subheader("Cost (USD)")
        st.bar_chart(df.set_index("agent")["cost_estimate_usd"])
        st.subheader("Latency (ms)")
        st.bar_chart(df.set_index("agent")["latency_estimate_ms"])
    with chart_cols[1]:
        st.subheader("Convergence efficiency")
        st.bar_chart(df.set_index("agent")["convergence_efficiency"])
        st.subheader("Tokens to success")
        st.bar_chart(df.set_index("agent")["tokens_to_success"].fillna(0))

    st.subheader("Per-turn traffic trace")
    trace_df = _trace_dataframe(reports)
    agents = trace_df["agent"].unique().tolist()
    chosen = st.selectbox("Agent", agents)
    sel = trace_df[trace_df["agent"] == chosen]
    st.area_chart(sel.set_index("turn_index")["cumulative_cost_usd"])
    st.dataframe(sel, use_container_width=True)


if __name__ == "__main__":  # pragma: no cover
    main()
