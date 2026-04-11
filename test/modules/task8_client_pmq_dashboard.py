from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from modules import task5_correlation_plus


REPORT_PATH = task5_correlation_plus.PUBLISHED_REPORT_PATH


def _load_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    try:
        return pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
    except Exception:
        return pd.DataFrame()


def load_dashboard_payload(path: Path) -> dict:
    return {
        "summary": _load_sheet(path, "summary_data"),
        "counts": _load_sheet(path, "quadrant_counts"),
        "insights": _load_sheet(path, "executive_insights"),
        "candidate_map": _load_sheet(path, "metric_candidate_map"),
        "plot": _load_sheet(path, "quadrant_plot_data"),
        "roster": _load_sheet(path, "executive_roster"),
    }


def summary_lookup(summary_df: pd.DataFrame) -> dict:
    if summary_df.empty or "Metric" not in summary_df.columns or "Value" not in summary_df.columns:
        return {}
    return {
        str(row["Metric"]).strip(): row["Value"]
        for _, row in summary_df.iterrows()
        if str(row.get("Metric", "")).strip()
    }


def render_count_card(title: str, value: int, subtitle: str, accent: str) -> None:
    st.markdown(
        f"""
        <div class="client-pmq-card" style="--card-accent:{accent};">
            <div class="client-pmq-card-title">{title}</div>
            <div class="client-pmq-card-value">{value}</div>
            <div class="client-pmq-card-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def run() -> None:
    st.markdown(
        """
        <style>
        .client-pmq-shell {
            background:
                radial-gradient(circle at top left, rgba(250, 204, 21, 0.20), transparent 28%),
                linear-gradient(135deg, #0f172a 0%, #111827 40%, #1e293b 100%);
            border-radius: 24px;
            padding: 30px 34px;
            margin-bottom: 22px;
            box-shadow: 0 24px 50px rgba(15, 23, 42, 0.18);
            border: 1px solid rgba(255,255,255,0.08);
        }
        .client-pmq-title {
            color: #f8fafc;
            font-size: 34px;
            font-weight: 800;
            letter-spacing: 0.2px;
            margin-bottom: 6px;
            font-family: Georgia, "Times New Roman", serif;
        }
        .client-pmq-subtitle {
            color: #cbd5e1;
            font-size: 15px;
            line-height: 1.65;
            max-width: 900px;
        }
        .client-pmq-pill {
            display: inline-block;
            padding: 8px 14px;
            margin: 0 10px 10px 0;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.10);
            color: #e2e8f0;
            font-size: 13px;
            border: 1px solid rgba(255,255,255,0.08);
        }
        .client-pmq-card {
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            border-radius: 22px;
            border: 1px solid #e2e8f0;
            border-top: 6px solid var(--card-accent);
            padding: 22px 22px 20px 22px;
            box-shadow: 0 16px 30px rgba(15, 23, 42, 0.08);
            min-height: 190px;
        }
        .client-pmq-card-title {
            color: #0f172a;
            font-size: 13px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 14px;
        }
        .client-pmq-card-value {
            color: #0f172a;
            font-size: 44px;
            font-weight: 800;
            line-height: 1;
            margin-bottom: 12px;
            font-family: Georgia, "Times New Roman", serif;
        }
        .client-pmq-card-subtitle {
            color: #475569;
            font-size: 14px;
            line-height: 1.6;
        }
        .client-pmq-panel {
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            border: 1px solid #dbe7f5;
            border-radius: 20px;
            box-shadow: 0 14px 28px rgba(15,23,42,0.06);
            padding: 22px 24px;
            margin-bottom: 18px;
        }
        .client-pmq-panel-title {
            color: #0f172a;
            font-size: 22px;
            font-weight: 800;
            margin-bottom: 6px;
            font-family: Georgia, "Times New Roman", serif;
        }
        .client-pmq-panel-subtitle {
            color: #64748b;
            font-size: 14px;
            line-height: 1.6;
            margin-bottom: 18px;
        }
        .client-pmq-insight {
            background: linear-gradient(180deg, #fffef7 0%, #ffffff 100%);
            border: 1px solid #f1e7c8;
            border-left: 6px solid #c79a2b;
            border-radius: 16px;
            padding: 16px 18px;
            margin-bottom: 12px;
        }
        .client-pmq-insight-headline {
            color: #0f172a;
            font-size: 15px;
            font-weight: 800;
            margin-bottom: 6px;
        }
        .client-pmq-insight-detail {
            color: #475569;
            font-size: 13px;
            line-height: 1.65;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="client-pmq-shell">
            <div class="client-pmq-title">Executive Performance Dashboard</div>
            <div class="client-pmq-subtitle">
                This client-facing dashboard reads only the published PMQ workbook. It is designed for leadership sharing and stays isolated from the operational upload flow.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not REPORT_PATH.exists():
        st.error(f"No published PMQ dashboard workbook was found at {REPORT_PATH}")
        st.caption("Publish the current PMQ report first from Performance Magic Quadrant Plus.")
        return

    payload = load_dashboard_payload(REPORT_PATH)
    summary_df = payload["summary"]
    counts_df = payload["counts"]
    insights_df = payload["insights"]
    candidate_map_df = payload["candidate_map"]
    plot_df = payload["plot"]
    roster_df = payload["roster"]

    if counts_df.empty or roster_df.empty:
        st.error("The published PMQ workbook is missing required sheets. Re-publish the dashboard report from Task 5.")
        return

    summary = summary_lookup(summary_df)
    context_cols = st.columns(4)
    context_items = [
        ("Population", summary.get("Population Scope", "N/A")),
        ("College", summary.get("College Filter", "N/A")),
        ("Batch", summary.get("Batch Filter", "N/A")),
        ("Published File", REPORT_PATH.name),
    ]
    for col, (label, value) in zip(context_cols, context_items):
        with col:
            st.markdown(f"<div class='client-pmq-pill'><b>{label}:</b> {value}</div>", unsafe_allow_html=True)

    st.markdown(f"<div class='client-pmq-pill'><b>Weighted Model:</b> {summary.get('Weighted Model', 'N/A')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='client-pmq-pill'><b>Bucket Basis:</b> {summary.get('Quadrant Bucket Basis', 'N/A')}</div>", unsafe_allow_html=True)

    count_lookup = {
        str(row["Metric"]).strip(): int(pd.to_numeric(row.get("Count"), errors="coerce") or 0)
        for _, row in counts_df.iterrows()
    }
    card_cols = st.columns(4)
    card_meta = [
        ("Deployable Candidates", count_lookup.get("Deployable Candidates", 0), "A and A+ performance-grade students ready for deployment review.", "#10b981"),
        ("Progressing Candidates", count_lookup.get("Progressing Candidates", 0), "B-grade students moving well and closest to the deployable band.", "#7c3aed"),
        ("Basic Competency", count_lookup.get("Basic Competency", 0), "C-grade students who have cleared baseline expectations.", "#0ea5e9"),
        ("Critical Intervention", count_lookup.get("Critical Intervention", 0), "F-grade students requiring immediate management attention.", "#ef4444"),
    ]
    for col, meta in zip(card_cols, card_meta):
        with col:
            render_count_card(*meta)

    chart_col, split_col = st.columns([1.25, 1])
    with chart_col:
        st.markdown("<div class='client-pmq-panel'><div class='client-pmq-panel-title'>Performance Distribution</div><div class='client-pmq-panel-subtitle'>Bucket split from the published PMQ workbook used for management review.</div></div>", unsafe_allow_html=True)
        bar_df = counts_df.copy()
        fig_bar = px.bar(
            bar_df,
            x="Count",
            y="Metric",
            orientation="h",
            color="Metric",
            color_discrete_map={
                "Deployable Candidates": "#10b981",
                "Progressing Candidates": "#7c3aed",
                "Basic Competency": "#0ea5e9",
                "Critical Intervention": "#ef4444",
            },
            text="Count",
        )
        fig_bar.update_layout(
            height=420,
            showlegend=False,
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Candidates",
            yaxis_title="",
        )
        fig_bar.update_traces(textposition="outside")
        st.plotly_chart(fig_bar, use_container_width=True)

    with split_col:
        st.markdown("<div class='client-pmq-panel'><div class='client-pmq-panel-title'>Leadership Insights</div><div class='client-pmq-panel-subtitle'>Narratives carried directly from the final PMQ report.</div></div>", unsafe_allow_html=True)
        if insights_df.empty:
            st.info("No insight sheet was found in the published report.")
        else:
            for _, row in insights_df.head(4).iterrows():
                st.markdown(
                    f"""
                    <div class="client-pmq-insight">
                        <div class="client-pmq-insight-headline">{row.get('Headline', '')}</div>
                        <div class="client-pmq-insight-detail">{row.get('Detail', '')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    if not plot_df.empty:
        st.markdown("<div class='client-pmq-panel'><div class='client-pmq-panel-title'>Performance Map</div><div class='client-pmq-panel-subtitle'>Performance score drives the bucket, while overall score gives leadership context.</div></div>", unsafe_allow_html=True)
        fig_scatter = px.scatter(
            plot_df,
            x="Overall Performance Score",
            y="Performance Score",
            color="Quadrant",
            color_discrete_map=task5_correlation_plus.QUADRANT_COLORS,
            hover_data={
                "Superset ID": True,
                "Candidate Name": True,
                "College": True,
                "Assigned Batch": True,
                "Assessment Composite Score": True,
                "Assignment GitHub Score": True,
                "Attendance %": True,
                "Trainer Feedback Score": True,
                "Overall Performance Score": ':.1f',
                "Performance Score": ':.1f',
            },
            size_max=24,
        )
        fig_scatter.add_hline(y=80, line_width=2, line_dash="dash", line_color="#94a3b8")
        fig_scatter.add_hline(y=70, line_width=2, line_dash="dash", line_color="#94a3b8")
        fig_scatter.add_hline(y=60, line_width=2, line_dash="dash", line_color="#94a3b8")
        fig_scatter.update_layout(
            height=560,
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=16, r=16, t=16, b=16),
            legend_title="",
        )
        fig_scatter.update_xaxes(title="Overall Performance Score", range=[0, 100], showgrid=False)
        fig_scatter.update_yaxes(title="Performance Score", range=[0, 100], showgrid=False)
        st.plotly_chart(fig_scatter, use_container_width=True)

    mapping_col, roster_col = st.columns([0.95, 1.25])
    with mapping_col:
        st.markdown("<div class='client-pmq-panel'><div class='client-pmq-panel-title'>Metric to Candidate Mapping</div><div class='client-pmq-panel-subtitle'>Quick candidate drilldown behind each management metric.</div></div>", unsafe_allow_html=True)
        if candidate_map_df.empty:
            st.info("No candidate mapping data was found in the published report.")
        else:
            for _, row in candidate_map_df.iterrows():
                with st.expander(f"{row.get('Metric', 'Metric')} ({row.get('Count', 0)})", expanded=False):
                    st.text(str(row.get("Candidate Details", "No candidates in this segment.")))

    with roster_col:
        st.markdown("<div class='client-pmq-panel'><div class='client-pmq-panel-title'>Executive Roster</div><div class='client-pmq-panel-subtitle'>Filtered leadership roster from the published PMQ workbook.</div></div>", unsafe_allow_html=True)
        bucket_options = ["All Buckets"] + sorted(roster_df["Quadrant"].dropna().astype(str).unique().tolist())
        filter_col1, filter_col2 = st.columns([1, 1.2])
        with filter_col1:
            selected_bucket = st.selectbox("Bucket", bucket_options, key="task8_bucket_filter")
        with filter_col2:
            search_text = st.text_input("Search Candidate / Superset ID", key="task8_search_text").strip().lower()
        display_df = roster_df.copy()
        if selected_bucket != "All Buckets":
            display_df = display_df[display_df["Quadrant"] == selected_bucket].copy()
        if search_text:
            display_df = display_df[
                display_df["Candidate Name"].astype(str).str.lower().str.contains(search_text, na=False)
                | display_df["Superset ID"].astype(str).str.lower().str.contains(search_text, na=False)
            ].copy()
        st.dataframe(display_df.fillna("N/A"), use_container_width=True, hide_index=True)

    with open(REPORT_PATH, "rb") as report_handle:
        st.download_button(
            "Download Published PMQ Workbook",
            data=report_handle.read(),
            file_name=REPORT_PATH.name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
