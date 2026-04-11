from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


REPORT_PATH = Path(__file__).resolve().parents[1] / "published_reports" / "pmq_client_dashboard.xlsx"
QUADRANT_ORDER = [
    "Deployable Candidates",
    "Progressing Candidates",
    "Basic Competency",
    "Critical Intervention",
]
QUADRANT_COLORS = {
    "Deployable Candidates": "#0f9d58",
    "Progressing Candidates": "#c48a12",
    "Basic Competency": "#1d4ed8",
    "Critical Intervention": "#d93025",
}
QUADRANT_DESCRIPTIONS = {
    "Deployable Candidates": "A and A+ performance-grade students ready for deployment conversations.",
    "Progressing Candidates": "B-grade students building momentum and moving toward deployable readiness.",
    "Basic Competency": "C-grade students who have crossed the baseline but still need guided uplift.",
    "Critical Intervention": "F-grade students requiring immediate management attention and structured intervention.",
}


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


def render_signal_card(title: str, value: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="client-pmq-signal-card">
            <div class="client-pmq-signal-title">{title}</div>
            <div class="client-pmq-signal-value">{value}</div>
            <div class="client-pmq-signal-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_dynamic_insights(filtered_df: pd.DataFrame) -> list[dict]:
    if filtered_df.empty:
        return []

    total = len(filtered_df)
    counts = filtered_df["Quadrant"].value_counts()
    avg_performance = pd.to_numeric(filtered_df["Performance Score"], errors="coerce").mean()
    avg_overall = pd.to_numeric(filtered_df["Overall Performance Score"], errors="coerce").mean()
    avg_assignment = pd.to_numeric(filtered_df["Assignment GitHub Score"], errors="coerce").mean()
    avg_assessment = pd.to_numeric(filtered_df["Assessment Composite Score"], errors="coerce").mean()
    deployable_share = counts.get("Deployable Candidates", 0) / total
    progressing_share = counts.get("Progressing Candidates", 0) / total
    critical_share = counts.get("Critical Intervention", 0) / total

    insights = [
        {
            "headline": f"{deployable_share:.0%} of the current cohort is in the deployable band.",
            "detail": "These students are currently clearing the strongest performance bucket and should anchor leadership-ready deployment discussions.",
        },
        {
            "headline": f"Average performance score is {avg_performance:.1f}.",
            "detail": "This is the bucket-driving score, built from assessment composite and assignment GitHub performance.",
        },
        {
            "headline": f"Average assessment composite is {avg_assessment:.1f} and average assignment score is {avg_assignment:.1f}.",
            "detail": "This split helps leadership see whether the cohort is stronger on structured assessments or on assignment execution.",
        },
    ]

    if critical_share > 0:
        insights.append(
            {
                "headline": f"{critical_share:.0%} of the cohort is in critical intervention.",
                "detail": "This group should move into the highest-priority review track with direct academic and management follow-up.",
            }
        )
    else:
        insights.append(
            {
                "headline": f"{progressing_share:.0%} of the cohort is in the progressing band, with average overall score at {avg_overall:.1f}.",
                "detail": "This segment has the strongest near-term movement opportunity into the deployable band with targeted coaching.",
            }
        )
    return insights


def filter_roster(roster_df: pd.DataFrame, college: str, batch: str, quadrant: str, search_text: str) -> pd.DataFrame:
    filtered_df = roster_df.copy()
    if college != "All Colleges":
        filtered_df = filtered_df[filtered_df["College"].astype(str) == college].copy()
    if batch != "All Batches":
        filtered_df = filtered_df[filtered_df["Assigned Batch"].astype(str) == batch].copy()
    if quadrant != "All Buckets":
        filtered_df = filtered_df[filtered_df["Quadrant"].astype(str) == quadrant].copy()
    if search_text:
        needle = search_text.strip().lower()
        filtered_df = filtered_df[
            filtered_df["Candidate Name"].astype(str).str.lower().str.contains(needle, na=False)
            | filtered_df["Superset ID"].astype(str).str.lower().str.contains(needle, na=False)
        ].copy()
    return filtered_df


def build_metric_candidate_map(filtered_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bucket in QUADRANT_ORDER:
        bucket_df = filtered_df[filtered_df["Quadrant"].astype(str) == bucket].copy()
        details = [
            f"{str(row['Superset ID']).strip()} - {str(row['Candidate Name']).strip()}"
            for _, row in bucket_df.iterrows()
        ]
        rows.append(
            {
                "Metric": bucket,
                "Count": int(len(bucket_df)),
                "Candidate Details": "\n".join(details) if details else "No candidates in this segment.",
            }
        )
    return pd.DataFrame(rows)


def run() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 2.2rem;
            max-width: 1380px;
        }
        .client-pmq-hero {
            background:
                radial-gradient(circle at top left, rgba(196, 138, 18, 0.30), transparent 24%),
                radial-gradient(circle at bottom right, rgba(15, 157, 88, 0.18), transparent 20%),
                linear-gradient(135deg, #111827 0%, #172033 46%, #0f172a 100%);
            border-radius: 28px;
            padding: 32px 34px 28px 34px;
            margin-bottom: 20px;
            box-shadow: 0 26px 54px rgba(15, 23, 42, 0.20);
            border: 1px solid rgba(255,255,255,0.08);
        }
        .client-pmq-title {
            color: #f8fafc;
            font-size: 42px;
            font-weight: 800;
            line-height: 1.05;
            margin-bottom: 10px;
            font-family: Georgia, "Times New Roman", serif;
        }
        .client-pmq-subtitle {
            color: #dbe4ef;
            font-size: 15px;
            line-height: 1.72;
            max-width: 940px;
        }
        .client-pmq-pill {
            display: inline-block;
            padding: 10px 15px;
            margin: 10px 10px 0 0;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.09);
            color: #f8fafc;
            font-size: 13px;
            border: 1px solid rgba(255,255,255,0.08);
            backdrop-filter: blur(5px);
        }
        .client-pmq-filter-panel {
            background: linear-gradient(180deg, #ffffff 0%, #fbfcfe 100%);
            border: 1px solid #d9e5f1;
            border-radius: 22px;
            padding: 18px 18px 12px 18px;
            box-shadow: 0 18px 32px rgba(15,23,42,0.05);
            margin-bottom: 20px;
        }
        .client-pmq-filter-title {
            color: #0f172a;
            font-size: 18px;
            font-weight: 800;
            margin-bottom: 4px;
        }
        .client-pmq-filter-subtitle {
            color: #64748b;
            font-size: 13px;
            margin-bottom: 14px;
        }
        .client-pmq-card {
            background: linear-gradient(180deg, #ffffff 0%, #f9fbfd 100%);
            border-radius: 24px;
            border: 1px solid #e2e8f0;
            border-top: 6px solid var(--card-accent);
            padding: 24px 24px 20px 24px;
            box-shadow: 0 18px 34px rgba(15, 23, 42, 0.08);
            min-height: 206px;
        }
        .client-pmq-card-title {
            color: #0f172a;
            font-size: 13px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 1.1px;
            margin-bottom: 14px;
        }
        .client-pmq-card-value {
            color: #0f172a;
            font-size: 50px;
            font-weight: 800;
            line-height: 1;
            margin-bottom: 12px;
            font-family: Georgia, "Times New Roman", serif;
        }
        .client-pmq-card-subtitle {
            color: #475569;
            font-size: 14px;
            line-height: 1.68;
        }
        .client-pmq-signal-card {
            background: linear-gradient(180deg, #ffffff 0%, #fbfcfe 100%);
            border-radius: 18px;
            border: 1px solid #dbe7f5;
            padding: 18px 18px 16px 18px;
            box-shadow: 0 14px 26px rgba(15, 23, 42, 0.05);
            min-height: 136px;
        }
        .client-pmq-signal-title {
            color: #64748b;
            font-size: 12px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }
        .client-pmq-signal-value {
            color: #0f172a;
            font-size: 30px;
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: 8px;
            font-family: Georgia, "Times New Roman", serif;
        }
        .client-pmq-signal-subtitle {
            color: #64748b;
            font-size: 13px;
            line-height: 1.6;
        }
        .client-pmq-panel {
            background: linear-gradient(180deg, #ffffff 0%, #fbfcfe 100%);
            border: 1px solid #dbe7f5;
            border-radius: 22px;
            box-shadow: 0 18px 32px rgba(15,23,42,0.06);
            padding: 22px 24px;
            margin-bottom: 18px;
        }
        .client-pmq-panel-title {
            color: #0f172a;
            font-size: 24px;
            font-weight: 800;
            margin-bottom: 6px;
            font-family: Georgia, "Times New Roman", serif;
        }
        .client-pmq-panel-subtitle {
            color: #64748b;
            font-size: 14px;
            line-height: 1.65;
            margin-bottom: 18px;
        }
        .client-pmq-insight {
            background: linear-gradient(180deg, #fffef8 0%, #ffffff 100%);
            border: 1px solid #f2e4bf;
            border-left: 6px solid #c48a12;
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
            line-height: 1.68;
        }
        div[data-testid="stMetric"] {
            background: linear-gradient(180deg, #ffffff 0%, #fbfcfe 100%);
            border: 1px solid #dbe7f5;
            border-radius: 16px;
            padding: 16px 18px 10px 18px;
            box-shadow: 0 14px 24px rgba(15,23,42,0.04);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="client-pmq-hero">
            <div class="client-pmq-title">Executive Performance Dashboard</div>
            <div class="client-pmq-subtitle">
                Leadership view built from the final published PMQ workbook. This dashboard is designed for executive review, strategic cohort scanning, and clear candidate segmentation without exposing the operational ingestion workflow.
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
    roster_df = payload["roster"]

    if roster_df.empty:
        st.error("The published PMQ workbook is missing the executive roster sheet. Re-publish the dashboard report from Task 5.")
        return

    for column in [
        "Assessment Composite Score",
        "Assignment GitHub Score",
        "Performance Score",
        "Overall Performance Score",
        "Attendance %",
        "Trainer Feedback Score",
    ]:
        if column in roster_df.columns:
            roster_df[column] = pd.to_numeric(roster_df[column], errors="coerce")

    summary = summary_lookup(summary_df)
    colleges = ["All Colleges"] + sorted(roster_df["College"].dropna().astype(str).unique().tolist())

    st.markdown(
        """
        <div class="client-pmq-filter-panel">
            <div class="client-pmq-filter-title">Executive Filters</div>
            <div class="client-pmq-filter-subtitle">Refine the leadership view by college, batch, bucket, or candidate. All cards and charts below react to the current selection.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    filter_cols = st.columns([1, 1, 1, 1.2])
    with filter_cols[0]:
        selected_college = st.selectbox("College", colleges, key="task8_college_filter")
    scoped_batch_df = roster_df if selected_college == "All Colleges" else roster_df[roster_df["College"].astype(str) == selected_college]
    batches = ["All Batches"] + sorted(scoped_batch_df["Assigned Batch"].dropna().astype(str).unique().tolist())
    with filter_cols[1]:
        selected_batch = st.selectbox("Batch", batches, key="task8_batch_filter")
    with filter_cols[2]:
        selected_quadrant = st.selectbox("Performance Bucket", ["All Buckets"] + QUADRANT_ORDER, key="task8_bucket_filter")
    with filter_cols[3]:
        search_text = st.text_input("Search Candidate / Superset ID", key="task8_search_text").strip()

    filtered_df = filter_roster(roster_df, selected_college, selected_batch, selected_quadrant, search_text)

    st.markdown(f"<div class='client-pmq-pill'><b>Population:</b> {summary.get('Population Scope', 'N/A')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='client-pmq-pill'><b>Weighted Model:</b> {summary.get('Weighted Model', 'N/A')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='client-pmq-pill'><b>Bucket Basis:</b> {summary.get('Quadrant Bucket Basis', 'N/A')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='client-pmq-pill'><b>Current View:</b> {selected_college} | {selected_batch} | {selected_quadrant}</div>", unsafe_allow_html=True)

    if filtered_df.empty:
        st.warning("No candidates match the current selection.")
        with open(REPORT_PATH, "rb") as report_handle:
            st.download_button(
                "Download Published PMQ Workbook",
                data=report_handle.read(),
                file_name=REPORT_PATH.name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        return

    counts = filtered_df["Quadrant"].value_counts()
    card_cols = st.columns(4)
    for col, bucket in zip(card_cols, QUADRANT_ORDER):
        with col:
            render_count_card(
                bucket,
                int(counts.get(bucket, 0)),
                QUADRANT_DESCRIPTIONS[bucket],
                QUADRANT_COLORS[bucket],
            )

    signal_cols = st.columns(4)
    avg_performance = pd.to_numeric(filtered_df["Performance Score"], errors="coerce").mean()
    avg_overall = pd.to_numeric(filtered_df["Overall Performance Score"], errors="coerce").mean()
    avg_assessment = pd.to_numeric(filtered_df["Assessment Composite Score"], errors="coerce").mean()
    avg_assignment = pd.to_numeric(filtered_df["Assignment GitHub Score"], errors="coerce").mean()
    signal_data = [
        ("Current View Size", f"{len(filtered_df)}", "Students currently included after filters."),
        ("Avg Performance", f"{avg_performance:.1f}", "Bucket-driving technical performance score."),
        ("Avg Overall Score", f"{avg_overall:.1f}", "Leadership-facing 50/35/10/5 weighted score."),
        ("Assessment / Assignment", f"{avg_assessment:.1f} / {avg_assignment:.1f}", "Average assessment composite versus assignment execution."),
    ]
    for col, (title, value, subtitle) in zip(signal_cols, signal_data):
        with col:
            render_signal_card(title, value, subtitle)

    overview_tab, insights_tab, roster_tab = st.tabs(["Overview", "Leadership Narrative", "Roster & Drilldown"])

    with overview_tab:
        left_col, right_col = st.columns([1.05, 1])
        with left_col:
            st.markdown(
                """
                <div class="client-pmq-panel">
                    <div class="client-pmq-panel-title">Performance Bucket Split</div>
                    <div class="client-pmq-panel-subtitle">High-level bucket composition for the current view.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            bar_df = pd.DataFrame(
                [{"Metric": bucket, "Count": int(counts.get(bucket, 0))} for bucket in QUADRANT_ORDER]
            )
            fig_bar = px.bar(
                bar_df,
                x="Count",
                y="Metric",
                orientation="h",
                color="Metric",
                color_discrete_map=QUADRANT_COLORS,
                text="Count",
            )
            fig_bar.update_layout(
                height=430,
                showlegend=False,
                plot_bgcolor="white",
                paper_bgcolor="white",
                margin=dict(l=8, r=8, t=8, b=8),
                xaxis_title="Candidates",
                yaxis_title="",
            )
            fig_bar.update_traces(textposition="outside")
            st.plotly_chart(fig_bar, use_container_width=True)

        with right_col:
            st.markdown(
                """
                <div class="client-pmq-panel">
                    <div class="client-pmq-panel-title">Score Distribution Map</div>
                    <div class="client-pmq-panel-subtitle">Performance score drives the bucket, while overall score shows the full management-weighted outcome.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            scatter_cols = [
                column
                for column in [
                    "Superset ID",
                    "Candidate Name",
                    "College",
                    "Assigned Batch",
                    "Quadrant",
                    "Performance Score",
                    "Overall Performance Score",
                    "Assessment Composite Score",
                    "Assignment GitHub Score",
                    "Attendance %",
                    "Trainer Feedback Score",
                ]
                if column in filtered_df.columns
            ]
            plot_df = filtered_df[scatter_cols].copy()
            fig_scatter = px.scatter(
                plot_df,
                x="Overall Performance Score",
                y="Performance Score",
                color="Quadrant",
                color_discrete_map=QUADRANT_COLORS,
                hover_data={
                    "Superset ID": True,
                    "Candidate Name": True,
                    "College": True,
                    "Assigned Batch": True,
                    "Assessment Composite Score": ':.1f',
                    "Assignment GitHub Score": ':.1f',
                    "Attendance %": ':.1f',
                    "Trainer Feedback Score": ':.1f',
                    "Overall Performance Score": ':.1f',
                    "Performance Score": ':.1f',
                },
            )
            fig_scatter.add_hline(y=80, line_width=2, line_dash="dash", line_color="#a0aec0")
            fig_scatter.add_hline(y=70, line_width=2, line_dash="dash", line_color="#a0aec0")
            fig_scatter.add_hline(y=60, line_width=2, line_dash="dash", line_color="#a0aec0")
            fig_scatter.update_layout(
                height=430,
                plot_bgcolor="white",
                paper_bgcolor="white",
                margin=dict(l=8, r=8, t=8, b=8),
                legend_title="",
            )
            fig_scatter.update_xaxes(title="Overall Performance Score", range=[0, 100], showgrid=False)
            fig_scatter.update_yaxes(title="Performance Score", range=[0, 100], showgrid=False)
            st.plotly_chart(fig_scatter, use_container_width=True)

        lower_left, lower_right = st.columns([1.05, 1])
        with lower_left:
            st.markdown(
                """
                <div class="client-pmq-panel">
                    <div class="client-pmq-panel-title">Top 10 Performers</div>
                    <div class="client-pmq-panel-subtitle">Highest overall performers in the current filtered view.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            top_df = filtered_df.sort_values(by="Overall Performance Score", ascending=False).head(10).copy()
            fig_top = px.bar(
                top_df,
                x="Overall Performance Score",
                y="Candidate Name",
                orientation="h",
                color="Quadrant",
                color_discrete_map=QUADRANT_COLORS,
                text="Overall Performance Score",
            )
            fig_top.update_layout(
                height=420,
                plot_bgcolor="white",
                paper_bgcolor="white",
                margin=dict(l=8, r=8, t=8, b=8),
                legend_title="",
                yaxis_title="",
            )
            fig_top.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            st.plotly_chart(fig_top, use_container_width=True)

        with lower_right:
            st.markdown(
                """
                <div class="client-pmq-panel">
                    <div class="client-pmq-panel-title">Bottom 10 Needing Focus</div>
                    <div class="client-pmq-panel-subtitle">Lowest overall performers in the current filtered view.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            bottom_df = filtered_df.sort_values(by="Overall Performance Score", ascending=True).head(10).copy()
            fig_bottom = px.bar(
                bottom_df,
                x="Overall Performance Score",
                y="Candidate Name",
                orientation="h",
                color="Quadrant",
                color_discrete_map=QUADRANT_COLORS,
                text="Overall Performance Score",
            )
            fig_bottom.update_layout(
                height=420,
                plot_bgcolor="white",
                paper_bgcolor="white",
                margin=dict(l=8, r=8, t=8, b=8),
                legend_title="",
                yaxis_title="",
            )
            fig_bottom.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            st.plotly_chart(fig_bottom, use_container_width=True)

    with insights_tab:
        insight_col, map_col = st.columns([1, 1.05])
        with insight_col:
            st.markdown(
                """
                <div class="client-pmq-panel">
                    <div class="client-pmq-panel-title">Leadership Narrative</div>
                    <div class="client-pmq-panel-subtitle">Automatically generated observations from the current selection.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            for insight in build_dynamic_insights(filtered_df):
                st.markdown(
                    f"""
                    <div class="client-pmq-insight">
                        <div class="client-pmq-insight-headline">{insight["headline"]}</div>
                        <div class="client-pmq-insight-detail">{insight["detail"]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with map_col:
            st.markdown(
                """
                <div class="client-pmq-panel">
                    <div class="client-pmq-panel-title">Bucket-to-Candidate Mapping</div>
                    <div class="client-pmq-panel-subtitle">Expand any bucket to see the candidates driving that metric.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            metric_map_df = build_metric_candidate_map(filtered_df)
            for _, row in metric_map_df.iterrows():
                with st.expander(f"{row['Metric']} ({row['Count']})", expanded=False):
                    st.text(row["Candidate Details"])

    with roster_tab:
        st.markdown(
            """
            <div class="client-pmq-panel">
                <div class="client-pmq-panel-title">Executive Roster</div>
                <div class="client-pmq-panel-subtitle">Detailed management roster for the current selection.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(filtered_df.fillna("N/A"), use_container_width=True, hide_index=True)

    with open(REPORT_PATH, "rb") as report_handle:
        st.download_button(
            "Download Published PMQ Workbook",
            data=report_handle.read(),
            file_name=REPORT_PATH.name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
