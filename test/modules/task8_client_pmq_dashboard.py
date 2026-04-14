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
    "Deployable Candidates": "#12b981",
    "Progressing Candidates": "#d4a017",
    "Basic Competency": "#3b82f6",
    "Critical Intervention": "#ef4444",
}
QUADRANT_DESCRIPTIONS = {
    "Deployable Candidates": "Students already in the A or A+ performance band and ready for deployment conversations.",
    "Progressing Candidates": "Students in the B band who are moving well but still need performance uplift to enter the top bucket.",
    "Basic Competency": "Students in the C band who have crossed the baseline and need structured technical improvement.",
    "Critical Intervention": "Students in the F band who require immediate academic and management follow-up.",
}
GRADE_ORDER = ["A+", "A", "B", "C", "F"]
TRACKER_COLUMNS = [
    "Assessment Composite Score",
    "Assignment GitHub Score",
    "Performance Score",
    "Overall Performance Score",
    "Attendance %",
    "Trainer Feedback Score",
    "Assessment GitHub Score",
    "Top Brains Score",
    "Assessment GitHub Weeks",
    "Assignment GitHub Weeks",
    "Top Brains Weeks",
]


def _load_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    try:
        return pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
    except Exception:
        return pd.DataFrame()


def _load_payload(path: Path) -> dict:
    return {
        "dashboard": _load_sheet(path, "dashboard_summary"),
        "summary": _load_sheet(path, "summary_data"),
        "counts": _load_sheet(path, "quadrant_counts"),
        "integrity": _load_sheet(path, "signal_integrity"),
        "insights": _load_sheet(path, "executive_insights"),
        "candidate_map": _load_sheet(path, "metric_candidate_map"),
        "plot": _load_sheet(path, "quadrant_plot_data"),
        "roster": _load_sheet(path, "executive_roster"),
        "jecrc_batch": _load_sheet(path, "jecrc_batch_performance"),
    }


def _summary_lookup(summary_df: pd.DataFrame) -> dict:
    if summary_df.empty or {"Metric", "Value"} - set(summary_df.columns):
        return {}
    return {
        str(row["Metric"]).strip(): row["Value"]
        for _, row in summary_df.iterrows()
        if str(row.get("Metric", "")).strip()
    }


def _format_value(value: object, digits: int = 1) -> str:
    if pd.isna(value):
        return "N/A"
    if isinstance(value, str):
        return value
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:.{digits}f}"


def _as_numeric(value: object) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return float(numeric)


def _resolve_average(primary_value: object, *fallback_values: object) -> float | None:
    ordered_values = (primary_value,) + fallback_values
    resolved = None
    for value in ordered_values:
        numeric = _as_numeric(value)
        if numeric is None:
            continue
        if resolved is None:
            resolved = numeric
        if numeric > 0:
            return numeric
    return resolved


def _coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = df.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def _filter_frame(df: pd.DataFrame, college: str, batch: str, quadrant: str, search_text: str) -> pd.DataFrame:
    filtered_df = df.copy()
    if "College" in filtered_df.columns and college != "All Colleges":
        filtered_df = filtered_df[filtered_df["College"].astype(str) == college].copy()
    if "Assigned Batch" in filtered_df.columns and batch != "All Batches":
        filtered_df = filtered_df[filtered_df["Assigned Batch"].astype(str) == batch].copy()
    if "Quadrant" in filtered_df.columns and quadrant != "All Buckets":
        filtered_df = filtered_df[filtered_df["Quadrant"].astype(str) == quadrant].copy()
    if search_text:
        needle = search_text.strip().lower()
        name_match = filtered_df.get("Candidate Name", pd.Series(index=filtered_df.index, dtype="object")).astype(str).str.lower().str.contains(needle, na=False)
        id_match = filtered_df.get("Superset ID", pd.Series(index=filtered_df.index, dtype="object")).astype(str).str.lower().str.contains(needle, na=False)
        filtered_df = filtered_df[name_match | id_match].copy()
    return filtered_df


def _build_dynamic_insights(filtered_df: pd.DataFrame) -> list[dict]:
    if filtered_df.empty:
        return []

    counts = filtered_df["Quadrant"].value_counts()
    total = len(filtered_df)
    deployable_share = counts.get("Deployable Candidates", 0) / total
    progressing_share = counts.get("Progressing Candidates", 0) / total
    basic_share = counts.get("Basic Competency", 0) / total
    avg_perf = filtered_df["Performance Score"].mean()
    avg_overall = filtered_df["Overall Performance Score"].mean()
    avg_assessment = filtered_df["Assessment Composite Score"].mean()
    avg_assignment = filtered_df["Assignment GitHub Score"].mean()
    avg_attendance = filtered_df["Attendance %"].mean()

    insights = [
        {
            "headline": f"{deployable_share:.0%} of the current view is in the deployable band.",
            "detail": "This is the strongest talent segment in the current selection and is the most ready for immediate business conversations.",
        },
        {
            "headline": f"Average performance score is {avg_perf:.1f}, while overall score stands at {avg_overall:.1f}.",
            "detail": "Performance score drives the bucket, while overall score reflects the full weighted model after attendance and trainer feedback are added.",
        },
        {
            "headline": f"Assessment composite averages {avg_assessment:.1f} and assignment GitHub averages {avg_assignment:.1f}.",
            "detail": "This comparison shows whether the cohort is stronger in structured assessments or in delivery-oriented assignment execution.",
        },
    ]

    if avg_assignment + 5 < avg_assessment:
        insights.append(
            {
                "headline": "Assignments are trailing the structured assessment signal.",
                "detail": "The cohort is performing better in composite assessments than in assignment execution, which suggests a delivery-conversion gap.",
            }
        )
    elif avg_assignment > avg_assessment + 5:
        insights.append(
            {
                "headline": "Assignment execution is outperforming the assessment composite.",
                "detail": "Students are delivering well on assignment work, which may indicate strong execution despite lower assessment consistency.",
            }
        )
    else:
        insights.append(
            {
                "headline": f"{progressing_share:.0%} of the view sits in the progressing band and {basic_share:.0%} in the baseline band.",
                "detail": "This is the near-term movement pool for focused mentoring, targeted reviews, and transition into the deployable category.",
            }
        )

    insights.append(
        {
            "headline": f"Average attendance is {avg_attendance:.1f}%.",
            "detail": "Attendance and trainer feedback remain part of the overall score and help signal delivery reliability across the cohort.",
        }
    )
    return insights


def _build_metric_candidate_map(filtered_df: pd.DataFrame) -> pd.DataFrame:
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


def _build_integrity_view(filtered_df: pd.DataFrame, integrity_df: pd.DataFrame) -> pd.DataFrame:
    published = {}
    if not integrity_df.empty and {"Metric", "Value"} <= set(integrity_df.columns):
        published = {str(row["Metric"]).strip(): row["Value"] for _, row in integrity_df.iterrows()}

    total = len(filtered_df)
    if total == 0:
        return pd.DataFrame()

    def coverage(series_name: str, week_name: str | None = None) -> int:
        if week_name and week_name in filtered_df.columns:
            return int((pd.to_numeric(filtered_df[week_name], errors="coerce").fillna(0) > 0).sum())
        if series_name in filtered_df.columns:
            return int(filtered_df[series_name].notna().sum())
        return 0

    rows = [
        {
            "Signal": "Assessment GitHub",
            "Current View Coverage": coverage("Assessment GitHub Score", "Assessment GitHub Weeks"),
            "Published Missing": int(float(published.get("Missing Assessment GitHub", 0) or 0)),
            "Published ID Matches": int(float(published.get("Assessment GitHub ID Matches", 0) or 0)),
        },
        {
            "Signal": "Assignment GitHub",
            "Current View Coverage": coverage("Assignment GitHub Score", "Assignment GitHub Weeks"),
            "Published Missing": int(float(published.get("Missing Assignment GitHub", 0) or 0)),
            "Published ID Matches": int(float(published.get("Assignment GitHub ID Matches", 0) or 0)),
        },
        {
            "Signal": "Top Brains",
            "Current View Coverage": coverage("Top Brains Score", "Top Brains Weeks"),
            "Published Missing": int(float(published.get("Missing Top Brains Signal", 0) or 0)),
            "Published ID Matches": int(float(published.get("Top Brains ID Matches", 0) or 0)),
        },
        {
            "Signal": "Trainer Feedback",
            "Current View Coverage": coverage("Trainer Feedback Score"),
            "Published Missing": int(float(published.get("Missing Feedback Signal", 0) or 0)),
            "Published ID Matches": int(float(published.get("Feedback ID Matches", 0) or 0)),
        },
    ]
    integrity_view = pd.DataFrame(rows)
    integrity_view["Coverage %"] = integrity_view["Current View Coverage"] / total * 100
    return integrity_view


def _render_primary_card(title: str, value: int, share: float, description: str, accent: str) -> None:
    st.markdown(
        f"""
        <div class="pmq-primary-card" style="--accent:{accent};">
            <div class="pmq-card-kicker">{title}</div>
            <div class="pmq-card-value">{value}</div>
            <div class="pmq-card-share">{share:.0%} of current view</div>
            <div class="pmq-card-detail">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_signal_card(title: str, value: str, detail: str) -> None:
    st.markdown(
        f"""
        <div class="pmq-signal-card">
            <div class="pmq-signal-title">{title}</div>
            <div class="pmq-signal-value">{value}</div>
            <div class="pmq-signal-detail">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_context_chip(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="pmq-chip">
            <span class="pmq-chip-label">{label}</span>
            <span class="pmq-chip-value">{value}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def run() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1420px;
            padding-top: 1.4rem;
            padding-bottom: 2.6rem;
        }
        [data-testid="stHorizontalBlock"] > div:has(.pmq-primary-card),
        [data-testid="stHorizontalBlock"] > div:has(.pmq-signal-card) {
            height: 100%;
        }
        .pmq-hero {
            background:
                radial-gradient(circle at top left, rgba(251, 191, 36, 0.18), transparent 24%),
                radial-gradient(circle at bottom right, rgba(45, 212, 191, 0.14), transparent 20%),
                linear-gradient(135deg, #0f172a 0%, #172033 46%, #1e293b 100%);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 30px;
            padding: 32px 34px 26px 34px;
            box-shadow: 0 30px 55px rgba(15, 23, 42, 0.20);
            margin-bottom: 18px;
        }
        .pmq-hero-title {
            color: #f8fafc;
            font-size: 40px;
            font-weight: 800;
            line-height: 1.05;
            margin-bottom: 10px;
            font-family: Georgia, "Times New Roman", serif;
        }
        .pmq-hero-subtitle {
            color: #d7e1ec;
            font-size: 15px;
            line-height: 1.75;
            max-width: 980px;
            margin-bottom: 18px;
        }
        .pmq-chip {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            margin: 8px 10px 0 0;
            padding: 10px 14px;
            border-radius: 999px;
            background: rgba(255,255,255,0.09);
            border: 1px solid rgba(255,255,255,0.08);
            color: #f8fafc;
        }
        .pmq-chip-label {
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #bfdbfe;
            font-weight: 700;
        }
        .pmq-chip-value {
            font-size: 13px;
            color: #ffffff;
            font-weight: 700;
        }
        .pmq-filter-wrap {
            background: linear-gradient(180deg, #ffffff 0%, #fbfcfe 100%);
            border: 1px solid #dbe4ef;
            border-radius: 24px;
            padding: 18px 18px 10px 18px;
            box-shadow: 0 18px 30px rgba(15, 23, 42, 0.05);
            margin-bottom: 18px;
        }
        .pmq-filter-title {
            color: #0f172a;
            font-size: 18px;
            font-weight: 800;
            margin-bottom: 4px;
        }
        .pmq-filter-subtitle {
            color: #64748b;
            font-size: 13px;
            line-height: 1.6;
            margin-bottom: 14px;
        }
        .pmq-primary-card {
            background: linear-gradient(180deg, #ffffff 0%, #f9fbfd 100%);
            border: 1px solid #e2e8f0;
            border-top: 6px solid var(--accent);
            border-radius: 24px;
            min-height: 218px;
            padding: 22px 22px 18px 22px;
            box-shadow: 0 18px 34px rgba(15, 23, 42, 0.08);
        }
        .pmq-card-kicker {
            color: #0f172a;
            font-size: 13px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 1.1px;
            margin-bottom: 14px;
        }
        .pmq-card-value {
            color: #0f172a;
            font-size: 52px;
            font-weight: 800;
            line-height: 1;
            margin-bottom: 10px;
            font-family: Georgia, "Times New Roman", serif;
        }
        .pmq-card-share {
            color: #0f172a;
            font-size: 14px;
            font-weight: 800;
            margin-bottom: 10px;
        }
        .pmq-card-detail {
            color: #475569;
            font-size: 14px;
            line-height: 1.68;
        }
        .pmq-signal-card {
            background: linear-gradient(180deg, #ffffff 0%, #fbfcfe 100%);
            border: 1px solid #dce8f6;
            border-radius: 18px;
            min-height: 146px;
            padding: 18px;
            box-shadow: 0 14px 24px rgba(15, 23, 42, 0.05);
        }
        .pmq-signal-title {
            color: #64748b;
            font-size: 12px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }
        .pmq-signal-value {
            color: #0f172a;
            font-size: 30px;
            font-weight: 800;
            line-height: 1.05;
            margin-bottom: 8px;
            font-family: Georgia, "Times New Roman", serif;
        }
        .pmq-signal-detail {
            color: #64748b;
            font-size: 13px;
            line-height: 1.58;
        }
        .pmq-panel {
            background: linear-gradient(180deg, #ffffff 0%, #fbfcfe 100%);
            border: 1px solid #dbe7f5;
            border-radius: 22px;
            box-shadow: 0 18px 32px rgba(15, 23, 42, 0.05);
            padding: 22px 24px;
            margin-bottom: 18px;
        }
        .pmq-panel-title {
            color: #0f172a;
            font-size: 24px;
            font-weight: 800;
            margin-bottom: 6px;
            font-family: Georgia, "Times New Roman", serif;
        }
        .pmq-panel-subtitle {
            color: #64748b;
            font-size: 14px;
            line-height: 1.65;
            margin-bottom: 16px;
        }
        .pmq-mini-note {
            background: linear-gradient(180deg, #fffef8 0%, #ffffff 100%);
            border: 1px solid #f0e0b9;
            border-left: 6px solid #d4a017;
            border-radius: 16px;
            padding: 16px 18px;
            margin-bottom: 12px;
        }
        .pmq-mini-note h4 {
            color: #0f172a;
            font-size: 15px;
            font-weight: 800;
            margin: 0 0 6px 0;
        }
        .pmq-mini-note p {
            color: #475569;
            font-size: 13px;
            line-height: 1.68;
            margin: 0;
        }
        .pmq-integrity-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 12px;
        }
        .pmq-integrity-card {
            background: #ffffff;
            border: 1px solid #dbe7f5;
            border-radius: 16px;
            padding: 16px;
        }
        .pmq-integrity-title {
            font-size: 14px;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 10px;
        }
        .pmq-integrity-row {
            display: flex;
            justify-content: space-between;
            gap: 10px;
            margin-bottom: 6px;
            color: #475569;
            font-size: 13px;
        }
        .pmq-badge {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 999px;
            background: #e8fff3;
            color: #047857;
            font-size: 12px;
            font-weight: 800;
            margin-top: 6px;
        }
        div[data-testid="stTabs"] button {
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="pmq-hero">
            <div class="pmq-hero-title">Executive Performance Dashboard</div>
            <div class="pmq-hero-subtitle">
                A leadership-facing view of the final PMQ workbook. This page translates the published report into a clearer performance story using the summary, integrity, bucket, insight, plotting, and roster sheets.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not REPORT_PATH.exists():
        st.error(f"No published PMQ workbook was found at {REPORT_PATH}")
        st.caption("Publish the final PMQ dashboard workbook first.")
        return

    payload = _load_payload(REPORT_PATH)
    summary_df = payload["summary"]
    counts_df = payload["counts"]
    integrity_df = payload["integrity"]
    insights_df = payload["insights"]
    jecrc_batch_df = payload["jecrc_batch"]
    plot_df = _coerce_numeric(payload["plot"], TRACKER_COLUMNS)
    roster_df = _coerce_numeric(payload["roster"], TRACKER_COLUMNS)

    if roster_df.empty:
        st.error("The published PMQ workbook is missing the executive roster sheet. Re-publish the dashboard report from Performance Magic Quadrant Plus.")
        return

    summary = _summary_lookup(summary_df)
    colleges = ["All Colleges"] + sorted(roster_df["College"].dropna().astype(str).unique().tolist())

    st.markdown(
        """
        <div class="pmq-filter-wrap">
            <div class="pmq-filter-title">View Controls</div>
            <div class="pmq-filter-subtitle">Use the filters below to refine the leadership view. Cards, insights, charts, and drilldowns all respond to the current selection.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    filter_cols = st.columns([1, 1, 1, 1.15])
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

    filtered_roster = _filter_frame(roster_df, selected_college, selected_batch, selected_quadrant, search_text)
    filtered_plot = _filter_frame(plot_df, selected_college, selected_batch, selected_quadrant, search_text)

    chip_cols = st.columns(4)
    with chip_cols[0]:
        _render_context_chip("Current View", f"{selected_college} | {selected_batch}")
    with chip_cols[1]:
        _render_context_chip("Population", str(len(filtered_roster)))
    with chip_cols[2]:
        _render_context_chip("Weighted Model", str(summary.get("Weighted Model", "Published workbook")))
    with chip_cols[3]:
        _render_context_chip("Bucket Basis", str(summary.get("Quadrant Bucket Basis", "Performance grade bands")))

    if filtered_roster.empty:
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

    counts = filtered_roster["Quadrant"].value_counts()
    total = len(filtered_roster)
    card_cols = st.columns(4)
    for col, bucket in zip(card_cols, QUADRANT_ORDER):
        with col:
            _render_primary_card(
                bucket,
                int(counts.get(bucket, 0)),
                (counts.get(bucket, 0) / total) if total else 0,
                QUADRANT_DESCRIPTIONS[bucket],
                QUADRANT_COLORS[bucket],
            )

    avg_assessment = filtered_roster["Assessment Composite Score"].mean()
    avg_assignment = filtered_roster["Assignment GitHub Score"].mean()
    avg_performance = filtered_roster["Performance Score"].mean()
    avg_overall = filtered_roster["Overall Performance Score"].mean()
    avg_attendance = _resolve_average(
        filtered_roster["Attendance %"].mean(),
        filtered_plot["Attendance %"].mean() if "Attendance %" in filtered_plot.columns else None,
        summary.get("Average Attendance %"),
    )
    avg_feedback = _resolve_average(
        filtered_roster["Trainer Feedback Score"].mean(),
        filtered_plot["Trainer Feedback Score"].mean() if "Trainer Feedback Score" in filtered_plot.columns else None,
        summary.get("Average Trainer Feedback Score"),
    )

    signal_cols = st.columns(6)
    signal_data = [
        ("Current View Size", f"{len(filtered_roster)}", "Students included after filters."),
        ("Avg Performance", _format_value(avg_performance), "The score that drives the performance bucket."),
        ("Avg Overall", _format_value(avg_overall), "Full weighted outcome from the published PMQ model."),
        ("Assessment Composite", _format_value(avg_assessment), "Average of GitHub assessment and Top Brains signals."),
        ("Assignment GitHub", _format_value(avg_assignment), "Average assignment execution score in the current view."),
        ("Attendance / Feedback", f"{_format_value(avg_attendance)} / {_format_value(avg_feedback)}", "Attendance and trainer signals in the current selection."),
    ]
    for col, (title, value, detail) in zip(signal_cols, signal_data):
        with col:
            _render_signal_card(title, value, detail)

    overview_tab, attendance_tab, signals_tab, candidates_tab = st.tabs(
        ["Overview", "Batch Attendance", "Signals & Insights", "Candidate Drilldown"]
    )

    with overview_tab:
        top_left, top_right = st.columns([1.05, 1])
        with top_left:
            st.markdown(
                """
                <div class="pmq-panel">
                    <div class="pmq-panel-title">Bucket Composition</div>
                    <div class="pmq-panel-subtitle">Current-view readiness split across the four published performance buckets.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            bucket_df = pd.DataFrame(
                [{"Metric": bucket, "Count": int(counts.get(bucket, 0))} for bucket in QUADRANT_ORDER]
            )
            fig_donut = px.pie(
                bucket_df,
                names="Metric",
                values="Count",
                hole=0.62,
                color="Metric",
                color_discrete_map=QUADRANT_COLORS,
            )
            fig_donut.update_traces(textposition="inside", textinfo="percent+label")
            fig_donut.update_layout(
                height=420,
                paper_bgcolor="white",
                plot_bgcolor="white",
                margin=dict(l=10, r=10, t=10, b=10),
                legend_title="",
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        with top_right:
            st.markdown(
                """
                <div class="pmq-panel">
                    <div class="pmq-panel-title">Score Driver Profile</div>
                    <div class="pmq-panel-subtitle">How the core PMQ score components are currently behaving in the selected view.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            driver_df = pd.DataFrame(
                [
                    {"Signal": "Assessment Composite", "Average Score": avg_assessment},
                    {"Signal": "Assignment GitHub", "Average Score": avg_assignment},
                    {"Signal": "Attendance", "Average Score": avg_attendance},
                    {"Signal": "Trainer Feedback", "Average Score": avg_feedback},
                    {"Signal": "Performance Score", "Average Score": avg_performance},
                    {"Signal": "Overall Score", "Average Score": avg_overall},
                ]
            )
            fig_driver = px.bar(
                driver_df,
                x="Average Score",
                y="Signal",
                orientation="h",
                text="Average Score",
                color="Signal",
                color_discrete_sequence=["#123b6d", "#1c64f2", "#12b981", "#d4a017", "#7c3aed", "#0f172a"],
            )
            fig_driver.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig_driver.update_layout(
                height=420,
                showlegend=False,
                paper_bgcolor="white",
                plot_bgcolor="white",
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="Average Score",
                yaxis_title="",
            )
            fig_driver.update_xaxes(range=[0, 100])
            st.plotly_chart(fig_driver, use_container_width=True)

        lower_left, lower_right = st.columns([1, 1.05])
        with lower_left:
            st.markdown(
                """
                <div class="pmq-panel">
                    <div class="pmq-panel-title">Performance Grade Mix</div>
                    <div class="pmq-panel-subtitle">Grade-level view of the bucket-driving performance score.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            grade_df = (
                filtered_roster["Performance Grade"]
                .astype(str)
                .value_counts()
                .reindex(GRADE_ORDER, fill_value=0)
                .rename_axis("Performance Grade")
                .reset_index(name="Students")
            )
            fig_grade = px.bar(
                grade_df,
                x="Performance Grade",
                y="Students",
                color="Performance Grade",
                text="Students",
                color_discrete_map={
                    "A+": "#0f9d58",
                    "A": "#34d399",
                    "B": "#d4a017",
                    "C": "#3b82f6",
                    "F": "#ef4444",
                },
            )
            fig_grade.update_layout(
                height=380,
                showlegend=False,
                paper_bgcolor="white",
                plot_bgcolor="white",
                margin=dict(l=10, r=10, t=10, b=10),
            )
            fig_grade.update_traces(textposition="outside")
            st.plotly_chart(fig_grade, use_container_width=True)

        with lower_right:
            st.markdown(
                """
                <div class="pmq-panel">
                    <div class="pmq-panel-title">Performance Ladder</div>
                    <div class="pmq-panel-subtitle">Top current-view candidates ranked by performance score, which directly drives the bucket assignment.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            ladder_df = filtered_plot.sort_values(by="Performance Score", ascending=False).head(12).copy()
            fig_ladder = px.bar(
                ladder_df,
                x="Performance Score",
                y="Candidate Name",
                orientation="h",
                color="Quadrant",
                text="Performance Score",
                color_discrete_map=QUADRANT_COLORS,
                hover_data={
                    "Superset ID": True,
                    "Assigned Batch": True,
                    "Assessment Composite Score": ":.1f",
                    "Assignment GitHub Score": ":.1f",
                    "Overall Performance Score": ":.1f",
                },
            )
            fig_ladder.update_layout(
                height=380,
                paper_bgcolor="white",
                plot_bgcolor="white",
                margin=dict(l=10, r=10, t=10, b=10),
                yaxis_title="",
                xaxis_title="Performance Score",
                legend_title="",
            )
            fig_ladder.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig_ladder.update_xaxes(range=[0, 100])
            st.plotly_chart(fig_ladder, use_container_width=True)

        st.markdown(
            """
            <div class="pmq-panel">
                <div class="pmq-panel-title">Published Snapshot</div>
                <div class="pmq-panel-subtitle">Summary values taken directly from the published PMQ workbook, useful as a reference point alongside the current filtered view.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        snapshot_cols = st.columns(6)
        snapshot_cards = [
            ("Published Students", str(summary.get("Total Students", "N/A")), "Workbook-level population in the published report."),
            ("Assessment Composite", str(summary.get("Average Assessment Composite Score", "N/A")), "Published workbook average."),
            ("Assignment GitHub", str(summary.get("Average Assignment GitHub Score", "N/A")), "Published workbook average."),
            ("Attendance", str(summary.get("Average Attendance %", "N/A")), "Published workbook average."),
            ("Trainer Feedback", str(summary.get("Average Trainer Feedback Score", "N/A")), "Published workbook average."),
            ("Overall Score", str(summary.get("Average Overall Performance Score", "N/A")), "Published workbook average."),
        ]
        for col, (title, value, detail) in zip(snapshot_cols, snapshot_cards):
            with col:
                _render_signal_card(title, value, detail)

    with attendance_tab:
        st.markdown(
            """
            <div class="pmq-panel">
                <div class="pmq-panel-title">Batch-wise Attendance Progress</div>
                <div class="pmq-panel-subtitle">Attendance and overall performance by batch for the current selection. This helps compare delivery discipline and outcome quality side by side.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        use_published_jecrc_batch = (
            not jecrc_batch_df.empty
            and selected_college in {"All Colleges", "JECRC"}
            and {"Assigned Batch", "Student Count", "Average Attendance %", "Average Overall Performance Score"} <= set(jecrc_batch_df.columns)
        )
        if use_published_jecrc_batch:
            batch_attendance_df = jecrc_batch_df.copy()
            if selected_batch != "All Batches":
                batch_attendance_df = batch_attendance_df[
                    batch_attendance_df["Assigned Batch"].astype(str) == selected_batch
                ].copy()
            batch_attendance_df = _coerce_numeric(
                batch_attendance_df,
                ["Student Count", "Average Attendance %", "Average Overall Performance Score"],
            )
            batch_attendance_df["Average Performance Score"] = pd.NA
            if not filtered_roster.empty and {"Assigned Batch", "Performance Score"} <= set(filtered_roster.columns):
                performance_lookup = (
                    filtered_roster.groupby("Assigned Batch", dropna=False)["Performance Score"]
                    .mean()
                    .round(1)
                )
                batch_attendance_df["Average Performance Score"] = batch_attendance_df["Assigned Batch"].map(performance_lookup)
            batch_attendance_df = batch_attendance_df.rename(columns={"Student Count": "Students"})
        else:
            batch_attendance_df = (
                filtered_roster.groupby("Assigned Batch", dropna=False)
                .agg(
                    Students=("Superset ID", "count"),
                    Average_Attendance=("Attendance %", "mean"),
                    Average_Overall=("Overall Performance Score", "mean"),
                    Average_Performance=("Performance Score", "mean"),
                )
                .reset_index()
                .rename(
                    columns={
                        "Average_Attendance": "Average Attendance %",
                        "Average_Overall": "Average Overall Performance",
                        "Average_Performance": "Average Performance Score",
                    }
                )
            )
        if not batch_attendance_df.empty:
            batch_attendance_df["Average Attendance %"] = batch_attendance_df["Average Attendance %"].round(1)
            batch_attendance_df["Average Overall Performance"] = batch_attendance_df["Average Overall Performance"].round(1)
            batch_attendance_df["Average Performance Score"] = batch_attendance_df["Average Performance Score"].round(1)

            batch_left, batch_right = st.columns([1.05, 1])
            with batch_left:
                fig_batch_attendance = px.bar(
                    batch_attendance_df,
                    x="Assigned Batch",
                    y="Average Attendance %",
                    color="Assigned Batch",
                    text="Average Attendance %",
                    color_discrete_sequence=["#1d4ed8", "#7c3aed", "#0ea5e9", "#10b981", "#f59e0b", "#ef4444"],
                )
                fig_batch_attendance.update_traces(texttemplate="%{text:.1f}", textposition="outside")
                fig_batch_attendance.update_layout(
                    height=420,
                    showlegend=False,
                    paper_bgcolor="white",
                    plot_bgcolor="white",
                    margin=dict(l=10, r=10, t=10, b=40),
                    xaxis_title="Batch",
                    yaxis_title="Average Attendance %",
                )
                fig_batch_attendance.update_yaxes(range=[0, 100])
                st.plotly_chart(fig_batch_attendance, use_container_width=True)

            with batch_right:
                fig_batch_compare = px.bar(
                    batch_attendance_df.melt(
                        id_vars=["Assigned Batch"],
                        value_vars=["Average Overall Performance", "Average Performance Score"],
                        var_name="Metric",
                        value_name="Average Score",
                    ),
                    x="Assigned Batch",
                    y="Average Score",
                    color="Metric",
                    barmode="group",
                    text="Average Score",
                    color_discrete_map={
                        "Average Overall Performance": "#0f172a",
                        "Average Performance Score": "#12b981",
                    },
                )
                fig_batch_compare.update_traces(texttemplate="%{text:.1f}", textposition="outside")
                fig_batch_compare.update_layout(
                    height=420,
                    paper_bgcolor="white",
                    plot_bgcolor="white",
                    margin=dict(l=10, r=10, t=10, b=40),
                    xaxis_title="Batch",
                    yaxis_title="Average Score",
                    legend_title="",
                )
                fig_batch_compare.update_yaxes(range=[0, 100])
                st.plotly_chart(fig_batch_compare, use_container_width=True)

            st.markdown(
                """
                <div class="pmq-panel" style="margin-top:16px;">
                    <div class="pmq-panel-title">Batch Summary Table</div>
                    <div class="pmq-panel-subtitle">Quick comparison of attendance strength and performance outcome by batch.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.dataframe(batch_attendance_df, use_container_width=True, hide_index=True)
        else:
            st.info("No batch-level attendance data is available for the current selection.")

    with signals_tab:
        left_col, right_col = st.columns([1, 1.1])
        with left_col:
            st.markdown(
                """
                <div class="pmq-panel">
                    <div class="pmq-panel-title">Signal Integrity</div>
                    <div class="pmq-panel-subtitle">Published matching quality and current-view signal availability, based on the dedicated integrity sheet in the workbook.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            integrity_view = _build_integrity_view(filtered_roster, integrity_df)
            integrity_cols = st.columns(2)
            for index, (_, row) in enumerate(integrity_view.iterrows()):
                with integrity_cols[index % 2]:
                    st.markdown(
                        f"""
                        <div class="pmq-integrity-card">
                            <div class="pmq-integrity-title">{row['Signal']}</div>
                            <div class="pmq-integrity-row"><span>Current View Coverage</span><strong>{int(row['Current View Coverage'])} / {len(filtered_roster)} ({row['Coverage %']:.0f}%)</strong></div>
                            <div class="pmq-integrity-row"><span>Published ID Matches</span><strong>{int(row['Published ID Matches'])}</strong></div>
                            <div class="pmq-integrity-row"><span>Published Missing Signals</span><strong>{int(row['Published Missing'])}</strong></div>
                            <div class="pmq-badge">Workbook integrity check</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            if not counts_df.empty:
                st.markdown(
                    """
                    <div class="pmq-panel" style="margin-top:18px;">
                        <div class="pmq-panel-title">Published Bucket Definitions</div>
                        <div class="pmq-panel-subtitle">Definitions sourced from the quadrant-count sheet in the published workbook.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                for _, row in counts_df.iterrows():
                    st.markdown(
                        f"""
                        <div class="pmq-mini-note">
                            <h4>{row['Metric']} ({int(row['Count'])})</h4>
                            <p>{row['Definition']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        with right_col:
            st.markdown(
                """
                <div class="pmq-panel">
                    <div class="pmq-panel-title">Published Insights & Current Narrative</div>
                    <div class="pmq-panel-subtitle">Workbook-provided observations blended with live commentary from the current filtered view.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if not insights_df.empty:
                for _, row in insights_df.iterrows():
                    st.markdown(
                        f"""
                        <div class="pmq-mini-note">
                            <h4>{row['Headline']}</h4>
                            <p>{row['Detail']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            for insight in _build_dynamic_insights(filtered_roster):
                st.markdown(
                    f"""
                    <div class="pmq-mini-note">
                        <h4>{insight['headline']}</h4>
                        <p>{insight['detail']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            focus_left, focus_right = st.columns(2)
            with focus_left:
                st.markdown(
                    """
                    <div class="pmq-panel" style="margin-top:18px;">
                        <div class="pmq-panel-title">Closest To Deployable</div>
                        <div class="pmq-panel-subtitle">Progressing candidates nearest to the top performance band.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                closest_df = (
                    filtered_roster[filtered_roster["Quadrant"].astype(str) == "Progressing Candidates"]
                    .sort_values(by="Performance Score", ascending=False)
                    .head(8)[
                        [
                            "Superset ID",
                            "Candidate Name",
                            "Assigned Batch",
                            "Performance Score",
                            "Overall Performance Score",
                        ]
                    ]
                    .copy()
                )
                st.dataframe(closest_df.fillna("N/A"), use_container_width=True, hide_index=True)

            with focus_right:
                st.markdown(
                    """
                    <div class="pmq-panel" style="margin-top:18px;">
                        <div class="pmq-panel-title">Needs Immediate Review</div>
                        <div class="pmq-panel-subtitle">Lowest-performing candidates in the current selection.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                review_df = (
                    filtered_roster.sort_values(by="Performance Score", ascending=True)
                    .head(8)[
                        [
                            "Superset ID",
                            "Candidate Name",
                            "Assigned Batch",
                            "Performance Score",
                            "Overall Performance Score",
                            "Quadrant",
                        ]
                    ]
                    .copy()
                )
                st.dataframe(review_df.fillna("N/A"), use_container_width=True, hide_index=True)

    with candidates_tab:
        st.markdown(
            """
            <div class="pmq-panel">
                <div class="pmq-panel-title">Candidate Drilldown</div>
                <div class="pmq-panel-subtitle">Bucket-level candidate mapping and the detailed roster for the current selection.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        using_published_candidate_map = (
            selected_college == "All Colleges"
            and selected_batch == "All Batches"
            and selected_quadrant == "All Buckets"
            and not search_text
            and not payload["candidate_map"].empty
        )
        candidate_map_df = payload["candidate_map"].copy() if using_published_candidate_map else _build_metric_candidate_map(filtered_roster)

        map_col, roster_col = st.columns([0.95, 1.05])
        with map_col:
            st.markdown(
                """
                <div class="pmq-panel">
                    <div class="pmq-panel-title">Bucket-to-Candidate Mapping</div>
                    <div class="pmq-panel-subtitle">Use the expanders below to inspect the candidate list behind each bucket count.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            for _, row in candidate_map_df.iterrows():
                count_value = int(row["Count"]) if not pd.isna(row["Count"]) else 0
                with st.expander(f"{row['Metric']} ({count_value})", expanded=False):
                    st.text(str(row["Candidate Details"]))

        with roster_col:
            st.markdown(
                """
                <div class="pmq-panel">
                    <div class="pmq-panel-title">Current View Roster</div>
                    <div class="pmq-panel-subtitle">Leadership-ready roster with the core PMQ columns preserved from the published workbook.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            preferred_columns = [
                "Superset ID",
                "Candidate Name",
                "College",
                "Assigned Batch",
                "Assessment GitHub Score",
                "Assessment GitHub Weeks",
                "Top Brains Score",
                "Assessment Composite Score",
                "Assignment GitHub Score",
                "Assignment GitHub Weeks",
                "Attendance %",
                "Trainer Feedback Score",
                "Performance Score",
                "Performance Grade",
                "Overall Performance Score",
                "Overall Grade",
                "Quadrant",
            ]
            display_columns = [column for column in preferred_columns if column in filtered_roster.columns]
            st.dataframe(filtered_roster[display_columns].fillna("N/A"), use_container_width=True, hide_index=True)

    with open(REPORT_PATH, "rb") as report_handle:
        st.download_button(
            "Download Published PMQ Workbook",
            data=report_handle.read(),
            file_name=REPORT_PATH.name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
