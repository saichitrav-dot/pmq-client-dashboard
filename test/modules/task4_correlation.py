import io
import re
import unicodedata

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


QUADRANT_COLORS = {
    "Q1: The Elite (Deployable)": "#10b981",
    "Q2: Flight Risk Geniuses": "#7c3aed",
    "Q3: Hardworking Strugglers": "#f59e0b",
    "Q4: Critical Intervention": "#ef4444",
}

QUADRANT_ACTIONS = {
    "Q1: The Elite (Deployable)": "Fast-track deployment readiness and preserve momentum with advanced assignments.",
    "Q2: Flight Risk Geniuses": "Prioritize trainer outreach and attendance accountability to prevent avoidable attrition.",
    "Q3: Hardworking Strugglers": "Maintain discipline while adding practice labs, code reviews, and problem-solving drills.",
    "Q4: Critical Intervention": "Create an intervention plan with attendance recovery, coaching check-ins, and close trainer follow-up.",
}

GRADE_ORDER = ["A+", "A", "B", "C", "F"]


def normalize_name(value):
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def score_to_grade(score):
    numeric_score = pd.to_numeric(score, errors="coerce")
    if pd.isna(numeric_score):
        return "N/A"
    numeric_score = float(numeric_score)
    if numeric_score >= 90:
        return "A+"
    if numeric_score >= 80:
        return "A"
    if numeric_score >= 70:
        return "B"
    if numeric_score >= 60:
        return "C"
    return "F"


def dataframe_from_session(data):
    if isinstance(data, pd.DataFrame):
        return data.copy()
    return pd.DataFrame(data or [])


def find_first_column(df, candidates):
    lowered = {str(column).strip().lower(): column for column in df.columns}
    for candidate in candidates:
        match = lowered.get(candidate.strip().lower())
        if match is not None:
            return match
    return None


def attendance_date_columns(df):
    return [
        column
        for column in df.columns
        if re.match(r"^\d{4}-\d{2}-\d{2}", str(column))
        or re.match(r"^\d{1,2}-[a-zA-Z]{3}-\d{2}", str(column))
    ]


def build_code_frame(eval_results):
    df = dataframe_from_session(eval_results)
    if df.empty:
        return pd.DataFrame()

    name_col = find_first_column(df, ["Candidate Name", "Trainee", "Student Name"])
    score_col = find_first_column(df, ["Score", "Raw Score", "AI Score", "Code Score"])
    if name_col is None or score_col is None:
        return pd.DataFrame()

    df["Candidate Name"] = df[name_col].astype(str).str.strip()
    df["Match_Name"] = df["Candidate Name"].apply(normalize_name)
    df["Code Score"] = pd.to_numeric(df[score_col], errors="coerce").fillna(0).clip(0, 100)

    for optional_col in ["PrimaryGap", "FilesAnalyzed", "ExperienceProfile", "Classification"]:
        if optional_col not in df.columns:
            df[optional_col] = np.nan

    keep_cols = [
        "Candidate Name",
        "Match_Name",
        "Code Score",
        "PrimaryGap",
        "FilesAnalyzed",
        "ExperienceProfile",
        "Classification",
    ]
    return (
        df[keep_cols]
        .sort_values(by=["Code Score", "Candidate Name"], ascending=[False, True])
        .drop_duplicates(subset=["Match_Name"], keep="first")
        .reset_index(drop=True)
    )


def build_tracker_frame(tracker_data):
    df = dataframe_from_session(tracker_data)
    if df.empty:
        return pd.DataFrame()

    name_col = find_first_column(df, ["Candidate Name", "Trainee", "Student Name"])
    if name_col is None:
        return pd.DataFrame()

    df["Candidate Name"] = df[name_col].astype(str).str.strip()
    df["Match_Name"] = df["Candidate Name"].apply(normalize_name)

    rating_cols = [column for column in df.columns if "rating" in str(column).lower()]
    if rating_cols:
        rating_frame = df[rating_cols].apply(pd.to_numeric, errors="coerce")
        df["Trainer Rating"] = rating_frame.mean(axis=1).fillna(0).clip(0, 5)
    else:
        df["Trainer Rating"] = 0.0

    batch_col = find_first_column(df, ["Assigned Batch", "Batch", "Batch Name"])
    if batch_col is None:
        df["Assigned Batch"] = "Unknown Batch"
    else:
        df["Assigned Batch"] = df[batch_col].fillna("Unknown Batch").astype(str).str.strip()

    theory_col = find_first_column(df, ["Numeric_Rating", "Theory Score", "Total Score", "totalScore"])
    if theory_col is None:
        df["Theory Score"] = np.nan
    else:
        values = pd.to_numeric(df[theory_col], errors="coerce")
        if values.max(skipna=True) <= 5:
            df["Theory Score"] = (values * 20).clip(0, 100)
        else:
            df["Theory Score"] = values.clip(0, 100)

    feedback_col = find_first_column(df, ["Trainer Feedback", "Feedback", "Comments"])
    if feedback_col is None:
        df["Trainer Feedback"] = ""
    else:
        df["Trainer Feedback"] = df[feedback_col].fillna("").astype(str)

    persona_col = find_first_column(df, ["Persona", "Persona Label", "Classification"])
    if persona_col is None:
        df["Persona"] = "Persona not available"
    else:
        df["Persona"] = df[persona_col].fillna("Persona not available").astype(str)

    keep_cols = [
        "Candidate Name",
        "Match_Name",
        "Trainer Rating",
        "Assigned Batch",
        "Theory Score",
        "Trainer Feedback",
        "Persona",
    ]
    return (
        df[keep_cols]
        .sort_values(by=["Trainer Rating", "Candidate Name"], ascending=[False, True])
        .drop_duplicates(subset=["Match_Name"], keep="first")
        .reset_index(drop=True)
    )


def compute_consistency_index(df):
    date_cols = attendance_date_columns(df)
    if not date_cols:
        return pd.Series(pd.to_numeric(df.get("Clean_Present_%", 0), errors="coerce").fillna(0).clip(0, 100), index=df.index)

    presence_frame = df[date_cols].astype(str).apply(
        lambda column: column.str.strip().str.upper().map({"P": 1, "A": 0, "L": 0, "OD": 1})
    )
    presence_frame = presence_frame.where(presence_frame.isin([0, 1]))
    variability = presence_frame.std(axis=1, skipna=True).fillna(0)
    attended_days = presence_frame.mean(axis=1, skipna=True).fillna(0) * 100
    consistency = (100 - (variability * 100)).clip(0, 100)
    return ((consistency * 0.6) + (attended_days * 0.4)).round(1)


def build_attendance_frame(att_data):
    df = dataframe_from_session(att_data)
    if df.empty:
        return pd.DataFrame()

    name_col = find_first_column(df, ["Candidate Name", "Trainee", "Student Name"])
    if name_col is None:
        return pd.DataFrame()

    df["Candidate Name"] = df[name_col].astype(str).str.strip()
    df["Match_Name"] = df["Candidate Name"].apply(normalize_name)

    attendance_col = find_first_column(df, ["Clean_Present_%", "Attendance %", "Present %"])
    if attendance_col is None:
        df["Attendance %"] = 0.0
    else:
        df["Attendance %"] = pd.to_numeric(df[attendance_col], errors="coerce").fillna(0).clip(0, 100)

    college_col = find_first_column(df, ["College", "College Name", "Institute"])
    status_col = find_first_column(df, ["Standard_Status", "Status", "Student Status"])
    cohort_col = find_first_column(df, ["Cohort", "Cohort Name"])

    df["College"] = df[college_col].fillna("Unknown College").astype(str).str.strip() if college_col else "Unknown College"
    df["Standard_Status"] = df[status_col].fillna("Unknown").astype(str).str.strip() if status_col else "Unknown"
    df["Cohort"] = df[cohort_col].fillna("Unknown Cohort").astype(str).str.strip() if cohort_col else "Unknown Cohort"
    df["Consistency Index"] = compute_consistency_index(df)

    keep_cols = [
        "Candidate Name",
        "Match_Name",
        "Attendance %",
        "Consistency Index",
        "Standard_Status",
        "College",
        "Cohort",
    ]
    return (
        df[keep_cols]
        .sort_values(by=["Attendance %", "Candidate Name"], ascending=[False, True])
        .drop_duplicates(subset=["Match_Name"], keep="first")
        .reset_index(drop=True)
    )


def build_executive_frame(code_df, tracker_df, attendance_df):
    attendance_base = attendance_df.copy()
    if attendance_base.empty:
        return pd.DataFrame()

    active_mask = attendance_base["Standard_Status"].astype(str).str.contains("Active", case=False, na=False)
    if active_mask.any():
        attendance_base = attendance_base.loc[active_mask].copy()

    attendance_base = attendance_base.rename(columns={"Candidate Name": "Attendance Candidate Name"})
    tracker_base = tracker_df.rename(columns={"Candidate Name": "Tracker Candidate Name"}).copy()
    code_base = code_df.rename(columns={"Candidate Name": "Code Candidate Name"}).copy()
    if code_base.empty:
        code_base = pd.DataFrame(
            columns=[
                "Match_Name",
                "Code Candidate Name",
                "Code Score",
                "PrimaryGap",
                "FilesAnalyzed",
                "ExperienceProfile",
                "Classification",
            ]
        )

    merged_df = attendance_base.merge(tracker_base, on="Match_Name", how="left")
    merged_df = merged_df.merge(code_base, on="Match_Name", how="left")
    merged_df["Candidate Name"] = (
        merged_df["Attendance Candidate Name"]
        .fillna(merged_df.get("Tracker Candidate Name"))
        .fillna(merged_df.get("Code Candidate Name"))
    )
    merged_df["Assigned Batch"] = merged_df.get("Assigned Batch", pd.Series(index=merged_df.index, dtype=object)).fillna("Unknown Batch")
    merged_df["College"] = merged_df.get("College", pd.Series(index=merged_df.index, dtype=object)).fillna("Unknown College")
    merged_df["Trainer Rating"] = pd.to_numeric(merged_df.get("Trainer Rating", 0), errors="coerce").fillna(0).clip(0, 5)
    merged_df["Theory Score"] = pd.to_numeric(merged_df.get("Theory Score", np.nan), errors="coerce")
    merged_df["Raw Code Score"] = pd.to_numeric(merged_df.get("Code Score", np.nan), errors="coerce")
    trainer_proxy = (merged_df["Trainer Rating"] * 12).clip(0, 100)
    merged_df["Skill Score"] = merged_df["Raw Code Score"].fillna(merged_df["Theory Score"]).fillna(trainer_proxy).fillna(0).clip(0, 100)
    merged_df["Skill Grade"] = merged_df["Skill Score"].apply(score_to_grade)
    merged_df["Theory Grade"] = merged_df["Theory Score"].apply(score_to_grade)
    merged_df["Attendance Grade"] = merged_df["Attendance %"].apply(score_to_grade)
    merged_df["Score Source"] = np.select(
        [
            merged_df["Raw Code Score"].notna(),
            merged_df["Theory Score"].notna(),
            merged_df["Trainer Rating"].gt(0),
        ],
        [
            "AI Code Evaluation",
            "Subjectivity Score Fallback",
            "Trainer Rating Fallback",
        ],
        default="No Skill Signal",
    )
    merged_df["PrimaryGap"] = merged_df.get("PrimaryGap", pd.Series(index=merged_df.index, dtype=object)).fillna("Gap not available")
    merged_df["Persona"] = merged_df.get("Persona", pd.Series(index=merged_df.index, dtype=object)).fillna("Persona not available")
    merged_df["Classification"] = merged_df.get("Classification", pd.Series(index=merged_df.index, dtype=object)).fillna("Not Classified")
    merged_df["FilesAnalyzed"] = pd.to_numeric(merged_df.get("FilesAnalyzed", 0), errors="coerce").fillna(0).astype(int)
    merged_df["ExperienceProfile"] = merged_df.get("ExperienceProfile", pd.Series(index=merged_df.index, dtype=object)).fillna("Not Available")
    return merged_df[merged_df["Candidate Name"].astype(str).str.strip().ne("")].copy()


def assign_quadrant(row):
    code_score = row.get("Skill Score", row.get("Code Score", 0))
    attendance_pct = row["Attendance %"]
    if code_score >= 60 and attendance_pct >= 75:
        return "Q1: The Elite (Deployable)"
    if code_score >= 60 and attendance_pct < 75:
        return "Q2: Flight Risk Geniuses"
    if code_score < 60 and attendance_pct >= 75:
        return "Q3: Hardworking Strugglers"
    return "Q4: Critical Intervention"


def build_exec_insights(df):
    if df.empty:
        return []

    total = len(df)
    counts = df["Quadrant"].value_counts()
    avg_attendance = df["Attendance %"].mean()
    avg_code = df["Skill Score"].mean()
    flight_risk_share = counts.get("Q2: Flight Risk Geniuses", 0) / total
    struggler_share = counts.get("Q3: Hardworking Strugglers", 0) / total
    critical_share = counts.get("Q4: Critical Intervention", 0) / total
    deployable_share = counts.get("Q1: The Elite (Deployable)", 0) / total

    insights = [
        {
            "headline": f"{deployable_share:.0%} of the filtered cohort are deployment-ready.",
                "detail": "These learners show both dependable attendance and credible skill signals.",
        },
        {
            "headline": f"{flight_risk_share:.0%} are high-skill but low-attendance flight-risk talents.",
            "detail": "Retention and accountability actions will protect the strongest technical learners from dropping away.",
        },
        {
            "headline": f"{struggler_share:.0%} attend consistently but still need coding acceleration.",
            "detail": "These students are coachable and may respond fastest to structured lab practice and review cycles.",
        },
    ]

    if critical_share > 0:
        insights.append(
            {
                "headline": f"{critical_share:.0%} require immediate intervention across both skill and dedication.",
                "detail": "This group needs close trainer supervision, attendance recovery, and targeted remediation plans.",
            }
        )
    if avg_attendance < 75:
        insights.append(
            {
                "headline": f"Average attendance is {avg_attendance:.1f}%, below the leadership comfort band.",
                "detail": "Cohort discipline risk is emerging even where technical capability exists.",
            }
        )
    if avg_code < 60:
        insights.append(
            {
                "headline": f"Average code score is {avg_code:.1f}, indicating practical readiness is still developing.",
                "detail": "The training program should emphasize project execution, code review depth, and problem-solving routines.",
            }
        )
    return insights[:4]


def build_action_roster(df):
    roster = df.copy()
    return roster[
        [
            "Candidate Name",
            "College",
            "Assigned Batch",
            "Attendance %",
            "Attendance Grade",
            "Raw Code Score",
            "Skill Score",
            "Skill Grade",
            "Theory Score",
            "Theory Grade",
            "Score Source",
            "Trainer Rating",
            "Skill Index",
            "Consistency Index",
            "Engagement Index",
            "Quadrant",
        ]
    ].rename(columns={"Raw Code Score": "Code Score"}).copy()


def to_excel_bytes(df):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="executive_roster")
    buffer.seek(0)
    return buffer.getvalue()


def render_kpi_card(title, count, description, accent):
    st.markdown(
        f"""
        <div style="background:#ffffff;border:1px solid #dbe7f5;border-top:4px solid {accent};border-radius:18px;padding:18px 18px 16px;box-shadow:0 12px 28px rgba(15,23,42,0.06);height:100%;">
            <div style="font-size:12px;font-weight:800;letter-spacing:0.08em;text-transform:uppercase;color:{accent};margin-bottom:10px;">{title}</div>
            <div style="font-size:34px;font-weight:900;color:#0f172a;line-height:1;">{count}</div>
            <div style="font-size:13px;color:#475569;margin-top:10px;line-height:1.5;">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def run():
    if "task4_quadrant_filter" not in st.session_state:
        st.session_state["task4_quadrant_filter"] = "All Quadrants"

    st.markdown(
        """
        <style>
        .gm-shell {
            background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 100%);
            border-radius: 22px;
            padding: 28px 32px;
            margin-bottom: 18px;
            box-shadow: 0 20px 40px rgba(15, 23, 42, 0.14);
        }
        .gm-title {
            color: #ffffff;
            font-size: 32px;
            font-weight: 900;
            margin: 0 0 8px 0;
        }
        .gm-subtitle {
            color: rgba(255,255,255,0.88);
            font-size: 15px;
            line-height: 1.6;
            max-width: 1050px;
        }
        .gm-panel {
            background: #ffffff;
            border: 1px solid #dbe7f5;
            border-radius: 18px;
            padding: 18px 20px;
            box-shadow: 0 12px 28px rgba(15,23,42,0.05);
            margin-bottom: 16px;
        }
        .gm-panel-title {
            color: #0f172a;
            font-size: 20px;
            font-weight: 800;
            margin-bottom: 6px;
        }
        .gm-panel-subtitle {
            color: #64748b;
            font-size: 14px;
            line-height: 1.6;
        }
        .gm-insight {
            background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
            border: 1px solid #dbe7f5;
            border-radius: 16px;
            padding: 14px 16px;
            margin-bottom: 10px;
        }
        .gm-insight-headline {
            color: #0f172a;
            font-size: 15px;
            font-weight: 800;
            margin-bottom: 4px;
        }
        .gm-insight-detail {
            color: #475569;
            font-size: 13px;
            line-height: 1.55;
        }
        </style>
        <div class="gm-shell">
            <div class="gm-title">Performance Magic Quadrant</div>
            <div class="gm-subtitle">
                Executive intelligence board unifying student dedication, practical skill, trainer perception, and attendance behavior
                into one high-signal decision surface for program directors and C-level reviews.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    eval_results = st.session_state.get("eval_results")
    tracker_data = st.session_state.get("tracker_data")
    att_data = st.session_state.get("att_data")

    if eval_results is None or tracker_data is None or att_data is None:
        st.warning("Run Task-1, Task-2, and Task-3 first so this module can consume the session-state datasets.")
        return

    code_df = build_code_frame(eval_results)
    tracker_df = build_tracker_frame(tracker_data)
    attendance_df = build_attendance_frame(att_data)

    if tracker_df.empty or attendance_df.empty:
        st.warning("The executive intelligence layer could not build a valid merged dataset from the available session-state records.")
        return

    merged_df = build_executive_frame(code_df, tracker_df, attendance_df)

    if merged_df.empty:
        st.warning("No active candidates were available for the executive intelligence layer.")
        return

    merged_df["Skill Index"] = (((merged_df["Trainer Rating"] * 20) + merged_df["Skill Score"]) / 2).round(1)
    merged_df["Engagement Index"] = ((merged_df["Attendance %"] * merged_df["Trainer Rating"]) / 5).round(1)
    merged_df["Quadrant"] = merged_df.apply(assign_quadrant, axis=1)
    merged_df["Bubble Size"] = merged_df["Trainer Rating"].replace(0, 0.4) * 8

    filter_panel = st.container(border=True)
    with filter_panel:
        st.markdown("#### Executive Filters")
        st.caption("College and batch filters update the quadrant, action metrics, and export roster together.")
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            college_options = ["All Colleges"] + sorted(merged_df["College"].dropna().astype(str).unique().tolist())
            selected_college = st.selectbox("College", college_options, key="task4_college")
        scoped_df = merged_df if selected_college == "All Colleges" else merged_df[merged_df["College"] == selected_college].copy()
        with filter_col2:
            batch_options = ["All Batches"] + sorted(scoped_df["Assigned Batch"].dropna().astype(str).unique().tolist())
            selected_batch = st.selectbox("Batch", batch_options, key="task4_batch")

    filtered_df = scoped_df.copy()
    if selected_batch != "All Batches":
        filtered_df = filtered_df[filtered_df["Assigned Batch"] == selected_batch].copy()

    if filtered_df.empty:
        st.warning("No candidates remain after applying the selected executive filters.")
        return

    counts = filtered_df["Quadrant"].value_counts()
    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        render_kpi_card(
            "Deployable Candidates",
            counts.get("Q1: The Elite (Deployable)", 0),
            "High attendance, strong code delivery, and favorable trainer confidence signals.",
            QUADRANT_COLORS["Q1: The Elite (Deployable)"],
        )
        if st.button("View Deployable", key="task4_view_q1", use_container_width=True):
            st.session_state["task4_quadrant_filter"] = "Q1: The Elite (Deployable)"
            st.rerun()
    with kpi_cols[1]:
        render_kpi_card(
            "Flight Risk Talents",
            counts.get("Q2: Flight Risk Geniuses", 0),
            "Technically capable learners whose attendance pattern suggests retention or discipline risk.",
            QUADRANT_COLORS["Q2: Flight Risk Geniuses"],
        )
        if st.button("View Flight Risk", key="task4_view_q2", use_container_width=True):
            st.session_state["task4_quadrant_filter"] = "Q2: Flight Risk Geniuses"
            st.rerun()
    with kpi_cols[2]:
        render_kpi_card(
            "Hardworking Strugglers",
            counts.get("Q3: Hardworking Strugglers", 0),
            "Consistent attendees who still need stronger code practice and problem-solving support.",
            QUADRANT_COLORS["Q3: Hardworking Strugglers"],
        )
        if st.button("View Strugglers", key="task4_view_q3", use_container_width=True):
            st.session_state["task4_quadrant_filter"] = "Q3: Hardworking Strugglers"
            st.rerun()
    with kpi_cols[3]:
        render_kpi_card(
            "Critical Intervention",
            counts.get("Q4: Critical Intervention", 0),
            "Low skill and low dedication combined, requiring the fastest leadership attention.",
            QUADRANT_COLORS["Q4: Critical Intervention"],
        )
        if st.button("View Critical", key="task4_view_q4", use_container_width=True):
            st.session_state["task4_quadrant_filter"] = "Q4: Critical Intervention"
            st.rerun()

    current_quadrant_filter = st.session_state.get("task4_quadrant_filter", "All Quadrants")
    if current_quadrant_filter not in {"All Quadrants"} | set(filtered_df["Quadrant"].unique().tolist()):
        current_quadrant_filter = "All Quadrants"
        st.session_state["task4_quadrant_filter"] = current_quadrant_filter

    reset_cols = st.columns([1, 3])
    with reset_cols[0]:
        if st.button("View All Cohort", key="task4_view_all", use_container_width=True):
            st.session_state["task4_quadrant_filter"] = "All Quadrants"
            st.rerun()

    insight_col, signal_col = st.columns([1.35, 1])
    with insight_col:
        st.markdown(
            """
            <div class="gm-panel">
                <div class="gm-panel-title">Executive AI Insight Panel</div>
                <div class="gm-panel-subtitle">Rule-based cohort intelligence summarizing the most important leadership signals in the current filtered view.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for insight in build_exec_insights(filtered_df):
            st.markdown(
                f"""
                <div class="gm-insight">
                    <div class="gm-insight-headline">{insight["headline"]}</div>
                    <div class="gm-insight-detail">{insight["detail"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with signal_col:
        st.markdown(
            """
            <div class="gm-panel">
                <div class="gm-panel-title">Leadership Signal Stack</div>
                <div class="gm-panel-subtitle">Composite indicators enrich the base quadrant without replacing its core classification logic. Skill scoring uses AI code evidence first, then falls back to subjectivity or trainer signals when practical evidence is missing.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        metric_col1.metric("Skill Index", f"{filtered_df['Skill Index'].mean():.1f}")
        metric_col2.metric("Consistency Index", f"{filtered_df['Consistency Index'].mean():.1f}")
        metric_col3.metric("Engagement Index", f"{filtered_df['Engagement Index'].mean():.1f}")
        metric_col4.metric("AI Code Coverage", int(filtered_df["Score Source"].eq("AI Code Evaluation").sum()))
        st.caption(
            "Skill Index blends trainer rating with code-first skill scoring. Consistency Index measures attendance stability. Engagement Index combines attendance and trainer perception."
        )

    st.markdown(
        """
        <div class="gm-panel">
            <div class="gm-panel-title">The Magic Quadrant</div>
            <div class="gm-panel-subtitle">X-axis shows dedication through attendance, Y-axis shows practical skill through AI code score, and bubble size reflects trainer perception.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    fig = px.scatter(
        filtered_df,
        x="Attendance %",
        y="Skill Score",
        size="Bubble Size",
        color="Quadrant",
        color_discrete_map=QUADRANT_COLORS,
        size_max=28,
        custom_data=[
            "Candidate Name",
            "Attendance %",
            "Attendance Grade",
            "Skill Score",
            "Skill Grade",
            "Raw Code Score",
            "Theory Score",
            "Theory Grade",
            "Trainer Rating",
            "Quadrant",
            "College",
            "Assigned Batch",
            "Skill Index",
            "Consistency Index",
            "Engagement Index",
            "Score Source",
        ],
    )
    fig.add_shape(type="rect", x0=75, y0=60, x1=100, y1=100, fillcolor=QUADRANT_COLORS["Q1: The Elite (Deployable)"], opacity=0.05, layer="below", line_width=0)
    fig.add_shape(type="rect", x0=0, y0=60, x1=75, y1=100, fillcolor=QUADRANT_COLORS["Q2: Flight Risk Geniuses"], opacity=0.05, layer="below", line_width=0)
    fig.add_shape(type="rect", x0=75, y0=0, x1=100, y1=60, fillcolor=QUADRANT_COLORS["Q3: Hardworking Strugglers"], opacity=0.06, layer="below", line_width=0)
    fig.add_shape(type="rect", x0=0, y0=0, x1=75, y1=60, fillcolor=QUADRANT_COLORS["Q4: Critical Intervention"], opacity=0.05, layer="below", line_width=0)
    fig.add_vline(x=75, line_width=2, line_dash="dash", line_color="#94a3b8")
    fig.add_hline(y=60, line_width=2, line_dash="dash", line_color="#94a3b8")
    fig.add_annotation(x=88, y=95, text="Deployable", showarrow=False, font=dict(size=22, color=QUADRANT_COLORS["Q1: The Elite (Deployable)"]))
    fig.add_annotation(x=36, y=95, text="Flight Risk", showarrow=False, font=dict(size=22, color=QUADRANT_COLORS["Q2: Flight Risk Geniuses"]))
    fig.add_annotation(x=87, y=10, text="Hardworking<br>Strugglers", showarrow=False, font=dict(size=19, color=QUADRANT_COLORS["Q3: Hardworking Strugglers"]))
    fig.add_annotation(x=34, y=10, text="Critical<br>Intervention", showarrow=False, font=dict(size=20, color=QUADRANT_COLORS["Q4: Critical Intervention"]))
    fig.update_traces(
        marker=dict(line=dict(width=1, color="rgba(15,23,42,0.18)")),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Attendance: %{customdata[1]:.1f}%<br>"
            "Attendance Grade: %{customdata[2]}<br>"
            "Skill Score: %{customdata[3]:.1f}<br>"
            "Skill Grade: %{customdata[4]}<br>"
            "AI Code Score: %{customdata[5]:.1f}<br>"
            "Theory Score: %{customdata[6]:.1f}<br>"
            "Theory Grade: %{customdata[7]}<br>"
            "Trainer Rating: %{customdata[8]:.2f}<br>"
            "Quadrant: %{customdata[9]}<br>"
            "College: %{customdata[10]}<br>"
            "Batch: %{customdata[11]}<br>"
            "Skill Index: %{customdata[12]:.1f}<br>"
            "Consistency Index: %{customdata[13]:.1f}<br>"
            "Engagement Index: %{customdata[14]:.1f}<br>"
            "Signal Source: %{customdata[15]}<extra></extra>"
        ),
    )
    fig.update_layout(
        height=680,
        margin=dict(t=24, b=24, l=24, r=24),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.16, x=0, title=""),
    )
    fig.update_xaxes(title="Student Dedication (Attendance %)", range=[0, 100], showgrid=False, zeroline=False)
    fig.update_yaxes(title="Skill Score (Code-First)", range=[0, 100], showgrid=False, zeroline=False)
    st.plotly_chart(fig, use_container_width=True)

    roster_source_df = filtered_df.copy()
    if current_quadrant_filter != "All Quadrants":
        roster_source_df = roster_source_df[roster_source_df["Quadrant"] == current_quadrant_filter].copy()
    roster_df = build_action_roster(roster_source_df)
    st.markdown(
        """
        <div class="gm-panel">
            <div class="gm-panel-title">Exportable Executive Roster</div>
            <div class="gm-panel-subtitle">Filtered student roster for leadership reviews, trainer governance calls, and client-facing reporting packs.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Current cohort view: {current_quadrant_filter}")
    export_col1, export_col2 = st.columns(2)
    with export_col1:
        st.download_button(
            "Download CSV",
            data=roster_df.to_csv(index=False).encode("utf-8"),
            file_name="magic_quadrant_roster.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with export_col2:
        st.download_button(
            "Download Excel",
            data=to_excel_bytes(roster_df),
            file_name="magic_quadrant_roster.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    st.dataframe(roster_df.fillna("N/A"), use_container_width=True, hide_index=True)
