import pandas as pd
import plotly.express as px
import streamlit as st

from modules import task2_assessment


def merge_top_brains_into_shared_session(top_brains_df, notices):
    existing_tracker = task2_assessment.dataframe_from_session(st.session_state.get("tracker_data"))
    if existing_tracker.empty:
        combined_tracker = top_brains_df.copy()
    else:
        combined_tracker = pd.concat([existing_tracker, top_brains_df], ignore_index=True)
        combined_tracker = combined_tracker.drop_duplicates().reset_index(drop=True)

    st.session_state["tracker_data"] = combined_tracker
    existing_notices = st.session_state.get("assessment_notices", [])
    st.session_state["assessment_notices"] = list(dict.fromkeys(existing_notices + notices))


def run():
    st.markdown(
        """
        <style>
        .qa-section { background: #ffffff; border: 1px solid #dbe7f5; border-radius: 16px; padding: 14px 18px; margin-bottom: 14px; box-shadow: 0 10px 22px rgba(15, 23, 42, 0.05); }
        .qa-section-title { color: #0f172a; font-size: 22px; font-weight: 800; margin-bottom: 4px; }
        .qa-section-subtitle { color: #64748b; font-size: 15px; }
        .qa-metric-card { background: #ffffff; border: 1px solid #e2e8f0; border-top: 4px solid #2563eb; border-radius: 14px; min-height: 112px; padding: 16px 12px; box-shadow: 0 8px 18px rgba(15, 23, 42, 0.06); display: flex; flex-direction: column; justify-content: center; text-align: center; }
        .qa-metric-value { font-size: 26px; font-weight: 800; line-height: 1.1; }
        .qa-metric-label { margin-top: 6px; color: #475569; font-size: 13px; font-weight: 800; letter-spacing: 0.06em; text-transform: uppercase; }
        .qa-metric-subtext { margin-top: 4px; color: #94a3b8; font-size: 12px; }
        .qa-banner { border: 1px solid #dbeafe; border-left: 5px solid #1d4ed8; border-radius: 14px; padding: 14px 16px; margin-bottom: 14px; }
        .qa-banner-text { color: #334155; font-size: 14px; font-weight: 600; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #0f172a 0%, #7c3aed 100%); padding: 24px 30px; border-radius: 18px; margin-bottom: 24px; box-shadow: 0 18px 36px rgba(15, 23, 42, 0.14);">
            <div style="color: #ffffff; font-size: 28px; font-weight: 800; margin-bottom: 6px;">Top Brains Performance</div>
            <div style="color: rgba(255, 255, 255, 0.88); font-size: 14px; max-width: 850px;">
                Dedicated view for Top Brains coding and MCQ performance, with week-wise score trends and explicit status tracking.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "task6_show_append_uploader" not in st.session_state:
        st.session_state["task6_show_append_uploader"] = False

    tracker_data = st.session_state.get("tracker_data")
    raw_tracker_df = task2_assessment.dataframe_from_session(tracker_data)
    if not raw_tracker_df.empty and "Superset ID" in raw_tracker_df.columns:
        raw_tracker_df["Superset ID"] = raw_tracker_df["Superset ID"].apply(task2_assessment.normalize_superset_id)
    _, top_brains_df = task2_assessment.split_tracker_sources(raw_tracker_df)

    if top_brains_df.empty or st.session_state.get("task6_show_append_uploader"):
        with st.container(border=True):
            title = "Load Top Brains Files" if top_brains_df.empty else "Add More Top Brains Files"
            subtitle = (
                "Upload Top Brains workbooks directly here. The normalized Top Brains data will also be synchronized into the shared assessment session to keep Task-2, Task-5, and Student 360 aligned."
            )
            task2_assessment.render_section_header(title, subtitle)
            uploaded_files = st.file_uploader(
                "Upload Top Brains Workbooks",
                type=["xlsx", "xlsm", "csv"],
                accept_multiple_files=True,
                key="task6_uploader",
            )
            action_col1, action_col2 = st.columns([1, 1])
            with action_col1:
                process_label = "Process Top Brains Files" if top_brains_df.empty else "Add Files to Current Top Brains Session"
                if uploaded_files and st.button(process_label, type="primary", key="task6_process_upload"):
                    with st.spinner("Processing Top Brains files and updating the shared assessment session..."):
                        tracker_res, _, notices = task2_assessment.process_files(uploaded_files)
                        _, uploaded_top_brains_df = task2_assessment.split_tracker_sources(tracker_res)
                        if uploaded_top_brains_df.empty:
                            st.warning("No Top Brains records were detected in the uploaded files.")
                        else:
                            merge_top_brains_into_shared_session(uploaded_top_brains_df, notices)
                            st.session_state["task6_show_append_uploader"] = False
                            st.rerun()
            with action_col2:
                if not top_brains_df.empty and st.button("Close Uploader", key="task6_close_uploader"):
                    st.session_state["task6_show_append_uploader"] = False
                    st.rerun()

        if top_brains_df.empty:
            st.info("Upload a Top Brains workbook to start this page. Task 2 is no longer required before opening Task 6.")
            return

    if not top_brains_df.empty:
        action_col1, action_col2 = st.columns(2)
        with action_col1:
            if st.button("Add More Files", key="task6_open_append"):
                st.session_state["task6_show_append_uploader"] = True
                st.rerun()
        with action_col2:
            if st.button("Refresh Page", key="task6_refresh"):
                st.rerun()

    available_files = sorted(top_brains_df["Source_File"].dropna().astype(str).unique().tolist()) if "Source_File" in top_brains_df.columns else []
    file_key = "task6_current_file_selection"
    batch_key = "task6_current_batch_selection"
    week_key = "task6_current_week_selection"
    filter_key = "task6_table_filter"

    if st.session_state.get(file_key) not in ["All Combined Top Brains"] + available_files:
        st.session_state[file_key] = "All Combined Top Brains"

    selected_file = st.session_state.get(file_key)
    if selected_file == "All Combined Top Brains":
        tracker_df = task2_assessment.build_combined_tracker_dataset(top_brains_df)
    else:
        tracker_df = top_brains_df[top_brains_df["Source_File"] == selected_file].copy()

    available_batches = (
        sorted(tracker_df["Assigned Batch"].dropna().astype(str).unique().tolist())
        if not tracker_df.empty and "Assigned Batch" in tracker_df.columns
        else []
    )
    if st.session_state.get(batch_key) not in ["All Batches"] + available_batches:
        st.session_state[batch_key] = "All Batches"

    scope_df = tracker_df.copy()
    if not scope_df.empty and st.session_state[batch_key] != "All Batches":
        scope_df = scope_df[scope_df["Assigned Batch"] == st.session_state[batch_key]].copy()

    plot_df = task2_assessment.aggregate_tracker_scope(scope_df) if not scope_df.empty else pd.DataFrame()
    week_map = task2_assessment.build_week_map(plot_df) if not plot_df.empty else {}
    week_options = ["All Weeks"] + list(week_map.keys()) if week_map else ["All Weeks"]
    if st.session_state.get(week_key) not in week_options:
        st.session_state[week_key] = week_options[0]

    with st.container(border=True):
        task2_assessment.render_section_header("Top Brains Controls", "Filter the Top Brains performance view by uploaded file, batch, and assessment week.")
        control_col1, control_col2, control_col3 = st.columns(3)
        with control_col1:
            st.selectbox("Select Top Brains File", ["All Combined Top Brains"] + available_files, key=file_key)
        with control_col2:
            st.selectbox("Select Target Batch", ["All Batches"] + available_batches, key=batch_key)
        with control_col3:
            st.selectbox("Select Week for Analysis", week_options, key=week_key)

    selected_file = st.session_state.get(file_key)
    selected_week = st.session_state.get(week_key)
    if selected_file == "All Combined Top Brains":
        tracker_df = task2_assessment.build_combined_tracker_dataset(top_brains_df)
    else:
        tracker_df = top_brains_df[top_brains_df["Source_File"] == selected_file].copy()
    if not tracker_df.empty and st.session_state[batch_key] != "All Batches":
        tracker_df = tracker_df[tracker_df["Assigned Batch"] == st.session_state[batch_key]].copy()

    plot_df = task2_assessment.aggregate_tracker_scope(tracker_df) if not tracker_df.empty else pd.DataFrame()
    week_map = task2_assessment.build_week_map(plot_df) if not plot_df.empty else {}
    selected_week_group = week_map.get(selected_week, {})
    rating_col = selected_week_group.get("rating")
    feedback_col = selected_week_group.get("feedback")
    topic_col = selected_week_group.get("topic")
    total_score_col = selected_week_group.get("total_score")
    total_marks_col = selected_week_group.get("total_marks")
    practical_score_col = selected_week_group.get("practical_score")

    if selected_week == "All Weeks":
        active_df = plot_df.copy()
        feedback_columns = [group.get("feedback") for group in week_map.values() if group.get("feedback")]
        total_score_columns = [group.get("total_score") for group in week_map.values() if group.get("total_score")]
        total_marks_columns = [group.get("total_marks") for group in week_map.values() if group.get("total_marks")]
        practical_score_columns = [group.get("practical_score") for group in week_map.values() if group.get("practical_score")]
        active_df["All Weeks - Feedback"] = task2_assessment.combine_text_columns(active_df, feedback_columns)
        feedback_col = "All Weeks - Feedback"
        if total_score_columns:
            active_df["All Weeks - TotalScore"] = task2_assessment.average_numeric_columns(active_df, total_score_columns)
            total_score_col = "All Weeks - TotalScore"
        else:
            total_score_col = None
        if total_marks_columns:
            active_df["All Weeks - TotalMarks"] = task2_assessment.average_numeric_columns(active_df, total_marks_columns).replace(0, 100).fillna(100)
            total_marks_col = "All Weeks - TotalMarks"
        else:
            total_marks_col = None
        if practical_score_columns:
            active_df["All Weeks - PracticalScore"] = task2_assessment.average_numeric_columns(active_df, practical_score_columns)
            practical_score_col = "All Weeks - PracticalScore"
        else:
            practical_score_col = None
    else:
        # Keep the full week cohort in scope so explicit statuses like
        # Absent / No Show / Drop out are still counted in the widgets
        # and detail grid even when the student has a zero score that week.
        active_df = plot_df.copy()

    if not active_df.empty and "Candidate Name" in active_df.columns:
        active_df["Candidate Name"] = active_df["Candidate Name"].astype(str).str.strip()
        if "Superset ID" in active_df.columns:
            active_df["Superset ID"] = active_df["Superset ID"].apply(task2_assessment.normalize_superset_id)
        active_df["Merge_Key"] = active_df.apply(task2_assessment.build_merge_key, axis=1)
        active_df, _ = task2_assessment.deduplicate_candidate_records(
            active_df,
            metric_column=practical_score_col or total_score_col,
            preferred_columns=[col for col in [total_score_col, total_marks_col, practical_score_col, feedback_col, topic_col] if col],
            context_label="top brains records",
        )

    assessment_scores = task2_assessment.calculate_assessment_scores(active_df, rating_col, total_score_col, total_marks_col) if not active_df.empty else pd.Series(dtype=float)
    practical_scores = (
        pd.to_numeric(active_df[practical_score_col], errors="coerce").fillna(0)
        if not active_df.empty and practical_score_col and practical_score_col in active_df.columns
        else pd.Series(dtype=float)
    )
    exception_week_map = week_map if selected_week == "All Weeks" else ({selected_week: selected_week_group} if selected_week_group else {})
    latest_status, absent_mask, no_show_mask, dropout_mask = task2_assessment.build_explicit_exception_masks(active_df, exception_week_map) if not active_df.empty else (
        pd.Series(dtype=object),
        pd.Series(dtype=bool),
        pd.Series(dtype=bool),
        pd.Series(dtype=bool),
    )
    pass_mask = assessment_scores.ge(60) & ~absent_mask & ~no_show_mask & ~dropout_mask if not assessment_scores.empty else pd.Series(dtype=bool)
    attention_mask = assessment_scores.gt(0) & assessment_scores.lt(60) & ~absent_mask & ~no_show_mask & ~dropout_mask if not assessment_scores.empty else pd.Series(dtype=bool)
    exception_mask = (absent_mask | no_show_mask | dropout_mask) if not active_df.empty else pd.Series(dtype=bool)
    total_students = int(len(active_df)) if not active_df.empty else 0
    avg_assessment = float(assessment_scores[assessment_scores.gt(0)].mean()) if not assessment_scores.empty and assessment_scores.gt(0).any() else None
    avg_practical = float(practical_scores[practical_scores.gt(0)].mean()) if not practical_scores.empty and practical_scores.gt(0).any() else None

    if st.session_state.get(filter_key) not in {"All", "Pass", "Attention", "Absent", "NoShow", "Dropout"}:
        st.session_state[filter_key] = "All"

    if selected_week == "All Weeks":
        metric_columns = st.columns(7)
        with metric_columns[0]:
            task2_assessment.render_metric_card(str(total_students), "Total Students", "Top Brains records in current view", "#2563eb")
            if st.button("View All", key="task6_btn_all"):
                st.session_state[filter_key] = "All"
                st.rerun()
        with metric_columns[1]:
            task2_assessment.render_metric_card(
                f"{avg_assessment:.1f} / 100" if avg_assessment is not None else "N/A",
                "Assessment Average",
                f"Grade {task2_assessment.score_to_grade(avg_assessment)}" if avg_assessment is not None else "No valid scored assessments",
                "#1d4ed8",
            )
        with metric_columns[2]:
            task2_assessment.render_metric_card(
                f"{avg_practical:.1f} / 100" if avg_practical is not None else "N/A",
                "Coding Average",
                f"Grade {task2_assessment.score_to_grade(avg_practical)}" if avg_practical is not None else "No practical scores detected",
                "#7c3aed",
            )
        with metric_columns[3]:
            task2_assessment.render_metric_card(str(int(pass_mask.sum()) if not pass_mask.empty else 0), "Pass / Active", "Students with valid passing score", "#059669")
            if st.button("View Pass", key="task6_btn_pass"):
                st.session_state[filter_key] = "Pass"
                st.rerun()
        with metric_columns[4]:
            task2_assessment.render_metric_card(str(int(attention_mask.sum()) if not attention_mask.empty else 0), "Scope for Improvement", "Students with low all-weeks assessment score", "#f59e0b")
            if st.button("View Scope for Improvement", key="task6_btn_attention"):
                st.session_state[filter_key] = "Attention"
                st.rerun()
        with metric_columns[5]:
            task2_assessment.render_metric_card(str(int(no_show_mask.sum()) if not no_show_mask.empty else 0), "No Show", "Latest meaningful all-weeks no-show status", "#f59e0b")
            if st.button("View No Show", key="task6_btn_noshow_all"):
                st.session_state[filter_key] = "NoShow"
                st.rerun()
        with metric_columns[6]:
            task2_assessment.render_metric_card(str(int(dropout_mask.sum()) if not dropout_mask.empty else 0), "Dropout", "Latest meaningful all-weeks dropout status", "#64748b")
            if st.button("View Dropout", key="task6_btn_dropout_all"):
                st.session_state[filter_key] = "Dropout"
                st.rerun()
        absent_count = int(absent_mask.sum()) if not absent_mask.empty else 0
        st.caption(f"All Weeks hides the Absent widget. {absent_count} student(s) currently resolve to latest status Absent; review week-wise data for exact absence detail.")
    else:
        metric_columns = st.columns(7)
        with metric_columns[0]:
            task2_assessment.render_metric_card(str(total_students), "Total Students", "Top Brains records in current view", "#2563eb")
            if st.button("View All", key="task6_btn_all"):
                st.session_state[filter_key] = "All"
                st.rerun()
        with metric_columns[1]:
            task2_assessment.render_metric_card(
                f"{avg_assessment:.1f} / 100" if avg_assessment is not None else "N/A",
                "Assessment Average",
                f"Grade {task2_assessment.score_to_grade(avg_assessment)}" if avg_assessment is not None else "No valid scored assessments",
                "#1d4ed8",
            )
        with metric_columns[2]:
            task2_assessment.render_metric_card(
                f"{avg_practical:.1f} / 100" if avg_practical is not None else "N/A",
                "Coding Average",
                f"Grade {task2_assessment.score_to_grade(avg_practical)}" if avg_practical is not None else "No practical scores detected",
                "#7c3aed",
            )
        with metric_columns[3]:
            task2_assessment.render_metric_card(str(int(pass_mask.sum()) if not pass_mask.empty else 0), "Pass / Active", "Students with valid passing score", "#059669")
            if st.button("View Pass", key="task6_btn_pass_week"):
                st.session_state[filter_key] = "Pass"
                st.rerun()
        with metric_columns[4]:
            task2_assessment.render_metric_card(str(int(absent_mask.sum()) if not absent_mask.empty else 0), "Absent", "Latest explicit absent status", "#d97706")
            if st.button("View Absent", key="task6_btn_absent"):
                st.session_state[filter_key] = "Absent"
                st.rerun()
        with metric_columns[5]:
            task2_assessment.render_metric_card(str(int(no_show_mask.sum()) if not no_show_mask.empty else 0), "No Show", "Latest explicit no-show status", "#f59e0b")
            if st.button("View No Show", key="task6_btn_noshow"):
                st.session_state[filter_key] = "NoShow"
                st.rerun()
        with metric_columns[6]:
            task2_assessment.render_metric_card(str(int(dropout_mask.sum()) if not dropout_mask.empty else 0), "Dropout", "Latest explicit dropout status", "#64748b")
            if st.button("View Dropout", key="task6_btn_dropout"):
                st.session_state[filter_key] = "Dropout"
                st.rerun()

    if not active_df.empty:
        grade_records_df = task2_assessment.build_grade_distribution_records(
            active_df,
            week_map,
            selected_week,
            practical_scores=practical_scores if not practical_scores.empty else None,
        )
        if not grade_records_df.empty:
            color_map = {"A+": "#1d4ed8", "A": "#10b981", "B": "#3b82f6", "C": "#f59e0b", "F": "#ef4444"}
            with st.container(border=True):
                task2_assessment.render_section_header("Top Brains Grade Distribution", "Assessment and coding grade buckets for the current Top Brains selection.")
                fig_progression = px.bar(
                    grade_records_df,
                    x="Scope",
                    y="Student Count",
                    color="Grade",
                    barmode="group",
                    category_orders={"Grade": task2_assessment.GRADE_ORDER},
                    color_discrete_map=color_map,
                    height=340,
                )
                fig_progression.update_layout(margin=dict(t=20, b=0, l=0, r=0), yaxis_title="Number of Students", plot_bgcolor="white", paper_bgcolor="white")
                st.plotly_chart(fig_progression, use_container_width=True)

        filter_label_map = {
            "All": "All Students",
            "Pass": "Passing / Active Students",
            "Attention": "Scope for Improvement",
            "Absent": "Absent Students",
            "NoShow": "No Show Students",
            "Dropout": "Dropped Out Students",
        }
        display_df = active_df[["Candidate Name"]].copy()
        if "Superset ID" in active_df.columns:
            display_df["Superset ID"] = active_df["Superset ID"]
        display_df["Assessment Score (Out of 100)"] = assessment_scores.where(assessment_scores.gt(0), other=pd.NA)
        display_df["Assessment Grade"] = assessment_scores.apply(task2_assessment.score_to_grade).replace("N/A", pd.NA)
        if practical_score_col and practical_score_col in active_df.columns:
            display_df["Practical Code (Out of 100)"] = practical_scores.where(practical_scores.gt(0), other=pd.NA)
            display_df["Practical Grade"] = practical_scores.apply(task2_assessment.score_to_grade).replace("N/A", pd.NA)
        display_df["Latest Status"] = latest_status.replace({"": pd.NA, "Active": "Pass"})
        if topic_col and topic_col in active_df.columns:
            display_df["Topic"] = active_df[topic_col]
        display_df["Assigned Batch"] = active_df.get("Assigned Batch", "Top Brains Tool")
        display_df["Source File"] = active_df.get("Source_File", "Top Brains Tool")

        filter_value = st.session_state.get(filter_key, "All")
        if filter_value == "Pass" and not pass_mask.empty:
            display_df = display_df.loc[pass_mask]
        elif filter_value == "Attention" and not attention_mask.empty:
            display_df = display_df.loc[attention_mask]
        elif filter_value == "Absent" and not absent_mask.empty:
            display_df = display_df.loc[absent_mask]
        elif filter_value == "NoShow" and not no_show_mask.empty:
            display_df = display_df.loc[no_show_mask]
        elif filter_value == "Dropout" and not dropout_mask.empty:
            display_df = display_df.loc[dropout_mask]

        with st.container(border=True):
            task2_assessment.render_section_header("Top Brains Roster", f"Showing: {filter_label_map.get(filter_value, 'All Students')}")
            st.dataframe(display_df.fillna("N/A"), use_container_width=True, hide_index=True)

        student_option_map = {}
        for _, row in active_df.iterrows():
            name = str(row.get("Candidate Name", "")).strip()
            if not name:
                continue
            student_option_map[name] = row.get("Merge_Key")
        if student_option_map:
            with st.container(border=True):
                task2_assessment.render_section_header("Student Week Trend", "Review one Top Brains learner across the loaded weeks.")
                selected_student = st.selectbox("Select Student", sorted(student_option_map.keys()), key="task6_selected_student")
                selected_key = student_option_map[selected_student]
                student_row = active_df[active_df["Merge_Key"] == selected_key].iloc[0]
                weekly_progress_df = task2_assessment.build_student_weekly_progress(student_row, week_map)
                if weekly_progress_df.empty:
                    st.info("No week-on-week Top Brains trend is available for the selected student.")
                else:
                    trend_col1, trend_col2 = st.columns([1.15, 0.85])
                    with trend_col1:
                        fig_student_trend = px.line(
                            weekly_progress_df,
                            x="Week",
                            y="Assessment Score (Out of 100)",
                            markers=True,
                            text="Assessment Grade",
                            color_discrete_sequence=["#7c3aed"],
                        )
                        fig_student_trend.update_traces(line=dict(width=3), textposition="top center")
                        fig_student_trend.update_layout(margin=dict(t=8, b=0, l=0, r=0), height=320, yaxis_title="Score /100", xaxis_title="")
                        st.plotly_chart(fig_student_trend, use_container_width=True)
                    with trend_col2:
                        st.markdown(task2_assessment.summarize_student_weekly_progress(weekly_progress_df))
                        st.dataframe(weekly_progress_df.fillna("N/A"), use_container_width=True, hide_index=True)
