import numpy as np
import pandas as pd
import streamlit as st

from modules import task2_assessment


def run():
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #0f172a 0%, #0f766e 100%); padding: 24px 30px; border-radius: 18px; margin-bottom: 24px; box-shadow: 0 18px 36px rgba(15, 23, 42, 0.14);">
            <div style="color: #ffffff; font-size: 28px; font-weight: 800; margin-bottom: 6px;">Student 360 Degree</div>
            <div style="color: rgba(255, 255, 255, 0.88); font-size: 14px; max-width: 850px;">
                Dedicated learner profile combining subjectivity records, Top Brains performance, and Task-1 code evidence from the shared portal session.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tracker_data = st.session_state.get("tracker_data")
    if tracker_data is None:
        st.warning("Run Task-2 first to load the shared assessment session. Student 360 will then read the same normalized records.")
        return

    raw_tracker_df = task2_assessment.dataframe_from_session(tracker_data)
    if raw_tracker_df.empty:
        st.warning("The shared assessment session is empty, so the Student 360 profile cannot be generated yet.")
        return
    if "Superset ID" in raw_tracker_df.columns:
        raw_tracker_df["Superset ID"] = raw_tracker_df["Superset ID"].apply(task2_assessment.normalize_superset_id)

    eval_df, shared_eval_notices = task2_assessment.get_shared_eval_dataframe()
    notices = list(dict.fromkeys(st.session_state.get("assessment_notices", []) + shared_eval_notices))

    practical_tracker_df = task2_assessment.build_practical_tracker_from_eval_df(eval_df) if not eval_df.empty else pd.DataFrame()
    tracker_df = task2_assessment.build_combined_tracker_dataset(
        raw_tracker_df,
        practical_tracker_frames=[practical_tracker_df] if not practical_tracker_df.empty else None,
    )
    if tracker_df.empty:
        st.warning("Student 360 could not build a combined tracker view from the current shared records.")
        return

    batch_key = "task7_current_batch_selection"
    week_key = "task7_current_week_selection"
    available_batches = (
        sorted(tracker_df["Assigned Batch"].dropna().astype(str).unique().tolist())
        if "Assigned Batch" in tracker_df.columns
        else []
    )
    if st.session_state.get(batch_key) not in ["All Batches"] + available_batches:
        st.session_state[batch_key] = "All Batches"

    scoped_tracker = tracker_df.copy()
    if st.session_state[batch_key] != "All Batches":
        scoped_tracker = scoped_tracker[scoped_tracker["Assigned Batch"] == st.session_state[batch_key]].copy()
    plot_df = task2_assessment.aggregate_tracker_scope(scoped_tracker) if not scoped_tracker.empty else pd.DataFrame()
    week_map = task2_assessment.build_week_map(plot_df) if not plot_df.empty else {}
    week_options = ["All Weeks"] + list(week_map.keys()) if week_map else ["All Weeks"]
    if st.session_state.get(week_key) not in week_options:
        st.session_state[week_key] = week_options[0]

    with st.container(border=True):
        task2_assessment.render_section_header("Student 360 Controls", "Narrow the student profile view by batch and assessment week.")
        control_col1, control_col2 = st.columns(2)
        with control_col1:
            st.selectbox("Select Target Batch", ["All Batches"] + available_batches, key=batch_key)
        with control_col2:
            st.selectbox("Select Week for Student 360", week_options, key=week_key)

    selected_week = st.session_state.get(week_key)
    scoped_tracker = tracker_df.copy()
    if st.session_state[batch_key] != "All Batches":
        scoped_tracker = scoped_tracker[scoped_tracker["Assigned Batch"] == st.session_state[batch_key]].copy()
    plot_df = task2_assessment.aggregate_tracker_scope(scoped_tracker) if not scoped_tracker.empty else pd.DataFrame()
    week_map = task2_assessment.build_week_map(plot_df) if not plot_df.empty else {}
    selected_week_group = week_map.get(selected_week, {})
    rating_col = selected_week_group.get("rating")
    feedback_col = selected_week_group.get("feedback")
    total_score_col = selected_week_group.get("total_score")
    total_marks_col = selected_week_group.get("total_marks")
    practical_score_col = selected_week_group.get("practical_score")

    if selected_week == "All Weeks":
        rating_columns = [group.get("rating") for group in week_map.values() if group.get("rating")]
        feedback_columns = [group.get("feedback") for group in week_map.values() if group.get("feedback")]
        total_score_columns = [group.get("total_score") for group in week_map.values() if group.get("total_score")]
        total_marks_columns = [group.get("total_marks") for group in week_map.values() if group.get("total_marks")]
        practical_score_columns = [group.get("practical_score") for group in week_map.values() if group.get("practical_score")]
        active_df = plot_df.copy()
        if not active_df.empty:
            active_df["All Weeks - Rating"] = task2_assessment.average_numeric_columns(active_df, rating_columns)
            active_df["All Weeks - Feedback"] = task2_assessment.combine_text_columns(active_df, feedback_columns)
        rating_col = "All Weeks - Rating"
        feedback_col = "All Weeks - Feedback"
        if total_score_columns:
            active_df["All Weeks - TotalScore"] = task2_assessment.average_numeric_columns(active_df, total_score_columns)
            total_score_col = "All Weeks - TotalScore"
        else:
            total_score_col = None
        if total_marks_columns:
            active_df["All Weeks - TotalMarks"] = task2_assessment.average_numeric_columns(active_df, total_marks_columns).replace(0, 60).fillna(60)
            total_marks_col = "All Weeks - TotalMarks"
        else:
            total_marks_col = None
        if practical_score_columns:
            active_df["All Weeks - PracticalScore"] = task2_assessment.average_numeric_columns(active_df, practical_score_columns)
            practical_score_col = "All Weeks - PracticalScore"
        else:
            practical_score_col = None
    else:
        candidate_presence_cols = [column for column in [rating_col, practical_score_col, total_score_col] if column and column in plot_df.columns]
        if candidate_presence_cols:
            presence_mask = pd.Series(False, index=plot_df.index)
            for column in candidate_presence_cols:
                numeric_values = pd.to_numeric(plot_df[column], errors="coerce").fillna(0)
                presence_mask = presence_mask | numeric_values.gt(0)
            active_df = plot_df[presence_mask].copy()
        else:
            active_df = plot_df.copy()

    if not active_df.empty and "Candidate Name" in active_df.columns:
        active_df["Candidate Name"] = active_df["Candidate Name"].astype(str).str.strip()
        if "Superset ID" in active_df.columns:
            active_df["Superset ID"] = active_df["Superset ID"].apply(task2_assessment.normalize_superset_id)
        active_df["Merge_Key"] = active_df.apply(task2_assessment.build_merge_key, axis=1)
        active_df, dedupe_notices = task2_assessment.deduplicate_candidate_records(
            active_df,
            metric_column=rating_col,
            preferred_columns=[col for col in [rating_col, feedback_col, total_score_col, total_marks_col, practical_score_col] if col],
            context_label="student 360 tracker records",
        )
        notices = list(dict.fromkeys(notices + dedupe_notices))

    if not eval_df.empty:
        eval_df["Candidate Name"] = eval_df["Candidate Name"].astype(str).str.strip()
        if "Superset ID" in eval_df.columns:
            eval_df["Superset ID"] = eval_df["Superset ID"].apply(task2_assessment.normalize_superset_id)
        eval_df["Merge_Key"] = eval_df.apply(task2_assessment.build_merge_key, axis=1)

    if not active_df.empty and not eval_df.empty:
        eval_merge, _ = task2_assessment.deduplicate_candidate_records(
            eval_df.rename(columns={"Candidate Name": "Code Candidate Name"}).copy(),
            metric_column="Score",
            preferred_columns=task2_assessment.EVAL_COLUMNS,
            context_label="student 360 evaluation results",
            partition_columns=["Merge_Key"],
        )
        merged_df = pd.merge(active_df, eval_merge, on="Merge_Key", how="outer")
        merged_df["Candidate Name"] = merged_df.get("Candidate Name", pd.Series(dtype=str)).fillna(merged_df.get("Code Candidate Name", pd.Series(dtype=str)))
        if "Superset ID_x" in merged_df.columns or "Superset ID_y" in merged_df.columns:
            merged_df["Superset ID"] = merged_df.get("Superset ID_x", pd.Series(index=merged_df.index, dtype=object)).fillna(
                merged_df.get("Superset ID_y", pd.Series(index=merged_df.index, dtype=object))
            )
        merged_df["Assigned Batch"] = merged_df.get("Assigned Batch", pd.Series(dtype=str)).fillna("Assessment Batch")
    elif not active_df.empty:
        merged_df = active_df.copy()
    elif not eval_df.empty:
        merged_df, _ = task2_assessment.deduplicate_candidate_records(
            eval_df.copy(),
            metric_column="Score",
            preferred_columns=task2_assessment.EVAL_COLUMNS,
            context_label="student 360 standalone evaluation",
            partition_columns=["Merge_Key"],
        )
        merged_df["Assigned Batch"] = "Code Eval Batch"
    else:
        merged_df = pd.DataFrame()

    if not merged_df.empty:
        merged_df["Candidate Name"] = merged_df["Candidate Name"].fillna("").astype(str).str.strip()
        merged_df = merged_df[merged_df["Candidate Name"].ne("")].copy()
        if "Merge_Key" not in merged_df.columns:
            merged_df["Merge_Key"] = merged_df.apply(task2_assessment.build_merge_key, axis=1)
        practical_from_tracker = (
            pd.to_numeric(merged_df[practical_score_col], errors="coerce")
            if practical_score_col and practical_score_col in merged_df.columns
            else pd.Series(np.nan, index=merged_df.index, dtype=float)
        )
        practical_from_eval = (
            pd.to_numeric(merged_df.get("Score", np.nan), errors="coerce")
            if "Score" in merged_df.columns
            else pd.Series(np.nan, index=merged_df.index, dtype=float)
        )
        merged_df["Score"] = practical_from_tracker.fillna(practical_from_eval).fillna(0).clip(0, 100).round(0)
        merged_df["Unified Combined Score (%)"] = merged_df.apply(lambda row: task2_assessment.calc_unified(row, rating_col), axis=1)
        merged_df["Student Persona"] = merged_df.apply(lambda row: task2_assessment.generate_persona(row, rating_col), axis=1)

    task2_assessment.render_global_warnings(notices, task2_assessment.build_validation_messages(eval_df, week_map, merged_df, None))
    task2_assessment.render_individual_performance_panel(
        merged_df,
        eval_df,
        week_map,
        rating_col,
        feedback_col,
        total_score_col=total_score_col,
        total_marks_col=total_marks_col,
    )
