import re
import time
import unicodedata

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

GITHUB_ASSESSMENT_TRACK = "Assessment Scores"
GITHUB_ASSIGNMENT_TRACK = "Assignment Scores"


EVAL_COLUMNS = [
    "Superset ID",
    "Candidate Name",
    "AssessmentWeek",
    "EvaluationTrack",
    "Score",
    "PrimaryGap",
    "Code Quality",
    "Structure",
    "Logic Correctness",
    "Security",
    "Documentation",
    "FilesAnalyzed",
    "ExperienceProfile",
    "ReviewStrictness",
    "Status",
    "Worksheet",
    "DeclaredTechnology",
    "LanguagesTested",
]

EVAL_DEFAULTS = {
    "Superset ID": "",
    "Candidate Name": "",
    "AssessmentWeek": "",
    "EvaluationTrack": GITHUB_ASSESSMENT_TRACK,
    "Score": 0,
    "PrimaryGap": "No critical gap identified.",
    "Code Quality": 0,
    "Structure": 0,
    "Logic Correctness": 0,
    "Security": 0,
    "Documentation": 0,
    "FilesAnalyzed": 0,
    "ExperienceProfile": "",
    "ReviewStrictness": "",
    "Status": "",
    "Worksheet": "",
    "DeclaredTechnology": "",
    "LanguagesTested": "",
}

PERSONA_COLORS = {
    "Champion (Strong Subjectivity & Practice)": "#10b981",
    "Quiet Coder (Weak Tests, Strong Code)": "#2563eb",
    "Subjectivity Heavy (Strong Observations, Weak Code)": "#f59e0b",
    "High Risk (Struggling Overall)": "#ef4444",
    "Subjectivity Anchor (Strong Subjectivity)": "#7c3aed",
    "Needs Practice Exposure": "#06b6d4",
    "Instructor Support Required": "#e11d48",
    "Code Anchor (Strong Practicals)": "#0f766e",
    "Code Developing (Improving Practicals)": "#84cc16",
    "Code Support Needed": "#f97316",
    "Unclassified": "#94a3b8",
}

GRADE_ORDER = ["A+", "A", "B", "C", "F"]


def normalize_key(value):
    text = str(value or "").strip().lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def normalize_name(value):
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def normalize_superset_id(value):
    text = str(value or "").strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return ""
    return text[:-2] if text.endswith(".0") else text


def normalize_evaluation_track(value):
    key = normalize_key(value)
    if key in {"assignmentscore", "assignmentscores", "assignment", "dailyassignment", "dailyassignments"}:
        return GITHUB_ASSIGNMENT_TRACK
    return GITHUB_ASSESSMENT_TRACK


def dataframe_from_session(data):
    if isinstance(data, pd.DataFrame):
        return data.copy()
    return pd.DataFrame(data or [])


def build_merge_key(row):
    superset_id = normalize_superset_id(row.get("Superset ID", ""))
    if superset_id:
        return f"id::{superset_id}"
    return f"name::{normalize_name(row.get('Candidate Name', ''))}"


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


def score_to_result(score):
    numeric_score = pd.to_numeric(score, errors="coerce")
    if pd.isna(numeric_score):
        return "N/A"
    return "Pass" if float(numeric_score) >= 60 else "Fail"


def rating_to_percent(value):
    numeric_value = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric_value):
        return np.nan
    return float(numeric_value) * 20.0


def calculate_assessment_scores(df, rating_col=None, total_score_col=None, total_marks_col=None):
    if df is None or df.empty:
        return pd.Series(dtype=float)

    score_components = []
    if total_score_col and total_score_col in df.columns:
        total_scores = pd.to_numeric(df[total_score_col], errors="coerce")
        if total_marks_col and total_marks_col in df.columns:
            total_marks = pd.to_numeric(df[total_marks_col], errors="coerce").replace(0, 60).fillna(60)
        else:
            total_marks = pd.Series(60, index=df.index, dtype=float)
        total_score_component = ((total_scores / total_marks) * 100).clip(0, 100)
        if total_score_component.gt(0).any():
            score_components.append(total_score_component)

    if rating_col and rating_col in df.columns:
        numeric_ratings = pd.to_numeric(df[rating_col], errors="coerce")
        rating_scores = numeric_ratings.apply(rating_to_percent)
        if numeric_ratings.gt(0).any():
            score_components.append(rating_scores)

    if not score_components:
        return pd.Series(np.nan, index=df.index, dtype=float)

    assessment_scores = pd.concat(score_components, axis=1).mean(axis=1, skipna=True)
    return assessment_scores.round(1)


def extract_week_label(text):
    match = re.search(r"\b(?:week|wk)[\s\-_]*0*(\d+)\b", str(text), re.IGNORECASE)
    if not match:
        return None
    return f"Week {int(match.group(1))}"


def extract_assignment_cycle_label(text):
    value = str(text or "").strip()
    if not value:
        return None
    if "assignment" not in normalize_key(value):
        return None

    word_match = re.search(
        r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b",
        value,
        re.IGNORECASE,
    )
    if word_match:
        return word_match.group(1).title()

    number_match = re.search(r"\b(\d+)(?:st|nd|rd|th)?\b", value, re.IGNORECASE)
    if number_match:
        return f"Assignment {int(number_match.group(1))}"

    return "Assignment"


def parse_github_cycle_column(column_name):
    key = normalize_key(column_name)
    if key.startswith("assignment"):
        return GITHUB_ASSIGNMENT_TRACK, extract_assignment_cycle_label(column_name) or str(column_name).strip()

    week_label = extract_week_label(column_name)
    if week_label and key.startswith("assessment"):
        return GITHUB_ASSESSMENT_TRACK, week_label

    return None, None


def is_github_link_text(value):
    text = str(value or "").strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return False
    return bool(re.match(r"^(https?://)?(www\.)?github\.com/", text, re.IGNORECASE))


def normalize_github_eval_status(value):
    normalized = normalize_assessment_status(value)
    if normalized:
        return normalized

    key = normalize_key(value)
    if key in {
        "notevaluated",
        "evaluationfailed",
        "invalidgithublink",
        "invalidlink",
        "missinggithublink",
        "missinglink",
        "blanklink",
    }:
        return "Not Evaluated"
    return ""


def detect_github_score_workbook(df):
    if df is None or df.empty:
        return False

    normalized = {normalize_key(col) for col in df.columns}
    has_name = bool(normalized & {"candidatename", "studentname", "trainee"})
    if not has_name:
        return False

    cycle_columns = [column for column in df.columns if parse_github_cycle_column(column)[0]]
    if not cycle_columns:
        return False

    score_like_count = 0
    link_like_count = 0
    for column in cycle_columns:
        sample = df[column].dropna().astype(str).str.strip()
        sample = sample[sample.ne("")]
        for value in sample.head(12):
            if is_github_link_text(value):
                link_like_count += 1
                continue
            if normalize_github_eval_status(value):
                score_like_count += 1
                continue
            numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            if pd.notna(numeric_value):
                score_like_count += 1

    return score_like_count > 0 and score_like_count >= link_like_count


def build_eval_dataframe_from_github_score_workbook(df, source_name, sheet_name=None):
    if df is None or df.empty:
        return pd.DataFrame(columns=EVAL_COLUMNS), []

    working_df = df.copy()
    working_df.columns = [str(col).strip() for col in working_df.columns]
    column_lookup = {normalize_key(column): column for column in working_df.columns}
    name_col = next((column_lookup[key] for key in ["candidatename", "studentname", "trainee"] if key in column_lookup), None)
    superset_col = next(
        (
            column_lookup[key]
            for key in ["supersetid", "superset", "studentsupersetid", "studentid", "candidateid"]
            if key in column_lookup
        ),
        None,
    )
    technology_col = next(
        (
            column_lookup[key]
            for key in ["technologyframework", "technology", "framework", "declaredtechnology"]
            if key in column_lookup
        ),
        None,
    )
    cycle_columns = [column for column in working_df.columns if parse_github_cycle_column(column)[0]]
    worksheet_label = f"{source_name} :: {sheet_name}" if sheet_name else source_name

    records = []
    ignored_link_cells = 0
    for _, row in working_df.iterrows():
        candidate_name = str(row.get(name_col, "")).strip() if name_col else ""
        if not candidate_name:
            continue

        superset_id = normalize_superset_id(row.get(superset_col, "")) if superset_col else ""
        declared_technology = str(row.get(technology_col, "")).strip() if technology_col else ""

        for column in cycle_columns:
            evaluation_track, cycle_label = parse_github_cycle_column(column)
            cell_value = row.get(column, "")
            cell_text = str(cell_value).strip()
            if cell_text.lower() in {"", "nan", "none", "null"}:
                continue
            if is_github_link_text(cell_text):
                ignored_link_cells += 1
                continue

            numeric_score = pd.to_numeric(pd.Series([cell_value]), errors="coerce").iloc[0]
            status = "Evaluated"
            score = np.nan
            if pd.notna(numeric_score):
                score = float(numeric_score)
            else:
                normalized_status = normalize_github_eval_status(cell_text)
                if not normalized_status:
                    continue
                status = normalized_status
                score = 0.0

            records.append(
                {
                    "Superset ID": superset_id,
                    "Candidate Name": candidate_name,
                    "AssessmentWeek": cycle_label,
                    "EvaluationTrack": evaluation_track,
                    "Score": score,
                    "PrimaryGap": EVAL_DEFAULTS["PrimaryGap"],
                    "Code Quality": 0,
                    "Structure": 0,
                    "Logic Correctness": 0,
                    "Security": 0,
                    "Documentation": 0,
                    "FilesAnalyzed": 0,
                    "ExperienceProfile": "",
                    "ReviewStrictness": "",
                    "Status": status,
                    "Worksheet": worksheet_label,
                    "DeclaredTechnology": declared_technology,
                    "LanguagesTested": "",
                }
            )

    if not records:
        return pd.DataFrame(columns=EVAL_COLUMNS), []

    eval_df, notices = normalize_eval_dataframe(pd.DataFrame(records))
    notices.append(
        f"Loaded {len(eval_df)} GitHub score record(s) from '{worksheet_label}' using the scored cycle columns."
    )
    if ignored_link_cells:
        notices.append(
            f"Ignored {ignored_link_cells} GitHub link cell(s) in '{worksheet_label}' because only scored/status values are imported into the executive flow."
        )
    return eval_df, notices


def derive_batch_label(sheet_name, sheet_index):
    if sheet_name:
        match = re.search(r"\bbatch[\s\-_]*0*(\d+)\b", str(sheet_name), re.IGNORECASE)
        if match:
            return f"Batch {int(match.group(1))}"
    if sheet_index is not None:
        return f"Batch {sheet_index + 1}"
    return None


def detect_lms_file(df):
    normalized = {normalize_key(col) for col in df.columns}
    if "studentname" in normalized or "studentdetails" in normalized:
        return True
    if not df.empty:
        first_row = " ".join(df.iloc[0].astype(str).tolist()).lower()
        return "studentname" in normalize_key(first_row)
    return False


def is_top_brains_practical_source(source_name):
    source_key = normalize_key(source_name)
    return any(
        token in source_key
        for token in [
            "topbrainsassessmentscores",
            "topbrainstoolassessment",
            "topbrainstoolassessments",
            "topbrainsassessment",
        ]
    )


def detect_eval_file(df):
    normalized = {normalize_key(col) for col in df.columns}
    name_keys = {"candidatename", "trainee", "studentname"}
    eval_keys = {
        "score",
        "aiscore",
        "codescore",
        "primarygap",
        "criticalgap",
        "codequality",
        "structure",
        "logiccorrectness",
        "security",
        "documentation",
    }
    return bool(normalized & name_keys) and bool(normalized & eval_keys)


def detect_tracker_file(df):
    if df is None or df.empty:
        return False
    normalized = {normalize_key(col) for col in df.columns}
    has_name = bool(normalized & {"candidatename", "studentname", "trainee"})
    has_week = any(extract_week_label(col) for col in df.columns)
    has_batch = bool(normalized & {"assignedbatch", "batch"})
    return has_name and (has_week or has_batch)


def collapse_duplicate_tracker_columns(df):
    if df.empty or not df.columns.duplicated().any():
        return df, []

    notices = []
    collapsed = pd.DataFrame(index=df.index)
    seen_columns = []

    for column in df.columns:
        if column in seen_columns:
            continue
        seen_columns.append(column)
        subset = df.loc[:, df.columns == column]
        if subset.shape[1] == 1:
            collapsed[column] = subset.iloc[:, 0]
            continue

        if column.endswith((" - Rating", " - TotalScore", " - TotalMarks")):
            merged_series = subset.apply(pd.to_numeric, errors="coerce").bfill(axis=1).iloc[:, 0]
        else:
            text_subset = subset.fillna("").astype(str)

            def pick_text(row):
                values = [value.strip() for value in row.tolist() if value.strip() and value.strip().lower() != "nan"]
                return values[-1] if values else ""

            merged_series = text_subset.apply(pick_text, axis=1)

        collapsed[column] = merged_series
        notices.append(f"Duplicate tracker columns were standardized into '{column}'.")

    return collapsed, notices


def normalize_eval_dataframe(df):
    notices = []
    rename_map = {}
    aliases = {
        "Superset ID": {"supersetid", "superset", "studentsupersetid", "studentid", "candidateid"},
        "Candidate Name": {"candidatename", "trainee", "studentname"},
        "AssessmentWeek": {"assessmentweek", "week", "weeklabel"},
        "EvaluationTrack": {"evaluationtrack", "scoretype", "githubtrack", "evaluationtype"},
        "Score": {"score", "aiscore", "codescore", "practicalscore"},
        "PrimaryGap": {"primarygap", "criticalgap", "primarytechgap"},
        "Code Quality": {"codequality", "quality"},
        "Structure": {"structure", "modularization"},
        "Logic Correctness": {"logiccorrectness", "logic", "logicscore"},
        "Security": {"security", "securityscore"},
        "Documentation": {"documentation", "doc", "docscore"},
        "FilesAnalyzed": {"filesanalyzed"},
        "ExperienceProfile": {"experienceprofile"},
        "ReviewStrictness": {"reviewstrictness"},
        "Status": {"status"},
        "Worksheet": {"worksheet"},
        "DeclaredTechnology": {"declaredtechnology"},
        "LanguagesTested": {"languagestested"},
    }

    for column in df.columns:
        key = normalize_key(column)
        for target, keys in aliases.items():
            if key in keys and target not in rename_map.values():
                rename_map[column] = target
                break

    normalized_df = df.rename(columns=rename_map).copy()
    missing_columns = []
    for column in EVAL_COLUMNS:
        if column not in normalized_df.columns:
            normalized_df[column] = EVAL_DEFAULTS[column]
            missing_columns.append(column)

    normalized_df = normalized_df[EVAL_COLUMNS].copy()
    normalized_df["Superset ID"] = normalized_df["Superset ID"].apply(normalize_superset_id)
    normalized_df["Candidate Name"] = normalized_df["Candidate Name"].astype(str).str.strip()
    for column in ["Score", "Code Quality", "Structure", "Logic Correctness", "Security", "Documentation"]:
        normalized_df[column] = pd.to_numeric(normalized_df[column], errors="coerce").fillna(EVAL_DEFAULTS[column])
    normalized_df["FilesAnalyzed"] = pd.to_numeric(normalized_df["FilesAnalyzed"], errors="coerce").fillna(EVAL_DEFAULTS["FilesAnalyzed"]).astype(int)
    normalized_df["PrimaryGap"] = normalized_df["PrimaryGap"].fillna(EVAL_DEFAULTS["PrimaryGap"]).astype(str).str.strip()
    for column in ["AssessmentWeek", "ExperienceProfile", "ReviewStrictness", "Status", "Worksheet", "DeclaredTechnology", "LanguagesTested"]:
        normalized_df[column] = normalized_df[column].fillna(EVAL_DEFAULTS[column]).astype(str).str.strip()
    normalized_df["EvaluationTrack"] = normalized_df["EvaluationTrack"].apply(normalize_evaluation_track)
    normalized_df = normalized_df[normalized_df["Candidate Name"].astype(str).str.strip().ne("")]

    if missing_columns:
        notices.append("Some evaluation columns were missing and default values were applied: " + ", ".join(missing_columns))

    return normalized_df, notices


def build_eval_dataframe_from_task1_session():
    session_results = st.session_state.get("evaluation_results", [])
    if not session_results:
        return pd.DataFrame(columns=EVAL_COLUMNS), []

    records = []
    for item in session_results:
        if isinstance(item, dict):
            status = str(item.get("status", "")).strip()
            trainee = item.get("trainee", "")
            score_breakdown = item.get("score_breakdown", {}) or {}
            repository = item
        else:
            status = str(getattr(item, "status", "")).strip()
            trainee = getattr(item, "trainee", "")
            score_breakdown = getattr(item, "score_breakdown", {}) or {}
            repository = None

        if status != "Evaluated":
            continue

        getter = (lambda key, default="": repository.get(key, default)) if repository is not None else (lambda key, default="": getattr(item, key, default))
        records.append(
            {
                "Superset ID": normalize_superset_id(getter("superset_id", getter("SupersetID", ""))),
                "Candidate Name": str(trainee).strip(),
                "AssessmentWeek": str(getter("assessment_week", getter("AssessmentWeek", ""))).strip(),
                "EvaluationTrack": normalize_evaluation_track(getter("evaluation_track", getter("EvaluationTrack", ""))),
                "Score": getter("score", 0),
                "PrimaryGap": getter("primary_gap", EVAL_DEFAULTS["PrimaryGap"]),
                "Code Quality": score_breakdown.get("Code Quality", 0),
                "Structure": score_breakdown.get("Structure", 0),
                "Logic Correctness": score_breakdown.get("Logic Correctness", 0),
                "Security": score_breakdown.get("Security", 0),
                "Documentation": score_breakdown.get("Documentation", 0),
                "FilesAnalyzed": getter("files_analyzed", 0),
                "ExperienceProfile": getter("experience_profile", ""),
                "ReviewStrictness": getter("review_strictness", ""),
                "Status": status,
                "Worksheet": getter("worksheet", ""),
                "DeclaredTechnology": getter("declared_technology", ""),
                "LanguagesTested": getter("languages_tested", ""),
            }
        )

    if not records:
        return pd.DataFrame(columns=EVAL_COLUMNS), []

    session_df = pd.DataFrame(records)
    normalized_df, notices = normalize_eval_dataframe(session_df)
    notices.append(f"Live Task-1 evaluation results detected in the current session: {len(normalized_df)} practical records merged.")
    return normalized_df, notices


def deduplicate_candidate_records(df, metric_column=None, preferred_columns=None, context_label="records", partition_columns=None):
    if df.empty or "Candidate Name" not in df.columns:
        return df, []

    working_df = df.copy()
    if "Merge_Key" not in working_df.columns:
        working_df["Merge_Key"] = working_df.apply(build_merge_key, axis=1)
    partition_keys = partition_columns or ["Merge_Key"]

    duplicate_mask = working_df.duplicated(subset=partition_keys, keep=False) & working_df["Merge_Key"].ne("")
    if not duplicate_mask.any():
        return working_df, []

    notices = []
    completeness_columns = preferred_columns or [col for col in working_df.columns if col not in {"Merge_Key"}]

    def score_row(row):
        completeness = 0
        for column in completeness_columns:
            value = row.get(column)
            if pd.notna(value) and str(value).strip() not in {"", "N/A", "nan"}:
                completeness += 1
        metric_value = pd.to_numeric(row.get(metric_column, 0), errors="coerce") if metric_column else 0
        metric_value = 0 if pd.isna(metric_value) else metric_value
        return pd.Series({"_completeness": completeness, "_metric": metric_value})

    score_frame = working_df.apply(score_row, axis=1)
    working_df = pd.concat([working_df, score_frame], axis=1)
    working_df = working_df.sort_values(by=[*partition_keys, "_metric", "_completeness"], ascending=[True] * len(partition_keys) + [False, False])
    working_df = working_df.drop_duplicates(subset=partition_keys, keep="first")
    working_df = working_df.drop(columns=["_completeness", "_metric"])
    return working_df, notices


def normalize_tracker_dataframe(df, source_name, sheet_name=None, sheet_index=None, include_sheet_in_source=False):
    normalized_df = df.copy()
    normalized_df.columns = [str(col).strip() for col in normalized_df.columns]
    rename_map = {}
    for column in normalized_df.columns:
        key = normalize_key(column)
        if key in {"candidatename", "studentname", "trainee"}:
            rename_map[column] = "Candidate Name"
        elif key in {"assignedbatch", "batch"}:
            rename_map[column] = "Assigned Batch"
        elif key in {"sourcefile", "source"}:
            rename_map[column] = "Source_File"
        else:
            week_label = extract_week_label(column)
            if week_label:
                lower_column = str(column).lower()
                if "rating" in lower_column:
                    rename_map[column] = f"{week_label} - Rating"
                elif "totalscore" in normalize_key(column):
                    rename_map[column] = f"{week_label} - TotalScore"
                elif "totalmarks" in normalize_key(column):
                    rename_map[column] = f"{week_label} - TotalMarks"
                elif "feedback" in lower_column or "comment" in lower_column:
                    rename_map[column] = f"{week_label} - Feedback"
                elif "topic" in lower_column:
                    rename_map[column] = f"{week_label} - Topic"

    normalized_df.rename(columns=rename_map, inplace=True)
    normalized_df, duplicate_notices = collapse_duplicate_tracker_columns(normalized_df)
    if "Candidate Name" in normalized_df.columns:
        normalized_df["Candidate Name"] = normalized_df["Candidate Name"].astype(str).str.strip()

    topic_columns = [col for col in normalized_df.columns if col.endswith(" - Topic")]
    for column in topic_columns:
        normalized_df[column] = normalized_df[column].ffill()

    if "Assigned Batch" not in normalized_df.columns:
        sheet_batch = derive_batch_label(sheet_name, sheet_index)
        if sheet_batch:
            normalized_df["Assigned Batch"] = sheet_batch
        else:
            batches = []
            current_batch = 1
            sl_col = next((col for col in normalized_df.columns if "sl" in str(col).lower() and "no" in str(col).lower()), None)
            for idx, row in normalized_df.iterrows():
                if sl_col and pd.notna(row.get(sl_col)):
                    value = str(row[sl_col]).strip()
                    if value in {"1", "1.0"} and idx > 0:
                        current_batch += 1
                batches.append(f"Batch {current_batch}")
            normalized_df["Assigned Batch"] = batches

    source_label = source_name
    if include_sheet_in_source and sheet_name:
        source_label = f"{source_name} :: {sheet_name}"
    normalized_df["Source_File"] = source_label
    return normalized_df, duplicate_notices


def build_lms_dataframe(df, file_name):
    df.columns = df.columns.str.strip()
    if "studentName" not in df.columns:
        return pd.DataFrame()

    df = df.dropna(subset=["studentName"]).copy()
    week_label = extract_week_label(file_name)
    if not week_label:
        file_match = re.search(r"(?:assessment|test)[\s\-_]*0*(\d+)", file_name, re.IGNORECASE)
        week_num = int(file_match.group(1)) if file_match else 1
        week_label = f"Week {week_num}"

    result = pd.DataFrame()
    result["Candidate Name"] = df["studentName"].astype(str).str.strip()
    result["Assigned Batch"] = "Top Brains Tool"
    result["Source_File"] = file_name

    raw_score = pd.to_numeric(df.get("totalScore", 0), errors="coerce").fillna(0)
    total_marks = pd.to_numeric(df.get("totalMarks", 60), errors="coerce").replace(0, 60).fillna(60)
    percent = (raw_score / total_marks) * 100
    stars = (percent / 20).round().clip(1, 5)

    result[f"{week_label} - Rating"] = stars
    result[f"{week_label} - TotalScore"] = raw_score
    result[f"{week_label} - TotalMarks"] = total_marks
    result[f"{week_label} - Feedback"] = "Subjectivity Score: " + percent.round(0).astype(int).astype(str) + "% (Automated Theory)"
    result[f"{week_label} - Topic"] = "Fundamental Training"
    result["Sl No"] = range(1, len(result) + 1)
    return result


def derive_top_brains_topic(sheet_name, df):
    cleaned_sheet_name = re.sub(
        r"^\s*\d{1,2}\s*[- ]?\s*[A-Za-z]{3,9}\s*[- ]?\s*\d{2,4}\s*[- ]?\s*",
        "",
        str(sheet_name or ""),
        flags=re.IGNORECASE,
    ).strip(" -_")
    if cleaned_sheet_name and re.search(r"[A-Za-z]{3,}", cleaned_sheet_name):
        return cleaned_sheet_name

    section_columns = [str(column).strip() for column in df.columns if str(column).strip().lower().startswith("section ")]
    if section_columns:
        section_name = re.sub(r"^section\s*", "", section_columns[0], flags=re.IGNORECASE)
        section_name = re.sub(r"\s*-\s*data\s*&\s*ai\s*$", "", section_name, flags=re.IGNORECASE).strip(" -_")
        if section_name:
            return section_name

    return str(sheet_name).strip()


def read_top_brains_practical_sheet(excel, sheet_name):
    preview_df = pd.read_excel(excel, sheet_name=sheet_name, header=None, nrows=6)
    header_row = 0
    for row_index in range(min(len(preview_df), 4)):
        row_values = [normalize_key(value) for value in preview_df.iloc[row_index].tolist()]
        if "studentname" in row_values:
            header_row = row_index
            break

    practical_df = pd.read_excel(excel, sheet_name=sheet_name, header=header_row)
    practical_df.columns = practical_df.columns.astype(str).str.strip()
    standardized_columns = {}
    for column_name in practical_df.columns:
        normalized_column = normalize_key(column_name)
        if normalized_column == "supersetid":
            standardized_columns[column_name] = "Superset ID"
        elif normalized_column == "studentname":
            standardized_columns[column_name] = "studentName"
        elif normalized_column == "totalscore":
            standardized_columns[column_name] = "totalScore"
        elif normalized_column == "totalmarks":
            standardized_columns[column_name] = "totalMarks"
        elif normalized_column == "result":
            standardized_columns[column_name] = "result"
    if standardized_columns:
        practical_df = practical_df.rename(columns=standardized_columns)

    superset_row_index = header_row + 1
    if superset_row_index < len(preview_df):
        rename_map = {}
        support_row = preview_df.iloc[superset_row_index].tolist()
        for column_index, column_name in enumerate(practical_df.columns):
            if not str(column_name).startswith("Unnamed"):
                continue
            if column_index >= len(support_row):
                continue
            support_value = str(support_row[column_index] or "").strip()
            if normalize_key(support_value) == "supersetid":
                rename_map[column_name] = "Superset ID"
        if rename_map:
            practical_df = practical_df.rename(columns=rename_map)

    return practical_df


def build_top_brains_practical_dataframe(df, source_name, sheet_name, week_label):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    if "studentName" not in df.columns:
        return pd.DataFrame()

    df = df.dropna(subset=["studentName"]).copy()
    topic_value = derive_top_brains_topic(sheet_name, df)
    total_score = pd.to_numeric(df.get("totalScore", 0), errors="coerce").fillna(0)
    total_marks = pd.to_numeric(df.get("totalMarks", 100), errors="coerce").replace(0, 100).fillna(100)
    practical_percent = ((total_score / total_marks) * 100).clip(0, 100).round(1)
    status_values = df.get("result", pd.Series("", index=df.index)).apply(normalize_assessment_status)

    result = pd.DataFrame()
    result["Candidate Name"] = df["studentName"].astype(str).str.strip()
    normalized_column_lookup = {normalize_key(column): column for column in df.columns}
    superset_col = next(
        (
            normalized_column_lookup[key]
            for key in ["supersetid", "studentsupersetid", "studentid", "candidateid"]
            if key in normalized_column_lookup
        ),
        None,
    )
    if superset_col is not None:
        result["Superset ID"] = df[superset_col].apply(normalize_superset_id)
    result["Assigned Batch"] = "Top Brains Tool"
    result["Source_File"] = source_name
    result[f"{week_label} - TotalScore"] = total_score
    result[f"{week_label} - TotalMarks"] = total_marks
    result[f"{week_label} - Feedback"] = status_values.replace("", "N/A")
    result[f"{week_label} - PracticalScore"] = practical_percent
    result[f"{week_label} - Topic"] = topic_value
    return result


def first_non_empty(series):
    for value in series:
        if pd.notna(value):
            text = str(value).strip()
            if text and text.lower() != "nan":
                return value
    return np.nan


def pick_topic_value(series):
    values = []
    for value in series:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if not text or text.lower() == "nan":
            continue
        values.append(text)
    if not values:
        return np.nan

    preferred = [value for value in values if value.lower() != "fundamental training"]
    if preferred:
        return preferred[0]
    return values[0]


def aggregate_tracker_scope(df):
    if df.empty or "Candidate Name" not in df.columns:
        return df

    working_df = df.copy()
    working_df["Candidate Name"] = working_df["Candidate Name"].astype(str).str.strip()
    if "Superset ID" in working_df.columns:
        working_df["Superset ID"] = working_df["Superset ID"].apply(normalize_superset_id)
    working_df["Merge_Key"] = working_df.apply(build_merge_key, axis=1)

    group_keys = ["Merge_Key"]

    week_numeric_pattern = re.compile(r" - (Rating|TotalScore|TotalMarks|PracticalScore)$")
    week_topic_pattern = re.compile(r" - Topic$")

    rows = []
    for _, group in working_df.groupby(group_keys, dropna=False, sort=False):
        row = {}
        for column in working_df.columns:
            if column == "Merge_Key":
                row[column] = group[column].iloc[0]
                continue
            if column == "Candidate Name":
                row[column] = first_non_empty(group[column])
                continue
            if column == "Source_File":
                row[column] = first_non_empty(group[column])
                continue
            if week_topic_pattern.search(str(column)):
                row[column] = pick_topic_value(group[column])
                continue
            if week_numeric_pattern.search(str(column)):
                numeric_values = pd.to_numeric(group[column], errors="coerce")
                row[column] = numeric_values.mean(skipna=True) if numeric_values.notna().any() else np.nan
                continue
            if str(column).lower() in {"sl no", "sl no."}:
                numeric_values = pd.to_numeric(group[column], errors="coerce")
                row[column] = numeric_values.min(skipna=True) if numeric_values.notna().any() else np.nan
                continue
            row[column] = first_non_empty(group[column])
        rows.append(row)

    aggregated_df = pd.DataFrame(rows)
    return aggregated_df.drop(columns=["Merge_Key"], errors="ignore")


def merge_top_brains_practical_tracker_data(tracker_res, practical_frames):
    if not practical_frames:
        return tracker_res

    practical_res = pd.concat([frame for frame in practical_frames if not frame.empty], ignore_index=True)
    if practical_res.empty:
        return tracker_res

    practical_res["Candidate Name"] = practical_res["Candidate Name"].astype(str).str.strip()
    if "Superset ID" in practical_res.columns:
        practical_res["Superset ID"] = practical_res["Superset ID"].apply(normalize_superset_id)
    practical_res["Merge_Key"] = practical_res.apply(build_merge_key, axis=1)
    practical_res = (
        practical_res.groupby("Merge_Key", as_index=False)
        .agg({column: first_non_empty for column in practical_res.columns if column != "Merge_Key"})
        .copy()
    )

    if tracker_res.empty:
        return practical_res.drop(columns=["Merge_Key"], errors="ignore")

    tracker_merge = tracker_res.copy()
    tracker_merge["Candidate Name"] = tracker_merge["Candidate Name"].astype(str).str.strip()
    if "Superset ID" in tracker_merge.columns:
        tracker_merge["Superset ID"] = tracker_merge["Superset ID"].apply(normalize_superset_id)
    tracker_merge["Merge_Key"] = tracker_merge.apply(build_merge_key, axis=1)

    merged = pd.merge(tracker_merge, practical_res, on="Merge_Key", how="outer", suffixes=("", "__practical"))
    for base_column in ["Candidate Name", "Assigned Batch", "Source_File", "Superset ID"]:
        practical_column = f"{base_column}__practical"
        if practical_column in merged.columns:
            merged[base_column] = merged.get(base_column, pd.Series(index=merged.index, dtype=object)).fillna(merged[practical_column])

    for column in practical_res.columns:
        if column in {"Merge_Key", "Candidate Name", "Assigned Batch", "Source_File", "Superset ID"}:
            continue
        practical_column = f"{column}__practical"
        if practical_column not in merged.columns:
            continue
        if column not in merged.columns:
            merged[column] = merged[practical_column]
            continue
        if column.endswith(" - Topic"):
            placeholder_mask = (
                merged[column].isna()
                | merged[column].astype(str).str.strip().eq("")
                | merged[column].astype(str).str.strip().eq("Fundamental Training")
            )
            merged[column] = merged[column].where(~placeholder_mask, merged[practical_column])
            continue
        if column.endswith(" - Feedback"):
            placeholder_mask = (
                merged[column].isna()
                | merged[column].astype(str).str.strip().isin(["", "N/A", "nan"])
            )
            merged[column] = merged[column].where(~placeholder_mask, merged[practical_column])
            continue
        merged[column] = merged[column].fillna(merged[practical_column])

    cleanup_columns = [column for column in merged.columns if column.endswith("__practical")] + ["Merge_Key"]
    return merged.drop(columns=cleanup_columns, errors="ignore")


def build_combined_tracker_dataset(raw_tracker_df, practical_tracker_frames=None):
    if raw_tracker_df is None or raw_tracker_df.empty:
        combined_df = pd.DataFrame()
    else:
        working_df = raw_tracker_df.copy()
        if "Source_File" in working_df.columns:
            source_series = working_df["Source_File"].fillna("").astype(str)
            top_brains_mask = source_series.apply(is_top_brains_practical_source)
            base_tracker_df = working_df.loc[~top_brains_mask].copy()
            top_brains_frames = [group.copy() for _, group in working_df.loc[top_brains_mask].groupby("Source_File", dropna=False, sort=False)]
        else:
            base_tracker_df = working_df
            top_brains_frames = []
        combined_df = merge_top_brains_practical_tracker_data(base_tracker_df, top_brains_frames)

    if practical_tracker_frames:
        combined_df = merge_top_brains_practical_tracker_data(combined_df, practical_tracker_frames)
    return combined_df


def split_tracker_sources(raw_tracker_df):
    if raw_tracker_df is None or raw_tracker_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    working_df = raw_tracker_df.copy()
    if "Source_File" not in working_df.columns:
        return working_df, pd.DataFrame()

    source_series = working_df["Source_File"].fillna("").astype(str)
    top_brains_mask = source_series.apply(is_top_brains_practical_source)
    subjectivity_df = working_df.loc[~top_brains_mask].copy()
    top_brains_df = working_df.loc[top_brains_mask].copy()
    return subjectivity_df, top_brains_df


def get_shared_eval_dataframe():
    uploaded_eval_df = pd.DataFrame(st.session_state.get("eval_results", []))
    session_eval_df, session_eval_notices = build_eval_dataframe_from_task1_session()
    eval_frames = []
    if not uploaded_eval_df.empty:
        eval_frames.append(uploaded_eval_df)
    if not session_eval_df.empty:
        eval_frames.append(session_eval_df)

    notices = list(session_eval_notices)
    eval_df = pd.concat(eval_frames, ignore_index=True) if eval_frames else pd.DataFrame(columns=EVAL_COLUMNS)
    if eval_df.empty:
        return eval_df, notices

    eval_df, eval_notices = normalize_eval_dataframe(eval_df)
    notices.extend(eval_notices)
    eval_df, dedupe_notices = deduplicate_candidate_records(
        eval_df,
        metric_column="Score",
        preferred_columns=EVAL_COLUMNS,
        context_label="evaluation results",
        partition_columns=["Merge_Key", "AssessmentWeek", "EvaluationTrack"] if "AssessmentWeek" in eval_df.columns else ["Merge_Key", "EvaluationTrack"],
    )
    notices.extend(dedupe_notices)
    return eval_df, list(dict.fromkeys(notices))


def filter_eval_dataframe_by_track(eval_df, track):
    if eval_df is None or eval_df.empty:
        return pd.DataFrame(columns=EVAL_COLUMNS)
    working_df = eval_df.copy()
    if "EvaluationTrack" not in working_df.columns:
        working_df["EvaluationTrack"] = GITHUB_ASSESSMENT_TRACK
    working_df["EvaluationTrack"] = working_df["EvaluationTrack"].apply(normalize_evaluation_track)
    return working_df[working_df["EvaluationTrack"] == normalize_evaluation_track(track)].copy()


def build_practical_tracker_from_eval_df(eval_df):
    if eval_df.empty or "AssessmentWeek" not in eval_df.columns:
        return pd.DataFrame()

    working_df = filter_eval_dataframe_by_track(eval_df, GITHUB_ASSESSMENT_TRACK)
    if working_df.empty:
        return pd.DataFrame()
    working_df["AssessmentWeek"] = working_df["AssessmentWeek"].astype(str).str.strip()
    working_df["Superset ID"] = working_df["Superset ID"].apply(normalize_superset_id) if "Superset ID" in working_df.columns else ""
    working_df = working_df[working_df["AssessmentWeek"].apply(extract_week_label).notna()].copy()
    if working_df.empty:
        return pd.DataFrame()

    working_df["WeekLabel"] = working_df["AssessmentWeek"].apply(extract_week_label)
    working_df["Candidate Name"] = working_df["Candidate Name"].astype(str).str.strip()
    working_df["Score"] = pd.to_numeric(working_df["Score"], errors="coerce").fillna(0)
    working_df["Merge_Key"] = working_df.apply(build_merge_key, axis=1)
    working_df, _ = deduplicate_candidate_records(
        working_df,
        metric_column="Score",
        preferred_columns=EVAL_COLUMNS,
        context_label="weekly evaluation results",
        partition_columns=["Merge_Key", "WeekLabel"],
    )

    rows = []
    for _, group in working_df.groupby("Merge_Key", sort=False):
        row = {
            "Superset ID": first_non_empty(group["Superset ID"]) if "Superset ID" in group.columns else "",
            "Candidate Name": first_non_empty(group["Candidate Name"]),
            "Assigned Batch": "Code Eval Batch",
            "Source_File": "Task1 Practical Evaluation",
        }
        for _, item in group.iterrows():
            week_label = item["WeekLabel"]
            row[f"{week_label} - PracticalScore"] = item["Score"]
        rows.append(row)

    return pd.DataFrame(rows)


def process_files(files):
    tracker_frames = []
    eval_frames = []
    top_brains_practical_frames = []
    notices = []

    for file in files:
        if file is None:
            continue

        source_name = file.name.replace(".xlsx", "").replace(".xlsm", "").replace(".csv", "")

        if file.name.endswith(".csv"):
            file.seek(0)
            peek = pd.read_csv(file, nrows=3)
            file.seek(0)
            if detect_eval_file(peek):
                eval_df = pd.read_csv(file)
                normalized_eval, eval_notices = normalize_eval_dataframe(eval_df)
                notices.extend(eval_notices)
                if not normalized_eval.empty:
                    eval_frames.append(normalized_eval)
                continue
            if detect_github_score_workbook(peek):
                scored_df = pd.read_csv(file)
                normalized_eval, eval_notices = build_eval_dataframe_from_github_score_workbook(scored_df, source_name)
                notices.extend(eval_notices)
                if not normalized_eval.empty:
                    eval_frames.append(normalized_eval)
                continue
            if detect_lms_file(peek):
                lms_df = pd.read_csv(file, header=1)
                tracker_frames.append(build_lms_dataframe(lms_df, source_name))
                continue
            tracker_df = pd.read_csv(file)
            if not detect_tracker_file(tracker_df):
                notices.append(f"Skipped '{source_name}' because no student tracker columns were detected.")
                continue
            normalized_tracker, tracker_notices = normalize_tracker_dataframe(tracker_df, source_name)
            notices.extend(tracker_notices)
            tracker_frames.append(normalized_tracker)
            continue

        file.seek(0)
        excel = pd.ExcelFile(file, engine="openpyxl")
        sheet_names = excel.sheet_names
        include_sheet = len(sheet_names) > 1

        for sheet_index, sheet_name in enumerate(sheet_names):
            peek = pd.read_excel(excel, sheet_name=sheet_name, nrows=3)
            if detect_eval_file(peek):
                eval_df = pd.read_excel(excel, sheet_name=sheet_name)
                normalized_eval, eval_notices = normalize_eval_dataframe(eval_df)
                notices.extend(eval_notices)
                if not normalized_eval.empty:
                    eval_frames.append(normalized_eval)
                continue
            if detect_github_score_workbook(peek):
                scored_df = pd.read_excel(excel, sheet_name=sheet_name)
                normalized_eval, eval_notices = build_eval_dataframe_from_github_score_workbook(
                    scored_df,
                    source_name,
                    sheet_name=sheet_name if include_sheet else None,
                )
                notices.extend(eval_notices)
                if not normalized_eval.empty:
                    eval_frames.append(normalized_eval)
                continue

            if detect_lms_file(peek):
                if is_top_brains_practical_source(source_name):
                    practical_df = read_top_brains_practical_sheet(excel, sheet_name)
                    week_label = f"Week {sheet_index + 1}"
                    top_brains_practical_frames.append(
                        build_top_brains_practical_dataframe(practical_df, source_name, sheet_name, week_label)
                    )
                    continue
                lms_df = pd.read_excel(excel, sheet_name=sheet_name, header=1)
                lms_label = f"{source_name} :: {sheet_name}" if include_sheet else source_name
                tracker_frames.append(build_lms_dataframe(lms_df, lms_label))
                continue

            tracker_df = pd.read_excel(excel, sheet_name=sheet_name)
            if not detect_tracker_file(tracker_df):
                notices.append(f"Skipped sheet '{sheet_name}' in '{source_name}' because it does not contain student tracker data.")
                continue
            normalized_tracker, tracker_notices = normalize_tracker_dataframe(
                tracker_df,
                source_name,
                sheet_name=sheet_name,
                sheet_index=sheet_index if include_sheet else None,
                include_sheet_in_source=include_sheet,
            )
            notices.extend(tracker_notices)
            tracker_frames.append(normalized_tracker)

    tracker_sources = [frame for frame in tracker_frames if not frame.empty] + [frame for frame in top_brains_practical_frames if not frame.empty]
    tracker_res = pd.concat(tracker_sources, ignore_index=True) if tracker_sources else pd.DataFrame()
    eval_res = pd.concat(eval_frames, ignore_index=True) if eval_frames else pd.DataFrame(columns=EVAL_COLUMNS)

    if not tracker_res.empty and "Candidate Name" in tracker_res.columns:
        tracker_res["Candidate Name"] = tracker_res["Candidate Name"].astype(str).str.strip()
    if not eval_res.empty:
        eval_res["Candidate Name"] = eval_res["Candidate Name"].astype(str).str.strip()

    return tracker_res, eval_res, notices


def build_week_map(df):
    week_map = {}
    for column in df.columns:
        week_label = extract_week_label(column)
        if not week_label:
            continue
        week_entry = week_map.setdefault(week_label, {})
        lower_column = str(column).lower()
        if "rating" in lower_column:
            week_entry["rating"] = column
        elif "totalscore" in normalize_key(column):
            week_entry["total_score"] = column
        elif "totalmarks" in normalize_key(column):
            week_entry["total_marks"] = column
        elif "feedback" in lower_column or "comment" in lower_column:
            week_entry["feedback"] = column
        elif "topic" in lower_column:
            week_entry["topic"] = column
        elif "practicalscore" in normalize_key(column):
            week_entry["practical_score"] = column

    filtered_week_map = {}
    for week_label, entry in week_map.items():
        has_values = False
        for column in entry.values():
            if column in df.columns and df[column].notna().any():
                has_values = True
                break
        if has_values:
            filtered_week_map[week_label] = entry

    return dict(
        sorted(
            filtered_week_map.items(),
            key=lambda item: int(re.search(r"(\d+)", item[0]).group(1)) if re.search(r"(\d+)", item[0]) else 999,
        )
    )


def generate_persona(row, rating_col):
    theory = pd.to_numeric(row.get(rating_col, 0), errors="coerce")
    practical = pd.to_numeric(row.get("Score", 0), errors="coerce")
    theory = 0 if pd.isna(theory) else theory
    practical = 0 if pd.isna(practical) else practical

    has_theory = bool(rating_col) and theory > 0
    has_practical = practical > 0

    if has_theory and has_practical:
        if theory >= 3.5 and practical >= 60:
            return "Champion (Strong Subjectivity & Practice)"
        if theory < 3.5 and practical >= 60:
            return "Quiet Coder (Weak Tests, Strong Code)"
        if theory >= 3.5 and practical < 60:
            return "Subjectivity Heavy (Strong Observations, Weak Code)"
        return "High Risk (Struggling Overall)"

    if has_theory:
        if theory >= 4:
            return "Subjectivity Anchor (Strong Subjectivity)"
        if theory >= 2.5:
            return "Needs Practice Exposure"
        return "Instructor Support Required"

    if has_practical:
        if practical >= 75:
            return "Code Anchor (Strong Practicals)"
        if practical >= 50:
            return "Code Developing (Improving Practicals)"
        return "Code Support Needed"

    return "Unclassified"


def calc_unified(row, rating_col):
    theory = pd.to_numeric(row.get(rating_col, 0), errors="coerce")
    practical = pd.to_numeric(row.get("Score", 0), errors="coerce")
    theory = 0 if pd.isna(theory) else theory
    practical = 0 if pd.isna(practical) else practical

    theory_percent = theory * 20
    if theory_percent > 0 and practical > 0:
        return (theory_percent + practical) / 2
    if theory_percent > 0:
        return theory_percent
    if practical > 0:
        return practical
    return 0


def detect_persona_mode(active_df, eval_df):
    has_theory = not active_df.empty
    has_practical = False
    if not eval_df.empty and "Score" in eval_df.columns:
        practical_scores = pd.to_numeric(eval_df["Score"], errors="coerce").fillna(0)
        has_practical = practical_scores.gt(0).any()
    elif not active_df.empty:
        practical_columns = [column for column in active_df.columns if column.endswith(" - PracticalScore")]
        if practical_columns:
            practical_scores = average_numeric_columns(active_df, practical_columns)
            has_practical = practical_scores.gt(0).any()
    if has_theory and has_practical:
        return "full"
    if has_theory or has_practical:
        return "limited"
    return "empty"


def generate_student_summary(student_row, rating_col, feedback_col):
    parts = []
    theory = pd.to_numeric(student_row.get(rating_col, 0), errors="coerce") if rating_col else 0
    practical = pd.to_numeric(student_row.get("Score", 0), errors="coerce")
    theory = 0 if pd.isna(theory) else theory
    practical = 0 if pd.isna(practical) else practical
    persona = str(student_row.get("Student Persona", "Unclassified"))
    trainer_feedback = str(student_row.get(feedback_col, "")).strip() if feedback_col else ""

    if theory >= 4:
        parts.append("This student shows strong subjectivity performance from trainer observations.")
    elif theory >= 2.5:
        parts.append("This student has a workable subjectivity base but still needs consistency.")
    elif theory > 0:
        parts.append("This student needs more support on core subjectivity fundamentals.")

    if practical >= 75:
        parts.append("Practical coding output is strong and trending above the cohort average.")
    elif practical >= 50:
        parts.append("Practical coding is developing, with room to improve structure and execution speed.")
    elif practical > 0:
        parts.append("Practical coding needs stronger hands-on reinforcement and guided review.")

    if persona != "Unclassified":
        parts.append(f"Current persona classification: {persona}.")
    if trainer_feedback and trainer_feedback.lower() not in {"nan", "n/a", "no feedback available"} and not is_status_feedback_text(trainer_feedback):
        if "automated theory" in trainer_feedback.lower():
            parts.append("Trainer feedback is currently system-generated from subjectivity scores.")
        else:
            parts.append(f"Trainer feedback suggests: {trainer_feedback}")
    if not parts:
        parts.append("This student has limited available data, so the summary is based on the uploaded records currently available.")
    return " ".join(parts)


def build_curriculum_directives(theme_counts, week_label, topic_label, has_practical_data):
    directives = []
    topic_text = topic_label or "the current module"
    ranked = sorted(theme_counts.items(), key=lambda item: item[1], reverse=True)
    for theme, count in ranked:
        if count <= 0:
            continue
        if theme == "Needs Applied Practice":
            guided_text = "guided coding" if has_practical_data else "guided practice"
            directives.append(
                {
                    "Focus Area": "Hands-on Practice",
                    "Action": f"Increase coding lab time in {week_label}",
                    "Priority": "High",
                    "Why This Week": f"{count} learners need more {guided_text} on {topic_text}.",
                }
            )
        elif theme == "Logic & Syntax Issues":
            directives.append(
                {
                    "Focus Area": "Logic Reinforcement",
                    "Action": f"Run a debugging clinic for {week_label}",
                    "Priority": "High",
                    "Why This Week": f"{count} learners need stronger algorithm and syntax correction practice.",
                }
            )
        elif theme == "Low Classroom Focus":
            directives.append(
                {
                    "Focus Area": "Classroom Engagement",
                    "Action": "Add short participation checkpoints",
                    "Priority": "Medium",
                    "Why This Week": f"{count} learners show participation risk and need tighter trainer touchpoints.",
                }
            )
        elif theme == "Strong Concept Grasp":
            directives.append(
                {
                    "Focus Area": "Advanced Extension",
                    "Action": "Create stretch tasks for faster learners",
                    "Priority": "Medium",
                    "Why This Week": f"{count} learners are ahead and can take on deeper problem solving.",
                }
            )
        elif theme == "Steady & Consistent":
            directives.append(
                {
                    "Focus Area": "Consistency",
                    "Action": "Maintain revision and walkthrough cadence",
                    "Priority": "Low",
                    "Why This Week": f"{count} learners are stable and should keep the current reinforcement rhythm.",
                }
            )
    if not directives:
        directives.append(
            {
                "Focus Area": "Feedback Quality",
                "Action": f"Collect more trainer feedback for {week_label}",
                "Priority": "Medium",
                "Why This Week": "More qualitative input is needed before making curriculum changes confidently.",
            }
        )
    return pd.DataFrame(directives[:4])


def get_student_skill_snapshot(student_row):
    skill_columns = ["Code Quality", "Structure", "Logic Correctness", "Security", "Documentation"]
    available = {}
    for column in skill_columns:
        value = pd.to_numeric(student_row.get(column, 0), errors="coerce")
        available[column] = 0 if pd.isna(value) else float(value)
    if not any(value > 0 for value in available.values()):
        return {
            "has_code_signal": False,
            "strongest_skill": "N/A",
            "strongest_value": 0,
            "weakest_skill": "N/A",
            "weakest_value": 0,
        }
    strongest = max(available, key=available.get) if available else "N/A"
    weakest = min(available, key=available.get) if available else "N/A"
    return {
        "has_code_signal": True,
        "strongest_skill": strongest,
        "strongest_value": available.get(strongest, 0),
        "weakest_skill": weakest,
        "weakest_value": available.get(weakest, 0),
    }


def describe_graph(title, description):
    st.markdown(f"**{title}**")
    st.caption(description)


def average_numeric_columns(df, columns):
    valid_columns = [column for column in columns if column and column in df.columns]
    if not valid_columns:
        return pd.Series(0, index=df.index, dtype=float)
    return df[valid_columns].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)


def is_status_feedback_text(value):
    return bool(normalize_assessment_status(value))


def normalize_assessment_status(value):
    normalized = normalize_key(value)
    if not normalized or normalized in {"na", "n/a", "nan"}:
        return ""
    if normalized in {"pass", "passed"}:
        return "Pass"
    if normalized in {"absent", "ab"}:
        return "Absent"
    if normalized in {"noshow", "didntjoin", "didnotjoin", "notjoined", "notjoin"}:
        return "No Show"
    if normalized in {"dropout", "dropoutinactive", "dropoutorinactive", "dropoutstatus", "dropoutstudent"}:
        return "Drop out"
    return ""


def build_status_masks(df, score_series, text_columns):
    valid_columns = [column for column in text_columns if column and column in df.columns]
    if valid_columns:
        status_frame = pd.DataFrame(index=df.index)
        for column in valid_columns:
            status_frame[column] = df[column].apply(normalize_assessment_status)
        explicit_dropout_mask = status_frame.eq("Drop out").any(axis=1)
        explicit_absent_mask = status_frame.isin(["Absent", "No Show"]).any(axis=1)
        has_explicit_status = status_frame.ne("").any(axis=1)
    else:
        explicit_dropout_mask = pd.Series(False, index=df.index, dtype=bool)
        explicit_absent_mask = pd.Series(False, index=df.index, dtype=bool)
        has_explicit_status = pd.Series(False, index=df.index, dtype=bool)

    numeric_scores = pd.to_numeric(score_series, errors="coerce").fillna(0)
    has_score = numeric_scores.gt(0)
    evaluated_mask = has_score
    dropout_mask = explicit_dropout_mask & ~has_score
    absent_mask = (explicit_absent_mask & ~has_score) | (~has_score & ~has_explicit_status & ~dropout_mask)
    absent_mask = absent_mask & ~dropout_mask
    return evaluated_mask, absent_mask, dropout_mask


def build_explicit_exception_masks(df, week_map):
    if not week_map:
        empty = pd.Series("", index=df.index, dtype=object)
        false_mask = pd.Series(False, index=df.index, dtype=bool)
        return empty, false_mask, false_mask.copy(), false_mask.copy()

    latest_status_series = pd.Series("", index=df.index, dtype=object)
    for _, columns in week_map.items():
        feedback_col = columns.get("feedback")
        rating_col = columns.get("rating")
        total_score_col = columns.get("total_score")
        practical_score_col = columns.get("practical_score")

        feedback_series = (
            df[feedback_col].fillna("").astype(str).str.strip()
            if feedback_col and feedback_col in df.columns
            else pd.Series("", index=df.index, dtype=object)
        )
        explicit_status = feedback_series.apply(normalize_assessment_status)

        rating_signal = (
            pd.to_numeric(df[rating_col], errors="coerce").fillna(0).gt(0)
            if rating_col and rating_col in df.columns
            else pd.Series(False, index=df.index, dtype=bool)
        )
        total_score_signal = (
            pd.to_numeric(df[total_score_col], errors="coerce").fillna(0).gt(0)
            if total_score_col and total_score_col in df.columns
            else pd.Series(False, index=df.index, dtype=bool)
        )
        practical_signal = (
            pd.to_numeric(df[practical_score_col], errors="coerce").fillna(0).gt(0)
            if practical_score_col and practical_score_col in df.columns
            else pd.Series(False, index=df.index, dtype=bool)
        )
        meaningful_feedback_signal = (
            feedback_series.ne("")
            & ~feedback_series.str.lower().isin(["nan", "n/a"])
            & ~feedback_series.apply(is_status_feedback_text)
        )

        week_state = pd.Series("", index=df.index, dtype=object)
        active_signal = rating_signal | total_score_signal | practical_signal | meaningful_feedback_signal
        week_state = week_state.where(~active_signal, "Active")
        week_state = week_state.where(explicit_status.eq(""), explicit_status)
        latest_status_series = latest_status_series.where(week_state.eq(""), week_state)

    absent_mask = latest_status_series.eq("Absent")
    no_show_mask = latest_status_series.eq("No Show")
    dropout_mask = latest_status_series.eq("Drop out")
    return latest_status_series, absent_mask, no_show_mask, dropout_mask


def combine_text_columns(df, columns):
    valid_columns = [column for column in columns if column and column in df.columns]
    if not valid_columns:
        return pd.Series("N/A", index=df.index, dtype=object)

    def pick_text(row):
        values = [str(row[column]).strip() for column in valid_columns if pd.notna(row[column]) and str(row[column]).strip() not in {"", "nan", "N/A"}]
        if not values:
            return "N/A"
        meaningful_values = [value for value in values if not is_status_feedback_text(value)]
        if meaningful_values:
            return meaningful_values[-1]
        return values[-1]

    return df.apply(pick_text, axis=1)


def build_student_weekly_progress(student_row, week_map):
    records = []
    for week_label, columns in week_map.items():
        rating_col = columns.get("rating")
        total_score_col = columns.get("total_score")
        total_marks_col = columns.get("total_marks")
        feedback_col = columns.get("feedback")

        rating_value = pd.to_numeric(student_row.get(rating_col, None), errors="coerce") if rating_col else None
        score_value = pd.to_numeric(student_row.get(total_score_col, None), errors="coerce") if total_score_col else None
        marks_value = pd.to_numeric(student_row.get(total_marks_col, None), errors="coerce") if total_marks_col else None
        feedback_value = str(student_row.get(feedback_col, "")).strip() if feedback_col else ""

        has_rating = pd.notna(rating_value) and float(rating_value) > 0
        has_score = pd.notna(score_value) and float(score_value) > 0
        if not has_rating and not has_score:
            continue

        theory_percent = None
        if has_score:
            denominator = float(marks_value) if pd.notna(marks_value) and float(marks_value) > 0 else 60.0
            theory_percent = round((float(score_value) / denominator) * 100.0, 1)
        elif has_rating:
            theory_percent = round(float(rating_value) * 20.0, 1)

        records.append(
            {
                "Week": week_label,
                "Subjectivity Metric": round(float(rating_value), 2) if has_rating else None,
                "Assessment Score (Out of 100)": theory_percent,
                "Assessment Grade": score_to_grade(theory_percent),
                "Feedback": feedback_value if feedback_value and feedback_value.lower() not in {"nan", "n/a"} else "No feedback available",
            }
        )

    if not records:
        return pd.DataFrame()

    progress_df = pd.DataFrame(records)
    progress_df["WeekOrder"] = progress_df["Week"].str.extract(r"(\d+)").astype(float)
    progress_df = progress_df.sort_values(by=["WeekOrder", "Week"]).drop(columns=["WeekOrder"])
    return progress_df.reset_index(drop=True)


def summarize_student_weekly_progress(progress_df):
    if progress_df.empty or len(progress_df) < 2:
        return "Only one valid subjectivity checkpoint is available, so a week-on-week trend cannot be inferred yet."

    first_row = progress_df.iloc[0]
    last_row = progress_df.iloc[-1]
    first_score = pd.to_numeric(first_row.get("Assessment Score (Out of 100)"), errors="coerce")
    last_score = pd.to_numeric(last_row.get("Assessment Score (Out of 100)"), errors="coerce")
    if pd.isna(first_score) or pd.isna(last_score):
        return "Week records are present, but the available scores are incomplete for a trend summary."

    delta = round(float(last_score) - float(first_score), 1)
    if delta >= 10:
        return f"Week-on-week performance improved by {delta:.1f} points from {first_row['Week']} to {last_row['Week']}."
    if delta <= -10:
        return f"Week-on-week performance dropped by {abs(delta):.1f} points from {first_row['Week']} to {last_row['Week']}."
    return f"Week-on-week performance is broadly stable, moving by {delta:.1f} points from {first_row['Week']} to {last_row['Week']}."


def build_grade_distribution_records(df, week_map, selected_week, practical_scores=None):
    records = []
    week_items = list(week_map.items()) if selected_week == "All Weeks" else [(selected_week, week_map.get(selected_week, {}))]

    for week_label, columns in week_items:
        if not columns:
            continue
        week_scores = calculate_assessment_scores(
            df,
            columns.get("rating"),
            columns.get("total_score"),
            columns.get("total_marks"),
        )
        week_scores = week_scores[week_scores.gt(0)]
        if week_scores.empty:
            continue
        grade_counts = week_scores.apply(score_to_grade).value_counts()
        for grade in GRADE_ORDER:
            records.append(
                {
                    "Scope": week_label,
                    "Grade": grade,
                    "Student Count": int(grade_counts.get(grade, 0)),
                    "Series": "Theory Assessment",
                }
            )

    if practical_scores is not None and not practical_scores.empty:
        valid_practical = practical_scores[practical_scores.gt(0)]
        if not valid_practical.empty:
            grade_counts = valid_practical.apply(score_to_grade).value_counts()
            scope_label = "Practical Code"
            for grade in GRADE_ORDER:
                records.append(
                    {
                        "Scope": scope_label,
                        "Grade": grade,
                        "Student Count": int(grade_counts.get(grade, 0)),
                        "Series": "Practical Code",
                    }
                )

    return pd.DataFrame(records)


def render_section_header(title, subtitle=""):
    st.markdown(
        f"""
        <div class="qa-section">
            <div class="qa-section-title">{title}</div>
            {f'<div class="qa-section-subtitle">{subtitle}</div>' if subtitle else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(value, label, subtext="", accent="#2563eb"):
    st.markdown(
        f"""
        <div class="qa-metric-card" style="border-top-color:{accent};">
            <div class="qa-metric-value" style="color:{accent};">{value}</div>
            <div class="qa-metric-label">{label}</div>
            <div class="qa-metric-subtext">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_banner(message, tone="info"):
    tone_map = {
        "info": ("#dbeafe", "#eff6ff", "#1d4ed8"),
        "warning": ("#fde68a", "#fffbeb", "#d97706"),
        "danger": ("#fecaca", "#fef2f2", "#dc2626"),
    }
    border, background, accent = tone_map.get(tone, tone_map["info"])
    st.markdown(
        f"""
        <div class="qa-banner" style="border-left-color:{accent}; background:{background}; border-color:{border};">
            <div class="qa-banner-text">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_validation_messages(eval_df, week_map, merged_df, selected_student):
    messages = []
    if not eval_df.empty and "Score" in eval_df.columns:
        practical_scores = pd.to_numeric(eval_df["Score"], errors="coerce").fillna(0)
        practical_count = int(practical_scores.gt(0).sum())
        if practical_count > 0:
            messages.append(("success", f"Evaluation results loaded: {practical_count} practical records available."))
    if merged_df.empty:
        messages.append(("warning", "No merged student records are available for the current filter combination."))
    if selected_student and merged_df.empty:
        messages.append(("warning", "Individual performance could not retrieve a student profile for the current selection."))
    return messages


def render_global_warnings(notices, validation_messages):
    seen = set()
    for message in notices:
        if message not in seen:
            if "standardized automatically" in message.lower():
                st.info(message)
            else:
                st.warning(message)
            seen.add(message)
    for tone, message in validation_messages:
        if message in seen:
            continue
        if tone == "success":
            st.success(message)
        elif tone == "warning":
            st.warning(message)
        else:
            st.info(message)
        seen.add(message)


def render_individual_performance_panel(merged_df, eval_df, week_map, rating_col, feedback_col, total_score_col=None, total_marks_col=None):
    render_section_header("Individual Performance", "A 360-degree student profile using subjectivity, practical, persona, and trainer observations.")
    if merged_df.empty:
        st.warning("No student profile can be displayed because the merged dataset is empty for the current filters.")
        return

    student_option_map = {}
    name_counts = merged_df["Candidate Name"].dropna().astype(str).str.strip().value_counts().to_dict()
    for _, row in merged_df.iterrows():
        name = str(row.get("Candidate Name", "")).strip()
        if not name:
            continue
        superset_id = normalize_superset_id(row.get("Superset ID", ""))
        label = f"{name} ({superset_id})" if superset_id and name_counts.get(name, 0) > 1 else name
        student_option_map[label] = row.get("Merge_Key")
    if not student_option_map:
        st.warning("Student records exist, but no valid candidate names were detected after normalization.")
        return

    with st.container(border=True):
        selected_student = st.selectbox("Search or Select Student", sorted(student_option_map.keys()), key="student_360_selected_student")

    selected_key = student_option_map[selected_student]
    student_matches = merged_df[merged_df["Merge_Key"] == selected_key].copy()
    if student_matches.empty:
        st.warning("The selected student profile could not be resolved from the merged records.")
        return

    student_row = student_matches.iloc[0]
    render_global_warnings([], build_validation_messages(eval_df, week_map, merged_df, selected_student))
    skill_snapshot = get_student_skill_snapshot(student_row)

    theory_score = calculate_assessment_scores(student_matches, rating_col, total_score_col, total_marks_col).iloc[0]
    theory_value = "N/A" if pd.isna(theory_score) or float(theory_score) <= 0 else float(theory_score)
    practical_value = student_row.get("Score", "N/A")
    practical_numeric = pd.to_numeric(practical_value, errors="coerce")
    practical_value = "N/A" if pd.isna(practical_numeric) or float(practical_numeric) <= 0 else int(round(float(practical_numeric)))
    persona_value = student_row.get("Student Persona", "Unclassified")

    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    with kpi_col1:
        render_metric_card(
            f"{theory_value:.1f} / 100" if theory_value != "N/A" else "N/A",
            "Assessment Score",
            f"Grade {score_to_grade(theory_value)}" if theory_value != "N/A" else "Current trainer-observation signal unavailable",
            "#2563eb",
        )
    with kpi_col2:
        render_metric_card(
            f"{practical_value} / 100" if practical_value != "N/A" else "N/A",
            "Practical Score",
            f"Grade {score_to_grade(practical_value)}" if practical_value != "N/A" else "Normalized code evaluation score",
            "#7c3aed",
        )
    with kpi_col3:
        render_metric_card(str(persona_value), "Persona", "Current learner profile", "#0f766e")

    snapshot_cols = st.columns(4)
    with snapshot_cols[0]:
        render_metric_card(
            f"{student_row.get('Unified Combined Score (%)', 0):.1f}" if pd.notna(student_row.get("Unified Combined Score (%)", 0)) else "0.0",
            "Unified Score",
            "Subjectivity + practical blend",
            "#1d4ed8",
        )
    with snapshot_cols[1]:
        if skill_snapshot["has_code_signal"]:
            render_metric_card(skill_snapshot["strongest_skill"], "Strongest Code Area", f"Score: {skill_snapshot['strongest_value']:.0f}", "#059669")
        else:
            render_metric_card("N/A", "Strongest Code Area", "No practical data uploaded", "#94a3b8")
    with snapshot_cols[2]:
        if skill_snapshot["has_code_signal"]:
            render_metric_card(skill_snapshot["weakest_skill"], "Weakest Code Area", f"Score: {skill_snapshot['weakest_value']:.0f}", "#dc2626")
        else:
            render_metric_card("N/A", "Weakest Code Area", "No practical data uploaded", "#94a3b8")
    with snapshot_cols[3]:
        profile_mode = "Subjectivity + Code" if pd.to_numeric(student_row.get("Score", 0), errors="coerce") > 0 and pd.to_numeric(student_row.get(rating_col, 0), errors="coerce") > 0 else "Partial Signal"
        render_metric_card(profile_mode, "Profile Mode", "Data used in this student summary", "#7c3aed")

    weekly_progress_df = build_student_weekly_progress(student_row, week_map)

    detail_col1, detail_col2 = st.columns([1.1, 1])
    with detail_col1:
        with st.container(border=True):
            render_section_header("Coding Skills Radar", "Practical competency spread across the code review dimensions.")
            radar_columns = [column for column in EVAL_COLUMNS if column in student_row.index and column not in {"Candidate Name", "Score", "PrimaryGap"}]
            radar_values = [pd.to_numeric(student_row.get(column, 0), errors="coerce") for column in radar_columns]
            radar_values = [0 if pd.isna(value) else value for value in radar_values]
            if any(value > 0 for value in radar_values):
                fig_radar = go.Figure(data=go.Scatterpolar(r=radar_values, theta=radar_columns, fill="toself", line_color="#2563eb"))
                fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=350)
                st.plotly_chart(fig_radar, use_container_width=True)
            else:
                st.info("Practical code metrics are not available yet for the selected student.")

    with detail_col2:
        with st.container(border=True):
            render_section_header("360 Degree Summary", "Rule-based guidance built from subjectivity, practical, persona, feedback, and code review signals.")
            st.markdown(generate_student_summary(student_row, rating_col, feedback_col))
            st.markdown("---")
            trainer_feedback = student_row.get(feedback_col, "No feedback available") if feedback_col else "No feedback available"
            trainer_feedback = "No feedback available" if pd.isna(trainer_feedback) else trainer_feedback
            st.markdown(f"**Trainer Feedback:** {trainer_feedback}")
            if skill_snapshot["has_code_signal"]:
                st.markdown(f"**Code Evaluation Signals:** Strongest area is `{skill_snapshot['strongest_skill']}` while the current coaching focus should be `{skill_snapshot['weakest_skill']}`.")
            else:
                st.markdown("**Code Evaluation Signals:** Practical code metrics are not available for this student yet.")
            experience_profile = student_row.get("ExperienceProfile", "Not available")
            files_analyzed = student_row.get("FilesAnalyzed", "Not available")
            st.markdown(f"**Task-1 Engine Context:** Experience profile: {experience_profile}. Files analyzed: {files_analyzed}.")

    with st.container(border=True):
        render_section_header("Week-on-Week Assessment Trend", "Tracks the selected student's subjectivity progression across uploaded assessment weeks.")
        if weekly_progress_df.empty:
            st.info("No week-on-week assessment history is available yet for the selected student.")
        else:
            trend_col1, trend_col2 = st.columns([1.15, 0.85])
            with trend_col1:
                fig_student_trend = px.line(
                    weekly_progress_df,
                    x="Week",
                    y="Assessment Score (Out of 100)",
                    markers=True,
                    text="Assessment Grade",
                    color_discrete_sequence=["#2563eb"],
                )
                fig_student_trend.update_traces(line=dict(width=3), textposition="top center")
                fig_student_trend.update_layout(
                    margin=dict(t=8, b=0, l=0, r=0),
                    height=320,
                    yaxis_title="Assessment Score /100",
                    xaxis_title="Assessment Week",
                )
                st.plotly_chart(fig_student_trend, use_container_width=True)
            with trend_col2:
                st.dataframe(weekly_progress_df, use_container_width=True, hide_index=True)
                st.markdown("---")
                st.markdown(summarize_student_weekly_progress(weekly_progress_df))


def run():
    if "tracker_data" not in st.session_state:
        st.session_state["tracker_data"] = None
    if "eval_results" not in st.session_state:
        st.session_state["eval_results"] = []
    if "assessment_notices" not in st.session_state:
        st.session_state["assessment_notices"] = []
    if "table_filter" not in st.session_state:
        st.session_state["table_filter"] = "All"
    if "current_file_selection" not in st.session_state:
        st.session_state["current_file_selection"] = None
    if "current_batch_selection" not in st.session_state:
        st.session_state["current_batch_selection"] = None
    if "current_week_selection" not in st.session_state:
        st.session_state["current_week_selection"] = None
    if "nlp_candidate_view" not in st.session_state:
        st.session_state["nlp_candidate_view"] = False
    if "show_append_uploader" not in st.session_state:
        st.session_state["show_append_uploader"] = False

    st.markdown(
        """
        <style>
        .qa-shell { background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 100%); padding: 24px 30px; border-radius: 18px; margin-bottom: 24px; box-shadow: 0 18px 36px rgba(15, 23, 42, 0.14); }
        .qa-title { color: #ffffff; font-size: 28px; font-weight: 800; margin-bottom: 6px; }
        .qa-subtitle { color: rgba(255, 255, 255, 0.88); font-size: 14px; max-width: 850px; }
        .qa-section { background: #ffffff; border: 1px solid #dbe7f5; border-radius: 16px; padding: 14px 18px; margin-bottom: 14px; box-shadow: 0 10px 22px rgba(15, 23, 42, 0.05); }
        .qa-section-title { color: #0f172a; font-size: 22px; font-weight: 800; margin-bottom: 4px; }
        .qa-section-subtitle { color: #64748b; font-size: 15px; }
        .qa-metric-card { background: #ffffff; border: 1px solid #e2e8f0; border-top: 4px solid #2563eb; border-radius: 14px; min-height: 112px; padding: 16px 12px; box-shadow: 0 8px 18px rgba(15, 23, 42, 0.06); display: flex; flex-direction: column; justify-content: center; text-align: center; }
        .qa-metric-value { font-size: 26px; font-weight: 800; line-height: 1.1; }
        .qa-metric-label { margin-top: 6px; color: #475569; font-size: 13px; font-weight: 800; letter-spacing: 0.06em; text-transform: uppercase; }
        .qa-metric-subtext { margin-top: 4px; color: #94a3b8; font-size: 12px; }
        .qa-banner { border: 1px solid #dbeafe; border-left: 5px solid #1d4ed8; border-radius: 14px; padding: 14px 16px; margin-bottom: 14px; }
        .qa-banner-text { color: #334155; font-size: 14px; font-weight: 600; }
        .qa-topic-box { background: #f8fbff; border: 1px solid #dbe7f5; border-left: 5px solid #2563eb; border-radius: 12px; padding: 14px 16px; margin-bottom: 16px; color: #334155; }
        </style>
        <div class="qa-shell">
            <div class="qa-title">Assessment & Feedback Analytics</div>
            <div class="qa-subtitle">Subjectivity-first analytics workspace for trainer ratings, week-wise assessment trends, feedback insights, intervention planning, and Student 360 review. Top Brains performance remains on its dedicated page.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state["tracker_data"] is not None:
        action_col1, action_col2, action_col3 = st.columns(3)
        with action_col1:
            add_label = "Close Add Files" if st.session_state["show_append_uploader"] else "Add More Files"
            if st.button(add_label):
                st.session_state["show_append_uploader"] = not st.session_state["show_append_uploader"]
                st.rerun()
        with action_col2:
            if st.button("Refresh Page"):
                st.rerun()
        with action_col3:
            if st.button("Reset Environment"):
                st.session_state["tracker_data"] = None
                st.session_state["eval_results"] = []
                st.session_state["assessment_notices"] = []
                st.session_state["table_filter"] = "All"
                st.session_state["nlp_candidate_view"] = False
                st.session_state["show_append_uploader"] = False
                st.rerun()

        if st.session_state["show_append_uploader"]:
            with st.container(border=True):
                render_section_header("Add Assessment Files", "Append more subjectivity trackers, companion Top Brains files, or practical result files to the shared assessment session without clearing existing data.")
                additional_files = st.file_uploader(
                    "Upload Additional Assessment Ecosystem Files",
                    type=["xlsx", "xlsm", "csv"],
                    accept_multiple_files=True,
                    key="task2_append_uploader",
                )
                if additional_files and st.button("Add Files to Current Session", type="primary"):
                    with st.spinner("Adding files to the current assessment session..."):
                        tracker_data, eval_data, notices = process_files(additional_files)
                        existing_tracker = st.session_state.get("tracker_data")
                        existing_eval = pd.DataFrame(st.session_state.get("eval_results", []))
                        combined_tracker = pd.concat([existing_tracker, tracker_data], ignore_index=True) if existing_tracker is not None and not existing_tracker.empty else tracker_data
                        combined_eval = pd.concat([existing_eval, eval_data], ignore_index=True) if not existing_eval.empty else eval_data
                        st.session_state["tracker_data"] = combined_tracker
                        st.session_state["eval_results"] = combined_eval.to_dict("records")
                        existing_notices = st.session_state.get("assessment_notices", [])
                        st.session_state["assessment_notices"] = list(dict.fromkeys(existing_notices + notices))
                        st.session_state["show_append_uploader"] = False
                        time.sleep(0.5)
                        st.rerun()

    if st.session_state["tracker_data"] is None:
        render_section_header("Secure Assessment Ingestion", "Upload subjectivity trackers, Top Brains workbooks, and practical evaluation result files into the shared assessment session.")
        uploaded_files = st.file_uploader("Upload Assessment Ecosystem Files", type=["xlsx", "xlsm", "csv"], accept_multiple_files=True)
        if uploaded_files and st.button("Process Assessment Analytics", type="primary"):
            with st.spinner("Processing files and validating uploaded data..."):
                tracker_data, eval_data, notices = process_files(uploaded_files)
                st.session_state["tracker_data"] = tracker_data
                st.session_state["eval_results"] = eval_data.to_dict("records")
                st.session_state["assessment_notices"] = notices
                st.session_state["show_append_uploader"] = False
                time.sleep(0.8)
                st.rerun()
        return

    raw_tracker_df = st.session_state["tracker_data"].copy()
    if not raw_tracker_df.empty and "Superset ID" in raw_tracker_df.columns:
        raw_tracker_df["Superset ID"] = raw_tracker_df["Superset ID"].apply(normalize_superset_id)
    subjectivity_tracker_df, top_brains_tracker_df = split_tracker_sources(raw_tracker_df)
    eval_df, shared_eval_notices = get_shared_eval_dataframe()
    assessment_eval_df = filter_eval_dataframe_by_track(eval_df, GITHUB_ASSESSMENT_TRACK) if not eval_df.empty else pd.DataFrame(columns=EVAL_COLUMNS)
    if shared_eval_notices:
        st.session_state["assessment_notices"] = list(dict.fromkeys(st.session_state.get("assessment_notices", []) + shared_eval_notices))

    if subjectivity_tracker_df.empty:
        loaded_top_brains_count = int(len(top_brains_tracker_df)) if not top_brains_tracker_df.empty else 0
        if loaded_top_brains_count > 0:
            st.info("The current shared assessment session contains Top Brains records only. Use the dedicated Top Brains page for that view, or upload subjectivity trackers to continue here.")
        else:
            st.warning("No subjectivity tracker records are currently available in the shared assessment session.")
        return

    raw_tracker_df = subjectivity_tracker_df.copy()

    available_files = sorted(raw_tracker_df["Source_File"].dropna().unique().tolist()) if "Source_File" in raw_tracker_df.columns else []

    selected_file = st.session_state.get("current_file_selection")
    if selected_file not in ["All Combined Assessments"] + available_files:
        st.session_state["current_file_selection"] = "All Combined Assessments"
    selected_file = st.session_state.get("current_file_selection")

    practical_tracker_df = build_practical_tracker_from_eval_df(assessment_eval_df) if not assessment_eval_df.empty else pd.DataFrame()
    practical_tracker_frames = [practical_tracker_df] if not practical_tracker_df.empty else []
    if selected_file == "All Combined Assessments":
        tracker_df = build_combined_tracker_dataset(raw_tracker_df, practical_tracker_frames=practical_tracker_frames)
    else:
        tracker_df = raw_tracker_df[raw_tracker_df["Source_File"] == selected_file].copy() if not raw_tracker_df.empty else pd.DataFrame()

    global_week_map = build_week_map(tracker_df) if not tracker_df.empty else {}

    batch_scope_df = tracker_df.copy()
    available_batches = (
        sorted(batch_scope_df["Assigned Batch"].dropna().unique().tolist())
        if "Assigned Batch" in batch_scope_df.columns
        else []
    )

    selected_batch = st.session_state.get("current_batch_selection")
    if selected_batch not in ["All Batches"] + available_batches:
        st.session_state["current_batch_selection"] = "All Batches"
    selected_batch = st.session_state.get("current_batch_selection")

    scoped_df = tracker_df.copy()
    if not scoped_df.empty and selected_batch != "All Batches":
        scoped_df = scoped_df[scoped_df["Assigned Batch"] == selected_batch]
    scoped_week_map = build_week_map(scoped_df) if not scoped_df.empty else {}
    if not scoped_week_map:
        scoped_week_map = global_week_map
    scoped_week_options = ["All Weeks"] + list(scoped_week_map.keys()) if scoped_week_map else ["All Weeks"]
    selected_week = st.session_state.get("current_week_selection")
    if selected_week not in scoped_week_options:
        st.session_state["current_week_selection"] = scoped_week_options[0]
    selected_week = st.session_state.get("current_week_selection")

    with st.container(border=True):
        render_section_header("Analytics Controls", "Filter by source file, batch, and week without changing the underlying ingestion or merge logic.")
        control_col1, control_col2, control_col3 = st.columns(3)
        with control_col1:
            st.selectbox(
                "Select Assessment File",
                ["All Combined Assessments"] + available_files,
                key="current_file_selection",
            )
        with control_col2:
            st.selectbox(
                "Select Target Batch",
                ["All Batches"] + available_batches,
                key="current_batch_selection",
            )
        with control_col3:
            st.selectbox(
                "Select Week for Analysis",
                scoped_week_options,
                key="current_week_selection",
            )

    if (
        selected_file != st.session_state["current_file_selection"]
        or selected_batch != st.session_state["current_batch_selection"]
        or selected_week != st.session_state["current_week_selection"]
    ):
        st.session_state["table_filter"] = "All"
        st.session_state["nlp_candidate_view"] = False

    selected_file = st.session_state.get("current_file_selection")
    selected_batch = st.session_state.get("current_batch_selection")
    selected_week = st.session_state.get("current_week_selection")

    plot_df = tracker_df.copy()
    if not plot_df.empty:
        if selected_batch != "All Batches":
            plot_df = plot_df[plot_df["Assigned Batch"] == selected_batch]
        plot_df = aggregate_tracker_scope(plot_df)

    week_map = build_week_map(plot_df) if not plot_df.empty else {}
    if not week_map:
        week_map = global_week_map
    selected_week_group = week_map.get(selected_week, {})
    rating_col = selected_week_group.get("rating")
    feedback_col = selected_week_group.get("feedback")
    topic_col = selected_week_group.get("topic")
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
            active_df["All Weeks - Rating"] = average_numeric_columns(active_df, rating_columns)
            active_df["All Weeks - Feedback"] = combine_text_columns(active_df, feedback_columns)
        rating_col = "All Weeks - Rating"
        feedback_col = "All Weeks - Feedback"
        if total_score_columns:
            active_df["All Weeks - TotalScore"] = average_numeric_columns(active_df, total_score_columns)
            total_score_col = "All Weeks - TotalScore"
        else:
            total_score_col = None
        if total_marks_columns:
            marks_series = average_numeric_columns(active_df, total_marks_columns).replace(0, 60).fillna(60)
            active_df["All Weeks - TotalMarks"] = marks_series
            total_marks_col = "All Weeks - TotalMarks"
        else:
            total_marks_col = None
        if practical_score_columns:
            active_df["All Weeks - PracticalScore"] = average_numeric_columns(active_df, practical_score_columns)
            practical_score_col = "All Weeks - PracticalScore"
        else:
            practical_score_col = None
        topic_col = None
    else:
        candidate_presence_cols = [column for column in [rating_col, practical_score_col] if column and column in plot_df.columns]
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
            active_df["Superset ID"] = active_df["Superset ID"].apply(normalize_superset_id)
        active_df["Merge_Key"] = active_df.apply(build_merge_key, axis=1)
        active_df, tracker_dedupe_notices = deduplicate_candidate_records(
            active_df,
            metric_column=rating_col,
            preferred_columns=[col for col in [rating_col, feedback_col, topic_col] if col],
            context_label="tracker records",
        )
        st.session_state["assessment_notices"] = list(
            dict.fromkeys(st.session_state.get("assessment_notices", []) + tracker_dedupe_notices)
        )
    if not assessment_eval_df.empty:
        if "Superset ID" in assessment_eval_df.columns:
            assessment_eval_df["Superset ID"] = assessment_eval_df["Superset ID"].apply(normalize_superset_id)
        assessment_eval_df["Candidate Name"] = assessment_eval_df["Candidate Name"].astype(str).str.strip()
        assessment_eval_df["Merge_Key"] = assessment_eval_df.apply(build_merge_key, axis=1)

    if not active_df.empty and not assessment_eval_df.empty:
        eval_merge, _ = deduplicate_candidate_records(
            assessment_eval_df.rename(columns={"Candidate Name": "Code Candidate Name"}).copy(),
            metric_column="Score",
            preferred_columns=EVAL_COLUMNS,
            context_label="merged evaluation results",
            partition_columns=["Merge_Key"],
        )
        merged_df = pd.merge(active_df, eval_merge, on="Merge_Key", how="outer")
        merged_df["Candidate Name"] = merged_df.get("Candidate Name", pd.Series(dtype=str)).fillna(merged_df.get("Code Candidate Name", pd.Series(dtype=str)))
        if "Superset ID_x" in merged_df.columns or "Superset ID_y" in merged_df.columns:
            merged_df["Superset ID"] = merged_df.get("Superset ID_x", pd.Series(index=merged_df.index, dtype=object)).fillna(
                merged_df.get("Superset ID_y", pd.Series(index=merged_df.index, dtype=object))
            )
        merged_df["Assigned Batch"] = merged_df.get("Assigned Batch", pd.Series(dtype=str)).fillna("Code Eval Batch")
    elif not active_df.empty:
        merged_df = active_df.copy()
    elif not assessment_eval_df.empty:
        merged_df, _ = deduplicate_candidate_records(
            assessment_eval_df.copy(),
            metric_column="Score",
            preferred_columns=EVAL_COLUMNS,
            context_label="standalone evaluation results",
            partition_columns=["Merge_Key"],
        )
        merged_df["Assigned Batch"] = "Code Eval Batch"
    else:
        merged_df = pd.DataFrame()

    if not merged_df.empty:
        merged_df["Candidate Name"] = merged_df["Candidate Name"].fillna("").astype(str).str.strip()
        merged_df = merged_df[merged_df["Candidate Name"].ne("")].copy()
        if "Merge_Key" not in merged_df.columns:
            merged_df["Merge_Key"] = merged_df.apply(build_merge_key, axis=1)
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
        merged_df, merged_notices = deduplicate_candidate_records(
            merged_df,
            metric_column="Score",
            preferred_columns=[col for col in ["Unified Combined Score (%)", "Score", rating_col, "PrimaryGap"] if col],
            context_label="merged records",
            partition_columns=["Merge_Key"],
        )
        st.session_state["assessment_notices"] = list(
            dict.fromkeys(st.session_state.get("assessment_notices", []) + merged_notices)
        )
        merged_df["Unified Combined Score (%)"] = merged_df.apply(lambda row: calc_unified(row, rating_col), axis=1)
        rating_series = (
            pd.to_numeric(merged_df[rating_col], errors="coerce").fillna(0)
            if rating_col and rating_col in merged_df.columns
            else pd.Series(0, index=merged_df.index, dtype=float)
        )
        merged_df["Numeric_Rating"] = rating_series
        merged_df["Student Persona"] = merged_df.apply(lambda row: generate_persona(row, rating_col), axis=1)

    if not active_df.empty and (
        (rating_col and rating_col in active_df.columns) or (total_score_col and total_score_col in active_df.columns)
    ):
        assessment_scores = calculate_assessment_scores(active_df, rating_col, total_score_col, total_marks_col)
        status_columns = [rating_col]
        if selected_week == "All Weeks":
            status_columns.extend(feedback_columns)
            status_columns.extend(rating_columns)
        else:
            status_columns.append(feedback_col)
        _, absent_mask, dropout_mask = build_status_masks(active_df, assessment_scores, status_columns)
        valid_assessment_scores = assessment_scores[(assessment_scores > 0) & ~absent_mask & ~dropout_mask]
        avg_theory = valid_assessment_scores.mean() if len(valid_assessment_scores) > 0 else 0
    else:
        avg_theory = 0
        valid_assessment_scores = pd.Series(dtype=float)

    practical_scores = (
        pd.to_numeric(merged_df["Score"], errors="coerce").fillna(0)
        if not merged_df.empty and "Score" in merged_df.columns
        else (pd.to_numeric(assessment_eval_df["Score"], errors="coerce").fillna(0) if not assessment_eval_df.empty else pd.Series(dtype=float))
    )
    has_practical_data = bool(not practical_scores.empty and practical_scores.gt(0).any())
    avg_practical = practical_scores[practical_scores.gt(0)].mean() if has_practical_data else None
    practical_student_count = int(practical_scores.gt(0).sum()) if not practical_scores.empty else 0
    validation_messages = build_validation_messages(assessment_eval_df, week_map, merged_df, None)
    render_global_warnings(st.session_state.get("assessment_notices", []), validation_messages)

    if not merged_df.empty:
        merged_assessment_scores = calculate_assessment_scores(merged_df, rating_col, total_score_col, total_marks_col)
        merged_status_columns = [rating_col]
        if selected_week == "All Weeks":
            merged_status_columns.extend(feedback_columns)
            merged_status_columns.extend(rating_columns)
        else:
            merged_status_columns.append(feedback_col)
        merged_evaluated_mask, merged_absent_mask, merged_dropout_mask = build_status_masks(
            merged_df, merged_assessment_scores, merged_status_columns
        )
        performance_signal = pd.to_numeric(merged_df.get("Unified Combined Score (%)", merged_assessment_scores), errors="coerce")
        performance_signal = performance_signal.fillna(merged_assessment_scores).fillna(pd.to_numeric(merged_df.get("Score", 0), errors="coerce")).fillna(0)
        merged_top_mask = performance_signal.gt(90) & ~merged_absent_mask & ~merged_dropout_mask
        merged_attention_mask = performance_signal.gt(0) & performance_signal.lt(60) & ~merged_absent_mask & ~merged_dropout_mask
        evaluated_count = int(merged_evaluated_mask.sum())
        total_student_count = int(len(merged_df))
        top_count = int(merged_top_mask.sum())
        attention_count = int(merged_attention_mask.sum())
    else:
        merged_top_mask = pd.Series(dtype=bool)
        merged_attention_mask = pd.Series(dtype=bool)
        merged_absent_mask = pd.Series(dtype=bool)
        merged_dropout_mask = pd.Series(dtype=bool)
        performance_signal = pd.Series(dtype=float)
        evaluated_count = 0
        total_student_count = 0
        top_count = 0
        attention_count = 0

    exception_week_map = week_map if selected_week == "All Weeks" else ({selected_week: selected_week_group} if selected_week_group else {})
    if not merged_df.empty:
        latest_exception_status, latest_absent_mask, latest_no_show_mask, latest_dropout_mask = build_explicit_exception_masks(
            merged_df, exception_week_map
        )
        if latest_exception_status.eq("").all():
            latest_exception_status = pd.Series("", index=merged_df.index, dtype=object)
            latest_absent_mask = merged_absent_mask.copy()
            latest_no_show_mask = pd.Series(False, index=merged_df.index, dtype=bool)
            latest_dropout_mask = merged_dropout_mask.copy()
    else:
        latest_exception_status = pd.Series(dtype=object)
        latest_absent_mask = pd.Series(dtype=bool)
        latest_no_show_mask = pd.Series(dtype=bool)
        latest_dropout_mask = pd.Series(dtype=bool)

    absent_count = int(latest_absent_mask.sum()) if not latest_absent_mask.empty else 0
    no_show_count = int(latest_no_show_mask.sum()) if not latest_no_show_mask.empty else 0
    dropout_count = int(latest_dropout_mask.sum()) if not latest_dropout_mask.empty else 0

    main_tabs = st.tabs(["Student Progression", "NLP Feedback Insights", "Executive View", "Student 360 Degree"])

    with main_tabs[0]:
        render_section_header("Student Progression", "Monitor subjectivity movement, practical momentum, and intervention needs for the selected cohort.")
        if topic_col and topic_col in plot_df.columns:
            topic_values = plot_df[topic_col].dropna().astype(str).unique().tolist()
            if topic_values:
                st.markdown(f"<div class='qa-topic-box'><strong>{selected_week} Topic Focus:</strong><br>{topic_values[0]}</div>", unsafe_allow_html=True)

        metric_columns = st.columns(7)
        with metric_columns[0]:
            render_metric_card(
                f"{avg_theory:.1f} / 100" if evaluated_count > 0 else "N/A",
                "Assessment Average",
                (
                    f"{evaluated_count} assessment-evaluated of {total_student_count} total | Grade {score_to_grade(avg_theory)}"
                    if evaluated_count > 0
                    else (f"{total_student_count} total students | No valid theory records" if total_student_count > 0 else "No valid theory records")
                ),
                "#2563eb",
            )
            if st.button("View All", key="btn_all"):
                st.session_state["table_filter"] = "All"
                st.rerun()
        with metric_columns[1]:
            if avg_practical is None or pd.isna(avg_practical):
                render_metric_card("N/A", "Code Average", f"{practical_student_count} practical-scored of {total_student_count} total", "#7c3aed")
            else:
                render_metric_card(
                    f"{avg_practical:.0f}",
                    "Code Average",
                    f"{practical_student_count} practical-scored of {total_student_count} total | Grade {score_to_grade(avg_practical)}",
                    "#7c3aed",
                )
                if st.button("View All Code", key="btn_code"):
                    st.session_state["table_filter"] = "Code"
                    st.rerun()
        with metric_columns[2]:
            render_metric_card(str(top_count), "Top Performers", "Unified score > 90", "#059669")
            if st.button("View Top", key="btn_top"):
                st.session_state["table_filter"] = "Top"
                st.rerun()
        with metric_columns[3]:
            render_metric_card(str(attention_count), "Scope for Improvement", "Unified score < 60", "#dc2626")
            if st.button("View Scope for Improvement", key="btn_attention"):
                st.session_state["table_filter"] = "Attention"
                st.rerun()
        with metric_columns[4]:
            render_metric_card(str(absent_count), "Absent", "Latest explicit absent status", "#d97706")
            if st.button("View Absent", key="btn_absent"):
                st.session_state["table_filter"] = "Absent"
                st.rerun()
        with metric_columns[5]:
            render_metric_card(str(no_show_count), "No Show", "Latest explicit no-show status", "#f59e0b")
            if st.button("View No Show", key="btn_noshow"):
                st.session_state["table_filter"] = "NoShow"
                st.rerun()
        with metric_columns[6]:
            render_metric_card(str(dropout_count), "Dropouts", "Exited or inactive", "#64748b")
            if st.button("View Dropouts", key="btn_dropout"):
                st.session_state["table_filter"] = "Dropout"
                st.rerun()

        if has_practical_data:
            with st.expander("Practical Code Evaluation Breakdown"):
                if not assessment_eval_df.empty:
                    st.dataframe(assessment_eval_df[EVAL_COLUMNS], use_container_width=True, hide_index=True)
                elif not merged_df.empty:
                    practical_breakdown = merged_df[["Candidate Name", "Assigned Batch", "Score"]].copy()
                    practical_breakdown["Practical Code (Out of 100)"] = pd.to_numeric(practical_breakdown["Score"], errors="coerce").round(1)
                    if topic_col and topic_col in merged_df.columns:
                        practical_breakdown["Topic"] = merged_df[topic_col]
                    practical_breakdown = practical_breakdown.drop(columns=["Score"])
                    st.dataframe(practical_breakdown.fillna("N/A"), use_container_width=True, hide_index=True)

        grade_records_df = build_grade_distribution_records(
            plot_df,
            week_map,
            selected_week,
            practical_scores=practical_scores if has_practical_data else None,
        )
        if not grade_records_df.empty:
            color_map = {"A+": "#1d4ed8", "A": "#10b981", "B": "#3b82f6", "C": "#f59e0b", "F": "#ef4444"}
            if selected_week == "All Weeks":
                fig_progression = px.bar(
                    grade_records_df,
                    x="Scope",
                    y="Student Count",
                    color="Grade",
                    barmode="group",
                    category_orders={"Grade": GRADE_ORDER},
                    color_discrete_map=color_map,
                    height=350,
                )
                fig_progression.update_layout(xaxis_title="Assessment Scope", legend_title_text="Grade")
            else:
                scoped_grade_df = grade_records_df[grade_records_df["Scope"] == selected_week].copy()
                if scoped_grade_df.empty:
                    scoped_grade_df = grade_records_df.copy()
                fig_progression = px.bar(
                    scoped_grade_df,
                    x="Grade",
                    y="Student Count",
                    color="Grade",
                    category_orders={"Grade": GRADE_ORDER},
                    color_discrete_map=color_map,
                    height=350,
                )
                fig_progression.update_layout(xaxis_title="Grade Bucket", legend_title_text="")
                fig_progression.update_traces(showlegend=False)
            fig_progression.update_layout(margin=dict(t=20, b=0, l=0, r=0), yaxis_title="Number of Students", plot_bgcolor="white", paper_bgcolor="white")
            with st.container(border=True):
                render_section_header("Unified Progression Chart", "Grade-bucket view of assessment scores normalized to 100 for the current selection.")
                st.plotly_chart(fig_progression, use_container_width=True)

        filter_label = "All Students"
        if st.session_state["table_filter"] == "Top":
            filter_label = "Top Performers"
        elif st.session_state["table_filter"] == "Code":
            filter_label = "Students With Practical Code Scores"
        elif st.session_state["table_filter"] == "Attention":
            filter_label = "Scope for Improvement"
        elif st.session_state["table_filter"] == "Absent":
            filter_label = "Absent Students"
        elif st.session_state["table_filter"] == "NoShow":
            filter_label = "No Show Students"
        elif st.session_state["table_filter"] == "Dropout":
            filter_label = "Dropped Out Students"

        render_section_header("Detailed Feedback Grid", f"Showing: {filter_label}")
        if not merged_df.empty:
            display_df = merged_df[["Candidate Name"]].copy()
            if "Superset ID" in merged_df.columns:
                display_df["Superset ID"] = merged_df["Superset ID"]
            assessment_scores = calculate_assessment_scores(merged_df, rating_col, total_score_col, total_marks_col)
            display_df["Assessment Score (Out of 100)"] = assessment_scores.where(assessment_scores.gt(0), other=pd.NA)
            display_df["Assessment Grade"] = assessment_scores.apply(score_to_grade).replace("N/A", pd.NA)
            if "Score" in merged_df.columns:
                practical_display = pd.to_numeric(merged_df["Score"], errors="coerce").round(0)
                display_df["Practical Code (Out of 100)"] = practical_display.where(practical_display.gt(0), other=pd.NA)
                display_df["Practical Grade"] = practical_display.apply(score_to_grade).replace("N/A", pd.NA)
            else:
                display_df["Practical Code (Out of 100)"] = "N/A"
            display_df["Unified Combined Score (%)"] = merged_df["Unified Combined Score (%)"].round(1)
            if not latest_exception_status.empty:
                display_df["Latest Status"] = latest_exception_status.replace({"": pd.NA, "Active": pd.NA})
            display_df["Trainer Feedback"] = merged_df[feedback_col] if feedback_col and feedback_col in merged_df.columns else "N/A"
            if st.session_state["table_filter"] == "Top":
                display_df = display_df.loc[merged_top_mask]
            elif st.session_state["table_filter"] == "Code":
                code_mask = pd.to_numeric(merged_df.get("Score", 0), errors="coerce").fillna(0) > 0
                display_df = display_df.loc[code_mask]
            elif st.session_state["table_filter"] == "Attention":
                display_df = display_df.loc[merged_attention_mask]
            elif st.session_state["table_filter"] == "Absent":
                display_df = display_df.loc[latest_absent_mask]
            elif st.session_state["table_filter"] == "NoShow":
                display_df = display_df.loc[latest_no_show_mask]
            elif st.session_state["table_filter"] == "Dropout":
                display_df = display_df.loc[latest_dropout_mask]
            with st.container(border=True):
                st.dataframe(display_df.fillna("N/A"), use_container_width=True, hide_index=True)
        else:
            st.info("No records found for the current dashboard filters.")

    with main_tabs[1]:
        render_section_header("NLP Feedback Insights", "Summarized trainer feedback patterns and classroom sentiment signals for the selected cohort.")
        if feedback_col and feedback_col in active_df.columns:
            automated_mask = active_df[feedback_col].astype(str).str.contains("Automated Theory", case=False, na=False)
            status_only_mask = active_df[feedback_col].apply(is_status_feedback_text)
            nlp_df = active_df[~automated_mask & ~status_only_mask].copy()
            if nlp_df.empty:
                st.info("The selected data currently contains status-based or automated feedback only.")
            else:
                if not has_practical_data:
                    st.info("Practical code data is not available in the current view. NLP insights below are derived only from trainer observations and feedback text.")
                feedback_text = nlp_df[feedback_col].dropna().astype(str).str.lower()
                theme_counts = {
                    "Strong Concept Grasp": feedback_text.str.contains(r"strong|excellent|clear|outstanding|grasp").sum(),
                    "Needs Applied Practice": feedback_text.str.contains(r"practice|practical|hands-on|execution|implement|code").sum(),
                    "Low Classroom Focus": feedback_text.str.contains(r"participation|focus|attention|distract").sum(),
                    "Logic & Syntax Issues": feedback_text.str.contains(r"logic|syntax|error|structure|bug").sum(),
                    "Steady & Consistent": feedback_text.str.contains(r"consistent|steady|reliable|maintain|average").sum(),
                }
                positive = theme_counts["Strong Concept Grasp"] + theme_counts["Steady & Consistent"]
                constructive = theme_counts["Needs Applied Practice"] + theme_counts["Logic & Syntax Issues"]
                behavioral = theme_counts["Low Classroom Focus"]
                total = positive + constructive + behavioral or 1
                topic_name = None
                if topic_col and topic_col in active_df.columns:
                    topic_values = active_df[topic_col].dropna().astype(str).unique().tolist()
                    topic_name = topic_values[0] if topic_values else None
                directive_df = build_curriculum_directives(theme_counts, selected_week, topic_name, has_practical_data)

                chart_col1, chart_col2 = st.columns(2)
                with chart_col1:
                    with st.container(border=True):
                        describe_graph("Classroom Sentiment Mix", "Balanced view of positive, constructive, and participation-linked feedback signals for the selected week. Slice labels show the share of feedback signals in each sentiment bucket.")
                        sentiment_df = pd.DataFrame({"Sentiment": ["Positive", "Constructive", "Behavioral"], "Value": [(positive / total) * 100, (constructive / total) * 100, (behavioral / total) * 100]})
                        fig_donut = px.pie(sentiment_df, names="Sentiment", values="Value", hole=0.58, color="Sentiment", color_discrete_map={"Positive": "#10b981", "Constructive": "#f59e0b", "Behavioral": "#ef4444"})
                        fig_donut.update_traces(texttemplate="%{percent:.1%}", hovertemplate="<b>%{label}</b><br>Share of signals: %{value:.1f}%<extra></extra>")
                        fig_donut.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=350, legend=dict(orientation="h", y=-0.15))
                        st.plotly_chart(fig_donut, use_container_width=True)
                with chart_col2:
                    with st.container(border=True):
                        describe_graph("Feedback Theme Distribution", "Theme counts highlight which learning signals dominate the current week and where curriculum attention should go.")
                        theme_df = pd.DataFrame(list(theme_counts.items()), columns=["Theme", "Student Count"]).sort_values(by="Student Count", ascending=True)
                        fig_theme = px.bar(theme_df, x="Student Count", y="Theme", orientation="h", color="Student Count", color_continuous_scale="Blues")
                        fig_theme.update_layout(
                            height=350,
                            margin=dict(t=0, b=0, l=0, r=0),
                            coloraxis_showscale=False,
                            yaxis=dict(autorange="reversed"),
                        )
                        st.plotly_chart(fig_theme, use_container_width=True)

                render_section_header("Strategic Curriculum Directives", f"Weekly coaching actions for {selected_week}{f' on {topic_name}' if topic_name else ''}.")
                directive_col1, directive_col2 = st.columns([1, 1.2])
                with directive_col1:
                    with st.container(border=True):
                        describe_graph("Directive Priority", "High-priority actions are driven by the most frequent feedback themes in the current week.")
                        priority_df = directive_df.copy()
                        priority_df["Priority Score"] = priority_df["Priority"].map({"High": 3, "Medium": 2, "Low": 1}).fillna(1)
                        fig_priority = px.bar(
                            priority_df,
                            x="Priority Score",
                            y="Action",
                            orientation="h",
                            color="Priority",
                            color_discrete_map={"High": "#dc2626", "Medium": "#f59e0b", "Low": "#2563eb"},
                            height=350,
                        )
                        fig_priority.update_layout(margin=dict(t=0, b=0, l=0, r=0), xaxis_title="Priority Weight", yaxis_title="")
                        st.plotly_chart(fig_priority, use_container_width=True)
                with directive_col2:
                    with st.container(border=True):
                        st.dataframe(directive_df, use_container_width=True, hide_index=True)
                        if st.button("View Candidates For Current NLP Week", key="btn_nlp_candidates"):
                            st.session_state["nlp_candidate_view"] = True
                if st.session_state.get("nlp_candidate_view"):
                    render_section_header("Current NLP Week Candidate List", "Candidates below are the learners whose trainer feedback is contributing to the current NLP insight view.")
                    nlp_candidates = merged_df.loc[merged_df["Merge_Key"].isin(nlp_df["Merge_Key"]), ["Candidate Name", "Assigned Batch"]].copy()
                    assessment_scores = calculate_assessment_scores(merged_df, rating_col, total_score_col, total_marks_col)
                    nlp_candidates["Assessment Score (Out of 100)"] = assessment_scores.loc[nlp_candidates.index].where(assessment_scores.loc[nlp_candidates.index].gt(0), other=pd.NA)
                    nlp_candidates["Assessment Grade"] = assessment_scores.loc[nlp_candidates.index].apply(score_to_grade).replace("N/A", pd.NA)
                    if "Score" in merged_df.columns:
                        practical_view = pd.to_numeric(merged_df.loc[nlp_candidates.index, "Score"], errors="coerce")
                        nlp_candidates["Practical Code (Out of 100)"] = practical_view.where(practical_view.gt(0), other=pd.NA)
                    if feedback_col and feedback_col in merged_df.columns:
                        nlp_candidates["Trainer Feedback"] = merged_df.loc[nlp_candidates.index, feedback_col]
                    with st.container(border=True):
                        st.dataframe(nlp_candidates.fillna("N/A"), use_container_width=True, hide_index=True)
        else:
            st.warning("No trainer feedback column was detected for the selected week.")

    with main_tabs[2]:
        persona_mode = detect_persona_mode(active_df, assessment_eval_df)
        render_section_header("Executive View", "Persona intelligence is always available, with full or limited mode based on uploaded data.")
        if persona_mode == "limited":
            render_banner("Limited Persona Mode – Partial Data Available", tone="warning")
        elif persona_mode == "empty":
            render_banner("No subjectivity or practical records are available for persona analytics yet.", tone="danger")

        if merged_df.empty:
            st.warning("Persona analytics cannot be generated because no student records matched the current filters.")
        else:
            summary_text = "Full persona model is active using both subjectivity and practical signals." if persona_mode == "full" else "Persona classifications are currently based on the data available for the selected cohort."
            render_banner(summary_text, tone="info")
            valid_personas = merged_df[merged_df["Student Persona"] != "Unclassified"].copy()
            if valid_personas.empty:
                st.info("Student records exist, but there is not enough rated data yet to form persona segments.")
            else:
                exec_col1, exec_col2 = st.columns(2)
                with exec_col1:
                    with st.container(border=True):
                        describe_graph("Persona Distribution", "Shows how the current cohort is split across subjectivity-practical learner personas. Slice labels show the percentage share of students in each persona.")
                        persona_counts = valid_personas.groupby("Student Persona").size().reset_index(name="Student Count")
                        fig_persona = px.pie(persona_counts, names="Student Persona", values="Student Count", hole=0.56, color="Student Persona", color_discrete_map=PERSONA_COLORS)
                        fig_persona.update_traces(
                            texttemplate="%{percent:.1%}",
                            hovertemplate="<b>%{label}</b><br>Students: %{value}<br>Share: %{percent:.1%}<extra></extra>",
                        )
                        fig_persona.update_layout(
                            showlegend=True,
                            margin=dict(t=8, b=8, l=8, r=8),
                            height=350,
                            legend=dict(orientation="h", y=-0.18, x=0, title_text="Persona"),
                        )
                        st.plotly_chart(fig_persona, use_container_width=True)
                with exec_col2:
                    with st.container(border=True):
                        describe_graph("Risk Volume by Batch", "Compares the concentration of personas across batches so trainers can identify where intervention load is building. Labels inside the bars show student headcount.")
                        batch_counts = valid_personas.groupby(["Assigned Batch", "Student Persona"]).size().reset_index(name="Count")
                        fig_risk = px.bar(batch_counts, y="Assigned Batch", x="Count", color="Student Persona", orientation="h", barmode="stack", text="Count", color_discrete_map=PERSONA_COLORS)
                        fig_risk.update_traces(texttemplate="%{text}", textposition="inside", hovertemplate="<b>%{y}</b><br>Persona: %{fullData.name}<br>Students: %{x}<extra></extra>")
                        fig_risk.update_layout(margin=dict(t=8, b=0, l=0, r=0), height=350, legend=dict(orientation="h", y=-0.18, x=0, title_text="Persona"), xaxis_title="Number of Students", yaxis_title="")
                        st.plotly_chart(fig_risk, use_container_width=True)

            render_section_header("Action Roster", "Student-level subjectivity, practical, and persona view for trainer intervention.")
            roster_columns = ["Candidate Name", "Assigned Batch"]
            if "Score" in merged_df.columns:
                roster_columns.append("Score")
            if "Unified Combined Score (%)" in merged_df.columns:
                roster_columns.append("Unified Combined Score (%)")
            roster_columns.append("Student Persona")
            action_df = merged_df[roster_columns].copy()
            action_df["Assessment Score (Out of 100)"] = calculate_assessment_scores(merged_df, rating_col, total_score_col, total_marks_col)
            action_df["Assessment Grade"] = action_df["Assessment Score (Out of 100)"].apply(score_to_grade)
            rename_map = {"Score": "AI Code Score (Out of 100)"}
            with st.container(border=True):
                st.dataframe(action_df.rename(columns=rename_map), use_container_width=True, hide_index=True)

    with main_tabs[3]:
        render_individual_performance_panel(
            merged_df,
            assessment_eval_df,
            week_map,
            rating_col,
            feedback_col,
            total_score_col=total_score_col,
            total_marks_col=total_marks_col,
        )
