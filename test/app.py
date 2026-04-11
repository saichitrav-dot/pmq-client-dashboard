from __future__ import annotations

import ast
import base64
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from io import BytesIO
from io import StringIO
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from urllib.parse import unquote, urlparse
from zipfile import ZipFile

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from modules import task2_assessment, task3_attendance, task4_correlation, task5_correlation_plus, task6_top_brains, task7_student_360

try:
    from pylint.lint import Run as PylintRun
    from pylint.reporters.json_reporter import JSONReporter
except ImportError:
    PylintRun = None
    JSONReporter = None

try:
    from radon.complexity import cc_visit
    from radon.metrics import mi_visit
    from radon.raw import analyze as radon_raw_analyze
except ImportError:
    cc_visit = None
    mi_visit = None
    radon_raw_analyze = None

try:
    from bandit.core import config as bandit_config
    from bandit.core import manager as bandit_manager
except ImportError:
    bandit_config = None
    bandit_manager = None


GITHUB_API = "https://api.github.com"
EXTENSION_LANGUAGE_MAP: Dict[str, str] = {
    ".py": "Python",
    ".ipynb": "Notebook",
    ".java": "Java",
    ".js": "JavaScript",
    ".jsx": "JavaScript",
    ".mjs": "JavaScript",
    ".cjs": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".mts": "TypeScript",
    ".cts": "TypeScript",
    ".cs": "C#",
    ".robot": "Robot Framework",
}
SUPPORTED_EXTENSIONS = set(EXTENSION_LANGUAGE_MAP)
EXCLUDED_PARTS = {
    "venv",
    ".venv",
    ".git",
    "build",
    "dist",
    "node_modules",
    "__pycache__",
    "bin",
    "obj",
    "packages",
    ".vs",
    ".metadata",
    ".settings",
    ".idea",
    ".gradle",
    ".mvn",
    "target",
    "out",
    "coverage",
    "testresults",
}
EXCLUDED_FILE_SUFFIXES = {
    ".designer.cs",
    ".generated.cs",
    ".g.cs",
    ".g.i.cs",
    ".assemblyinfo.cs",
    ".min.js",
    ".min.ts",
}
EXCLUDED_FILE_NAMES = {
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
}
RESERVED_GITHUB_ROOT_SEGMENTS = {
    "users",
    "orgs",
    "settings",
    "topics",
    "collections",
    "marketplace",
    "apps",
    "features",
    "site",
    "about",
    "login",
    "signup",
    "explore",
}
MAX_FILE_SIZE_BYTES = 200_000
MAX_NOTEBOOK_SIZE_BYTES = 2_000_000
MAX_FILES_PER_REPOSITORY = 40
MAX_AI_FILES = 6
MAX_AI_CHARS = 22_000
REQUEST_TIMEOUT = 30
TASK1_WORKBOOK_BATCH_SIZE = 50
SUPPORTED_STACKS_MESSAGE = ".py, .ipynb, .java, .js, .jsx, .ts, .tsx, .cs, and .robot"
ALL_WEEKS_LABEL = "All Week Assessment Data"
ALL_ASSESSMENTS_LABEL = "All Assessments"
ALL_ASSIGNMENTS_LABEL = "All Assignments"
ASSESSMENT_SCORE_TRACK = "Assessment Scores"
ASSIGNMENT_SCORE_TRACK = "Assignment Scores"
ALL_GITHUB_TRACKS_LABEL = "All GitHub Scores"
SPECIAL_RESULT_STATUSES = {"Absent", "Dropout", "No Show"}
SCORE_WEIGHTS: Dict[str, int] = {
    "Code Quality": 25,
    "Structure": 20,
    "Logic Correctness": 20,
    "Security": 15,
    "Documentation": 5,
    "AI Evaluation": 15,
}

SUBMISSION_NAME_KEYS = {
    "candidatename",
    "studentname",
    "name",
    "trainee",
    "traineename",
    "candidate",
    "fullname",
}
SUBMISSION_SOURCE_KEYS = {
    "git",
    "github",
    "link",
    "links",
    "giturl",
    "githuburl",
    "githublink",
    "gitlink",
    "assessmentlink",
    "assessmenturl",
    "repourl",
    "repository",
    "repo",
    "repositorylink",
    "githubrepo",
    "projectsource",
    "source",
    "assignmentlink",
    "assignmenturl",
    "submissionlink",
    "submissionurl",
    "projectlink",
    "projecturl",
    "solutionlink",
    "solutionurl",
}
SUBMISSION_TECH_KEYS = {
    "technology",
    "technologyframework",
    "tech",
    "techstack",
    "language",
    "programminglanguage",
    "stack",
    "framework",
    "tool",
}
SUBMISSION_COLLEGE_KEYS = {
    "college",
}
SUBMISSION_ID_KEYS = {
    "supersetid",
    "candidateid",
    "studentid",
    "studentcode",
    "rollnumber",
    "rollno",
    "id",
}
WORKBOOK_HEADER_SCAN_ROWS = 20
NOT_EVALUATED_CLASSIFICATION = "Not Evaluated"
VALIDATION_RULE_PROFILES: Dict[str, Dict[str, Any]] = {
    "Resilient (Recommended)": {
        "skip_blank_links": True,
        "decode_github_subpaths": True,
        "invalid_link_action": "mark_not_evaluated",
        "missing_code_action": "mark_not_evaluated",
        "exclude_failed_from_averages": True,
    },
    "Strict Review": {
        "skip_blank_links": True,
        "decode_github_subpaths": True,
        "invalid_link_action": "mark_not_evaluated",
        "missing_code_action": "mark_not_evaluated",
        "exclude_failed_from_averages": True,
    },
}

_json_cache: Dict[Tuple[str, str], Any] = {}
_blob_cache: Dict[Tuple[str, str], str] = {}
_repo_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
_ai_review_cache: Dict[str, Any] = {}
_cache_lock = threading.Lock()
_command_cache: Dict[Tuple[str, ...], Optional[str]] = {}
_app_logger: Optional[logging.Logger] = None


def get_app_logger() -> logging.Logger:
    global _app_logger
    if _app_logger is not None:
        return _app_logger

    log_dir = Path(__file__).resolve().parent / "runtime_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "task1_backend.log"

    logger = logging.getLogger("evaluation_portal.task1")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = RotatingFileHandler(log_path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
    _app_logger = logger
    return logger


def log_task1_event(event: str, **fields: Any) -> None:
    logger = get_app_logger()
    payload = {"event": event}
    payload.update(fields)
    try:
        logger.info(json.dumps(payload, default=str, ensure_ascii=True))
    except Exception:
        logger.info("%s | %s", event, str(fields))


def sync_task1_results_to_shared_eval() -> None:
    try:
        shared_eval_df, sync_notices = task2_assessment.build_eval_dataframe_from_task1_session()
        st.session_state["task1_shared_eval_results"] = shared_eval_df.to_dict("records")
        log_task1_event(
            "task1_shared_eval_synced",
            shared_rows=len(shared_eval_df),
            source_results=len(st.session_state.get("evaluation_results", [])),
            notices=len(sync_notices),
        )
    except Exception as exc:
        log_task1_event("task1_shared_eval_sync_failed", error=str(exc))

PORTAL_MODULES = [
    "GitHub Evaluation Engine",
    "Top Brains Performance",
    "Assessment & Feedback Analytics",
    "Executive Attendance Intelligence",
    "Performance Magic Quadrant Plus",
]


@dataclass
class RepositoryFile:
    path: str
    sha: str
    size: int
    language: str
    content: str


@dataclass
class StaticAnalysisResult:
    pylint_score: float = 0.0
    pylint_messages: List[Dict[str, Any]] = field(default_factory=list)
    naming_issues: int = 0
    error_count: int = 0
    fatal_count: int = 0
    documentation_issues: int = 0
    average_complexity: float = 0.0
    maintainability_index: float = 0.0
    security_issues: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class StructuralAnalysisResult:
    function_count: int = 0
    class_count: int = 0
    method_count: int = 0
    max_nesting_depth: int = 0
    modularity_score: float = 0.0
    unique_libraries: List[str] = field(default_factory=list)
    docstring_ratio: float = 0.0
    python_files_analyzed: int = 0
    java_files_analyzed: int = 0
    other_files_analyzed: int = 0
    detected_languages: List[str] = field(default_factory=list)
    detected_frameworks: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class AIReviewResult:
    provider: str = ""
    model: str = ""
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    suggested_improvements: List[str] = field(default_factory=list)
    security: List[str] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    summary: str = ""
    raw_response: str = ""
    error: Optional[str] = None


@dataclass
class EvaluationResult:
    trainee: str
    repository: str
    primary_file: str
    files_analyzed: int
    score: int
    classification: str
    experience_profile: str
    strengths: List[str]
    weaknesses: List[str]
    security: List[str]
    primary_gap: str
    static_analysis: StaticAnalysisResult
    structural_analysis: StructuralAnalysisResult
    ai_review: AIReviewResult
    file_tree: str
    primary_snapshot: str
    reviewed_files: List[RepositoryFile]
    score_breakdown: Dict[str, float]
    review_strictness: int
    status: str
    issue_type: str
    issue_message: str
    output: Dict[str, Any]
    worksheet: str = ""
    declared_technology: str = ""
    languages_tested: str = ""
    superset_id: str = ""
    assessment_week: str = ""
    source_file: str = ""
    evaluation_track: str = ASSESSMENT_SCORE_TRACK


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def normalize_score(value: float, minimum: float, maximum: float) -> float:
    if maximum <= minimum:
        return 0.0
    return clamp((value - minimum) / (maximum - minimum) * 100.0, 0.0, 100.0)


def score_from_penalty(penalty: float) -> float:
    return clamp(100.0 - penalty, 0.0, 100.0)


def strictness_multiplier(review_strictness: int) -> float:
    return {
        25: 0.85,
        50: 1.0,
        75: 1.15,
        100: 1.3,
    }.get(review_strictness, 1.0)


def bucket_rating(score: float) -> str:
    if score >= 85:
        return "Excellent"
    if score >= 70:
        return "Good"
    if score >= 50:
        return "Average"
    return "Needs Attention"


def scoring_profile(experience_profile: str) -> Dict[str, Any]:
    if experience_profile == "Experienced":
        return {
            "pylint_base": 20.0,
            "pylint_scale": 7.5,
            "complexity_threshold": 8.0,
            "complexity_penalty": 7.5,
            "error_weight": 8.0,
            "fatal_weight": 15.0,
            "naming_weight": 2.0,
            "nesting_threshold": 3.0,
            "nesting_penalty": 15.0,
            "doc_penalty": 6.0,
            "doc_penalty_cap": 12,
            "penalty_scale": 1.0,
            "baseline": 50.0,
            "baseline_center": 50.0,
            "baseline_scale": 1.0,
            "thresholds": {"Excellent": 85, "Good": 70, "Needs Attention": 50},
        }
    return {
        "pylint_base": 35.0,
        "pylint_scale": 6.0,
        "complexity_threshold": 10.0,
        "complexity_penalty": 4.5,
        "error_weight": 5.0,
        "fatal_weight": 14.0,
        "naming_weight": 0.8,
        "nesting_threshold": 4.0,
        "nesting_penalty": 8.0,
        "doc_penalty": 2.5,
        "doc_penalty_cap": 8,
        "penalty_scale": 0.85,
        "baseline": 60.0,
        "baseline_center": 55.0,
        "baseline_scale": 0.7,
        "thresholds": {"Excellent": 90, "Good": 80, "Needs Attention": 60},
    }


def security_penalty_weight(issue: Dict[str, Any]) -> float:
    severity = str(issue.get("severity", "")).upper()
    return {
        "HIGH": 14.0,
        "MEDIUM": 8.0,
        "LOW": 4.0,
    }.get(severity, 6.0)


def safe_json_loads(payload: str) -> Any:
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", payload, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def is_blank_value(value: Any) -> bool:
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null", "na", "n/a"}


def normalize_superset_id(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return ""
    return text[:-2] if text.endswith(".0") else text


def extract_week_label(value: Any) -> Optional[str]:
    match = re.search(r"\b(?:week|wk)[\s\-_]*0*(\d+)\b", str(value or ""), flags=re.IGNORECASE)
    if not match:
        return None
    return f"Week {int(match.group(1))}"


def extract_assignment_cycle_label(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    if not text:
        return None
    cleaned = re.sub(r"\s+", " ", text)
    word_map = {
        "first": "First",
        "second": "Second",
        "third": "Third",
        "fourth": "Fourth",
        "fifth": "Fifth",
        "sixth": "Sixth",
        "seventh": "Seventh",
        "eighth": "Eighth",
        "ninth": "Ninth",
        "tenth": "Tenth",
    }
    lowered = cleaned.lower()
    for word, label in word_map.items():
        if re.search(rf"\b{word}\b", lowered):
            return label

    ordinal_match = re.search(r"\b(\d+)(?:st|nd|rd|th)\b", lowered)
    if ordinal_match:
        ordinal_value = int(ordinal_match.group(1))
        return word_map.get(
            {
                1: "first",
                2: "second",
                3: "third",
                4: "fourth",
                5: "fifth",
                6: "sixth",
                7: "seventh",
                8: "eighth",
                9: "ninth",
                10: "tenth",
            }.get(ordinal_value, ""),
            f"Assignment {ordinal_value}",
        )

    numeric_match = re.search(r"\bassignment[\s\-_]*(\d+)\b", lowered) or re.search(r"\b(\d+)\b", lowered)
    if numeric_match:
        numeric_value = int(numeric_match.group(1))
        return {
            1: "First",
            2: "Second",
            3: "Third",
            4: "Fourth",
            5: "Fifth",
            6: "Sixth",
            7: "Seventh",
            8: "Eighth",
            9: "Ninth",
            10: "Tenth",
        }.get(numeric_value, f"Assignment {numeric_value}")

    suffix_match = re.search(r"assignment[\s\-_]*(.*)$", cleaned, flags=re.IGNORECASE)
    if suffix_match:
        suffix = re.sub(r"^[\s:._-]+", "", suffix_match.group(1)).strip()
        if suffix:
            return suffix.title()
    return None


def is_all_cycle_label(label: Any) -> bool:
    normalized = normalize_column_name(label)
    return normalized in {
        normalize_column_name(ALL_WEEKS_LABEL),
        normalize_column_name(ALL_ASSESSMENTS_LABEL),
        normalize_column_name(ALL_ASSIGNMENTS_LABEL),
        "allcycles",
    }


def default_cycle_label_for_track(track: str) -> str:
    return ALL_ASSIGNMENTS_LABEL if normalize_evaluation_track(track) == ASSIGNMENT_SCORE_TRACK else ALL_ASSESSMENTS_LABEL


def track_display_label(track: str) -> str:
    if track == ALL_GITHUB_TRACKS_LABEL:
        return "All GitHub Data"
    normalized_track = normalize_evaluation_track(track)
    if normalized_track == ASSIGNMENT_SCORE_TRACK:
        return "Assignment"
    if normalized_track == ASSESSMENT_SCORE_TRACK:
        return "Assessment"
    return str(track).strip() or "Assessment"


def normalize_cycle_label(label: Any, track: str = ASSESSMENT_SCORE_TRACK) -> str:
    text = str(label or "").strip()
    if not text or is_all_cycle_label(text):
        return default_cycle_label_for_track(track)
    return text


def assessment_week_sort_key(label: str) -> Tuple[int, str]:
    if is_all_cycle_label(label):
        return (0, label)
    text = str(label or "").strip()
    week_match = re.search(r"\bweek[\s\-_]*0*(\d+)\b", text, flags=re.IGNORECASE)
    if week_match:
        return (10 + int(week_match.group(1)), text)
    assignment_order = {
        "first": 1,
        "second": 2,
        "third": 3,
        "fourth": 4,
        "fifth": 5,
        "sixth": 6,
        "seventh": 7,
        "eighth": 8,
        "ninth": 9,
        "tenth": 10,
    }
    normalized = normalize_column_name(text)
    if normalized in assignment_order:
        return (100 + assignment_order[normalized], text)
    generic_number = re.search(r"(\d+)", text)
    return (500 + int(generic_number.group(1)) if generic_number else 999, text)


def sort_week_labels(labels: Iterable[str]) -> List[str]:
    unique_labels = [str(label).strip() for label in labels if str(label).strip()]
    return sorted(dict.fromkeys(unique_labels), key=assessment_week_sort_key)


def classify_submission_source_status(value: Any) -> Optional[str]:
    lowered = str(value or "").strip().lower()
    if lowered.startswith(("http://", "https://", "www.")) or "github.com" in lowered:
        return None
    normalized = re.sub(r"[^a-z0-9]+", "", lowered)
    if not normalized:
        return None
    if normalized in {"absent", "ab"} or "absent" in lowered:
        return "Absent"
    if normalized in {"noshow"} or "no show" in lowered:
        return "No Show"
    if normalized in {"dropout", "drop", "droppedout"} or "drop out" in lowered:
        return "Dropout"
    return None


def parse_imported_workbook_score(value: Any) -> Optional[int]:
    if pd.isna(value):
        return None
    text = str(value or "").strip()
    if not text:
        return None
    if classify_submission_source_status(text):
        return None
    normalized = normalize_column_name(text)
    if normalized == "notevaluated":
        return None
    score_match = re.fullmatch(r"(\d{1,3}(?:\.\d+)?)\s*(?:/100|%)?", text)
    if not score_match:
        return None
    try:
        return int(round(clamp(float(score_match.group(1)), 0.0, 100.0)))
    except Exception:
        return None


def is_imported_not_evaluated_value(value: Any) -> bool:
    return normalize_column_name(value) == "notevaluated"


def get_result_submission_id(result: EvaluationResult) -> str:
    if isinstance(result.output, dict):
        return str(result.output.get("SubmissionID", "")).strip()
    return ""


def notebook_size_limit(file_suffix: str) -> int:
    return MAX_NOTEBOOK_SIZE_BYTES if file_suffix == ".ipynb" else MAX_FILE_SIZE_BYTES


def sanitize_python_source(content: str) -> str:
    cleaned_lines = []
    for line in content.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("%") or stripped.startswith("!"):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def extract_notebook_code(notebook_text: str) -> str:
    try:
        payload = json.loads(notebook_text)
    except json.JSONDecodeError:
        return ""

    code_blocks: List[str] = []
    for cell in payload.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        if isinstance(source, list):
            cell_text = "".join(str(item) for item in source)
        else:
            cell_text = str(source)
        cleaned_lines = []
        for line in cell_text.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("%") or stripped.startswith("!"):
                continue
            cleaned_lines.append(line)
        block = "\n".join(cleaned_lines).strip()
        if block:
            code_blocks.append(block)
    return "\n\n".join(code_blocks).strip()


def sanitize_text_source(content: str) -> str:
    return content.replace("\x00", "").strip() + "\n"


def normalize_repository_content(relative_path: str, raw_content: str) -> Optional[Tuple[str, str]]:
    suffix = Path(relative_path).suffix.lower()
    if suffix == ".ipynb":
        notebook_code = extract_notebook_code(raw_content)
        if not notebook_code:
            return None
        cleaned = sanitize_python_source(notebook_code)
        if not cleaned:
            return None
        return "Python", cleaned + "\n"
    if suffix == ".py":
        cleaned = sanitize_python_source(raw_content)
        return "Python", (cleaned + "\n") if cleaned else ""
    if suffix == ".java":
        return "Java", sanitize_text_source(raw_content)
    if suffix in {".js", ".jsx", ".mjs", ".cjs"}:
        return "JavaScript", sanitize_text_source(raw_content)
    if suffix in {".ts", ".tsx", ".mts", ".cts"}:
        return "TypeScript", sanitize_text_source(raw_content)
    if suffix == ".cs":
        return "C#", sanitize_text_source(raw_content)
    if suffix == ".robot":
        return "Robot Framework", sanitize_text_source(raw_content)
    return None


def build_repository_file(relative_path: str, sha: str, size: int, raw_content: str) -> Optional[RepositoryFile]:
    normalized = normalize_repository_content(relative_path, raw_content)
    if normalized is None:
        return None
    language, content = normalized
    if language == "Python" and not content.strip():
        return None
    return RepositoryFile(
        path=relative_path,
        sha=sha,
        size=size,
        language=language,
        content=content,
    )


def sanitize_repository_input(repo_input: str) -> str:
    candidate = repo_input.strip()
    if not candidate:
        raise ValueError("Repository input is empty.")

    if candidate.startswith("https://") or candidate.startswith("http://"):
        parsed = urlparse(candidate)
        if parsed.netloc.lower() != "github.com":
            raise ValueError("Only github.com repositories are supported.")
        parts = [part for part in parsed.path.strip("/").split("/") if part]
        if parts and parts[0].lower() in RESERVED_GITHUB_ROOT_SEGMENTS:
            raise ValueError("The source must be a GitHub repository or repository subfolder, not a GitHub project or profile page.")
    else:
        parts = [part for part in candidate.strip("/").split("/") if part]

    if len(parts) < 2:
        raise ValueError("Repository must be in owner/repo format.")

    owner, repo = parts[0], parts[1]
    repo = repo.removesuffix(".git")
    full_name = f"{owner}/{repo}"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", full_name):
        raise ValueError("Repository contains unsupported characters.")
    return full_name


def parse_github_reference(repo_input: str) -> Dict[str, str]:
    candidate = repo_input.strip()
    branch = ""
    subpath = ""

    if candidate.startswith("https://") or candidate.startswith("http://"):
        parsed = urlparse(candidate)
        if parsed.netloc.lower() != "github.com":
            raise ValueError("Only github.com repositories are supported.")
        parts = [part for part in parsed.path.strip("/").split("/") if part]
        if parts and parts[0].lower() in RESERVED_GITHUB_ROOT_SEGMENTS:
            raise ValueError("The source must be a GitHub repository or repository subfolder, not a GitHub project or profile page.")
        if len(parts) >= 4 and parts[2] in {"tree", "blob"}:
            branch = parts[3]
            subpath = unquote("/".join(parts[4:]))
        full_name = sanitize_repository_input("/".join(parts[:2]))
    else:
        full_name = sanitize_repository_input(candidate)

    return {
        "owner_repo": full_name,
        "branch": branch,
        "subpath": subpath.strip("/"),
    }


def is_excluded_path(path: str) -> bool:
    normalized_path = path.replace("\\", "/").strip("/")
    if not normalized_path:
        return True
    lowered_parts = [part.lower() for part in Path(normalized_path).parts]
    if any(part in EXCLUDED_PARTS for part in lowered_parts):
        return True
    lowered_name = Path(normalized_path).name.lower()
    if lowered_name in EXCLUDED_FILE_NAMES:
        return True
    if any(lowered_name.endswith(suffix) for suffix in EXCLUDED_FILE_SUFFIXES):
        return True
    return False


def build_headers(github_token: str) -> Dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "ai-code-evaluation-engine",
    }
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"
    return headers


def cached_get_json(url: str, github_token: str) -> Any:
    cache_key = (url, github_token or "")
    with _cache_lock:
        if cache_key in _json_cache:
            return _json_cache[cache_key]

    try:
        response = requests.get(url, headers=build_headers(github_token), timeout=REQUEST_TIMEOUT)
    except Exception as exc:
        log_task1_event(
            "task1_github_api_request_failed",
            url=url,
            timeout_seconds=REQUEST_TIMEOUT,
            has_token=bool(github_token),
            error=str(exc),
        )
        raise
    if response.status_code == 403 and response.headers.get("X-RateLimit-Remaining") == "0":
        log_task1_event(
            "task1_github_api_rate_limited",
            url=url,
            status_code=response.status_code,
            has_token=bool(github_token),
        )
        raise RuntimeError("GitHub API rate limit exceeded. Add a GitHub token in the sidebar to continue.")
    if response.status_code >= 400:
        log_task1_event(
            "task1_github_api_bad_status",
            url=url,
            status_code=response.status_code,
            has_token=bool(github_token),
        )
    response.raise_for_status()
    payload = response.json()

    with _cache_lock:
        _json_cache[cache_key] = payload
    return payload


def cached_get_blob(owner_repo: str, sha: str, github_token: str) -> str:
    cache_key = (f"{owner_repo}:{sha}", github_token or "")
    with _cache_lock:
        if cache_key in _blob_cache:
            return _blob_cache[cache_key]

    url = f"{GITHUB_API}/repos/{owner_repo}/git/blobs/{sha}"
    payload = cached_get_json(url, github_token)
    content = payload.get("content", "")
    encoding = payload.get("encoding", "")
    if encoding != "base64":
        raise ValueError(f"Unsupported blob encoding for {sha}.")
    decoded = base64.b64decode(content).decode("utf-8", errors="replace")

    with _cache_lock:
        _blob_cache[cache_key] = decoded
    return decoded


def determine_primary_file(files: List[RepositoryFile]) -> str:
    if not files:
        return ""
    preferred_names = {
        "main.py",
        "__main__.py",
        "app.py",
        "run.py",
        "main.ipynb",
        "Main.java",
        "Application.java",
        "index.ts",
        "index.js",
        "app.ts",
        "app.js",
        "Program.cs",
        "main.robot",
    }
    ordered_files = sorted(files, key=lambda item: (Path(item.path).name not in preferred_names, -item.size, item.path))
    return ordered_files[0].path


def detect_repository_frameworks(files: List[RepositoryFile]) -> List[str]:
    frameworks: List[str] = []
    combined_text = "\n".join(f"{item.path}\n{item.content[:4000]}" for item in files).lower()
    lowered_paths = [item.path.lower() for item in files]

    framework_rules = [
        (("pytest", "conftest.py", "pytest.ini", "import pytest"), "Pytest"),
        (("@playwright/test", "playwright.", "page.goto(", "browser.new_context"), "Playwright"),
        (("robot framework", "*** test cases ***", "*** keywords ***", "library    "), "Robot Framework"),
        (("xunit", "[fact]", "[theory]", "nunit", "[testmethod]", "[test]"), ".NET Test Framework"),
    ]
    for patterns, label in framework_rules:
        if any(pattern in combined_text for pattern in patterns) and label not in frameworks:
            frameworks.append(label)
    if "Pytest" not in frameworks and any(
        Path(path).name.startswith("test_") or Path(path).name.endswith("_test.py") or Path(path).name == "conftest.py"
        for path in lowered_paths
    ):
        frameworks.append("Pytest")
    if "Playwright" not in frameworks and any("playwright.config." in path or ".spec." in Path(path).name for path in lowered_paths):
        frameworks.append("Playwright")
    if "Robot Framework" not in frameworks and any(path.endswith(".robot") for path in lowered_paths):
        frameworks.append("Robot Framework")
    return frameworks


def build_file_tree(paths: Iterable[str]) -> str:
    root: Dict[str, Any] = {}
    for path in sorted(paths):
        current = root
        parts = Path(path).parts
        for part in parts:
            current = current.setdefault(part, {})

    lines: List[str] = []

    def walk(node: Dict[str, Any], prefix: str = "") -> None:
        entries = list(node.items())
        for index, (name, child) in enumerate(entries):
            connector = "`-- " if index == len(entries) - 1 else "|-- "
            lines.append(f"{prefix}{connector}{name}")
            extension = "    " if index == len(entries) - 1 else "|   "
            walk(child, prefix + extension)

    walk(root)
    return "\n".join(lines)


def build_repository_payload(repository_name: str, source_files: List[RepositoryFile], default_branch: str = "") -> Dict[str, Any]:
    source_files = sorted(source_files, key=lambda item: (-item.size, item.path))[:MAX_FILES_PER_REPOSITORY]
    if not source_files:
        raise ValueError(
            f"No supported code files were found. The engine currently reads {SUPPORTED_STACKS_MESSAGE} files and ignores PDFs and other non-code artifacts."
        )
    primary_file = determine_primary_file(source_files)
    primary_snapshot = next((item.content for item in source_files if item.path == primary_file), "")
    detected_languages = sorted(
        {
            "Notebook (Python)" if Path(item.path).suffix.lower() == ".ipynb" else item.language
            for item in source_files
            if item.language
        }
    )
    detected_frameworks = detect_repository_frameworks(source_files)
    return {
        "repository": repository_name,
        "default_branch": default_branch,
        "files": source_files,
        "primary_file": primary_file,
        "primary_snapshot": primary_snapshot,
        "file_tree": build_file_tree([item.path for item in source_files]),
        "detected_languages": detected_languages,
        "detected_frameworks": detected_frameworks,
    }


def fetch_public_github_repository(repo_input: str) -> Dict[str, Any]:
    reference = parse_github_reference(repo_input)
    owner_repo = reference["owner_repo"]
    branch_candidates = [reference["branch"]] if reference["branch"] else ["main", "master"]
    subpath = reference["subpath"]
    cache_key = (f"publiczip:{owner_repo}:{'|'.join(branch_candidates)}:{subpath}", "")
    with _cache_lock:
        if cache_key in _repo_cache:
            return _repo_cache[cache_key]

    last_error = "Unable to download public repository archive."
    for branch in branch_candidates:
        archive_url = f"https://codeload.github.com/{owner_repo}/zip/refs/heads/{branch}"
        log_task1_event(
            "task1_public_archive_attempt",
            repository=owner_repo,
            branch=branch,
            subpath=subpath,
            url=archive_url,
            timeout_seconds=REQUEST_TIMEOUT,
        )
        try:
            response = requests.get(archive_url, timeout=REQUEST_TIMEOUT)
        except Exception as exc:
            last_error = f"Archive download failed for branch '{branch}'."
            log_task1_event(
                "task1_public_archive_request_failed",
                repository=owner_repo,
                branch=branch,
                subpath=subpath,
                url=archive_url,
                error=str(exc),
            )
            continue
        if response.status_code != 200:
            last_error = f"Archive download failed for branch '{branch}' with status {response.status_code}."
            log_task1_event(
                "task1_public_archive_bad_status",
                repository=owner_repo,
                branch=branch,
                subpath=subpath,
                url=archive_url,
                status_code=response.status_code,
            )
            continue

        source_files: List[RepositoryFile] = []
        with ZipFile(BytesIO(response.content)) as archive:
            members = sorted((item for item in archive.infolist() if not item.is_dir()), key=lambda item: item.filename)
            if not members:
                continue

            root_prefix = members[0].filename.split("/")[0]
            base_prefix = f"{root_prefix}/"
            if subpath:
                base_prefix = f"{base_prefix}{subpath.strip('/')}/"

            for member in members:
                if not member.filename.startswith(base_prefix):
                    continue

                relative_path = member.filename[len(base_prefix):]
                if not relative_path or is_excluded_path(relative_path):
                    continue

                suffix = Path(relative_path).suffix.lower()
                if suffix not in SUPPORTED_EXTENSIONS or member.file_size > notebook_size_limit(suffix):
                    continue

                content = archive.read(member).decode("utf-8", errors="replace")
                repo_file = build_repository_file(relative_path, f"zip:{branch}:{relative_path}", member.file_size, content)
                if repo_file is not None:
                    source_files.append(repo_file)

        repository = build_repository_payload(owner_repo, source_files, branch)
        log_task1_event(
            "task1_public_archive_complete",
            repository=owner_repo,
            branch=branch,
            subpath=subpath,
            file_count=len(repository["files"]),
            languages=",".join(repository.get("detected_languages", [])),
        )
        with _cache_lock:
            _repo_cache[cache_key] = repository
        return repository

    log_task1_event(
        "task1_public_archive_exhausted",
        repository=owner_repo,
        branches=",".join(branch_candidates),
        subpath=subpath,
        error=last_error,
    )
    raise RuntimeError(last_error)


def fetch_github_repository(repo_input: str, github_token: str) -> Dict[str, Any]:
    reference = parse_github_reference(repo_input)
    owner_repo = reference["owner_repo"]
    requested_branch = reference["branch"]
    requested_subpath = reference["subpath"]
    cache_key = (f"{owner_repo}:{requested_branch}:{requested_subpath}", github_token or "")
    with _cache_lock:
        if cache_key in _repo_cache:
            return _repo_cache[cache_key]

    log_task1_event(
        "task1_github_fetch_start",
        repository=owner_repo,
        requested_branch=requested_branch,
        requested_subpath=requested_subpath,
        has_token=bool(github_token),
    )
    try:
        default_branch = requested_branch or cached_get_json(f"{GITHUB_API}/repos/{owner_repo}", github_token)["default_branch"]
        tree_url = f"{GITHUB_API}/repos/{owner_repo}/git/trees/{default_branch}?recursive=1"
        tree_payload = cached_get_json(tree_url, github_token)
        tree_items = tree_payload.get("tree", [])

        source_files: List[RepositoryFile] = []
        path_prefix = f"{requested_subpath.strip('/')}/" if requested_subpath else ""

        for item in tree_items:
            path = item.get("path", "")
            if item.get("type") != "blob":
                continue
            if path_prefix and not path.startswith(path_prefix):
                continue

            relative_path = path[len(path_prefix):] if path_prefix else path
            if is_excluded_path(relative_path):
                continue

            suffix = Path(relative_path).suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                continue

            size = int(item.get("size", 0))
            if size > notebook_size_limit(suffix):
                continue

            content = cached_get_blob(owner_repo, item["sha"], github_token)
            repo_file = build_repository_file(relative_path, item["sha"], size, content)
            if repo_file is not None:
                source_files.append(repo_file)

        repository = build_repository_payload(owner_repo, source_files, default_branch)
    except Exception as exc:
        log_task1_event(
            "task1_github_fetch_exception",
            repository=owner_repo,
            requested_branch=requested_branch,
            requested_subpath=requested_subpath,
            has_token=bool(github_token),
            error=str(exc),
        )
        if github_token:
            raise
        repository = fetch_public_github_repository(repo_input)

    log_task1_event(
        "task1_github_fetch_complete",
        repository=owner_repo,
        resolved_branch=repository.get("default_branch", ""),
        requested_subpath=requested_subpath,
        file_count=len(repository["files"]),
        languages=",".join(repository.get("detected_languages", [])),
    )
    with _cache_lock:
        _repo_cache[cache_key] = repository
    return repository


def fetch_local_repository(repo_input: str) -> Dict[str, Any]:
    root = Path(repo_input).expanduser().resolve()
    if not root.exists():
        raise ValueError(f"Local path does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"Local path must be a folder: {root}")

    cache_key = (f"local:{root}", "")
    with _cache_lock:
        if cache_key in _repo_cache:
            return _repo_cache[cache_key]

    source_files: List[RepositoryFile] = []
    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue

        relative_path = file_path.relative_to(root).as_posix()
        if is_excluded_path(relative_path):
            continue

        suffix = file_path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            continue

        size = file_path.stat().st_size
        if size > notebook_size_limit(suffix):
            continue

        content = file_path.read_text(encoding="utf-8", errors="replace")
        repo_file = build_repository_file(relative_path, f"local:{relative_path}", size, content)
        if repo_file is not None:
            source_files.append(repo_file)

    repository = build_repository_payload(str(root), source_files)

    with _cache_lock:
        _repo_cache[cache_key] = repository
    return repository


def fetch_repository(repo_input: str, github_token: str, source_mode: str = "auto") -> Dict[str, Any]:
    candidate_path = Path(repo_input).expanduser()
    if source_mode == "local" or (source_mode == "auto" and candidate_path.exists()):
        return fetch_local_repository(repo_input)
    return fetch_github_repository(repo_input, github_token)


def normalize_column_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def deduplicate_headers(values: Iterable[Any]) -> List[str]:
    counts: Dict[str, int] = {}
    headers: List[str] = []
    for index, value in enumerate(values, start=1):
        header = str(value).strip()
        if is_blank_value(header):
            header = f"Column {index}"
        seen_count = counts.get(header, 0)
        counts[header] = seen_count + 1
        if seen_count:
            header = f"{header}_{seen_count + 1}"
        headers.append(header)
    return headers


def find_submission_column(normalized_columns: Dict[str, str], exact_keys: set[str], fuzzy_tokens: Iterable[str]) -> Optional[str]:
    for key, original in normalized_columns.items():
        if key in exact_keys:
            return original
    for key, original in normalized_columns.items():
        if any(token in key for token in fuzzy_tokens):
            return original
    return None


def find_submission_source_columns(columns: Iterable[Any]) -> List[str]:
    source_columns: List[str] = []
    fuzzy_tokens = ("github", "git", "repo", "repository", "assignment", "assessment", "assesment", "submission", "project", "solution", "link")
    for column in columns:
        original = str(column).strip()
        normalized = normalize_column_name(original)
        if not normalized:
            continue
        if normalized in SUBMISSION_SOURCE_KEYS or any(token in normalized for token in fuzzy_tokens):
            source_columns.append(original)
            continue
        if extract_week_label(original) or extract_assignment_cycle_label(original):
            source_columns.append(original)
    return source_columns


def parse_header_date(value: Any) -> Optional[pd.Timestamp]:
    text = str(value or "").strip()
    if not text:
        return None
    cleaned = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", text, flags=re.IGNORECASE)
    date_patterns = [
        r"(\d{1,2}\s*[-/ ]\s*[A-Za-z]{3,9}\s*[-/ ]\s*\d{2,4})",
        r"([A-Za-z]{3,9}\s*[-/ ]\s*\d{1,2}\s*[-/ ]\s*\d{2,4})",
        r"(\d{1,2}\s*[-/ ]\s*[A-Za-z]{3,9})",
        r"([A-Za-z]{3,9}\s*[-/ ]\s*\d{1,2})",
    ]
    for pattern in date_patterns:
        match = re.search(pattern, cleaned)
        if not match:
            continue
        parsed = pd.to_datetime(match.group(1), dayfirst=True, errors="coerce")
        if not pd.isna(parsed):
            return parsed
    parsed = pd.to_datetime(cleaned, dayfirst=True, errors="coerce")
    return None if pd.isna(parsed) else parsed


def infer_submission_track(column_name: Any, default_track: str = ASSESSMENT_SCORE_TRACK) -> str:
    normalized = normalize_column_name(column_name)
    if any(token in normalized for token in ("assignment", "dailyassignment")):
        return ASSIGNMENT_SCORE_TRACK
    if any(token in normalized for token in ("assessment", "assesment", "week", "wk")):
        return ASSESSMENT_SCORE_TRACK
    return normalize_evaluation_track(default_track)


def derive_submission_cycle_metadata(
    source_columns: List[str],
    sheet_name: str = "",
    default_track: str = ASSESSMENT_SCORE_TRACK,
) -> Dict[str, Dict[str, str]]:
    if not source_columns:
        return {}

    metadata: Dict[str, Dict[str, str]] = {}
    track_groups: Dict[str, List[str]] = {}
    for column in source_columns:
        track = infer_submission_track(column, default_track)
        track_groups.setdefault(track, []).append(column)

    for track, grouped_columns in track_groups.items():
        explicit_labels: Dict[str, str] = {}
        for column in grouped_columns:
            cycle_label = extract_assignment_cycle_label(column) if track == ASSIGNMENT_SCORE_TRACK else extract_week_label(column)
            if cycle_label:
                explicit_labels[column] = cycle_label
        if explicit_labels:
            for column, cycle_label in explicit_labels.items():
                metadata[column] = {"track": track, "cycle": cycle_label}
            remaining_columns = [column for column in grouped_columns if column not in explicit_labels]
        else:
            remaining_columns = list(grouped_columns)

        if not remaining_columns:
            continue

        dated_columns = [(column, parse_header_date(column)) for column in remaining_columns]
        if any(parsed is not None for _, parsed in dated_columns):
            ordered_columns = [column for column, _ in sorted(dated_columns, key=lambda item: item[1] or pd.Timestamp.max)]
            for index, column in enumerate(ordered_columns, start=1):
                cycle_label = (
                    {1: "First", 2: "Second", 3: "Third"}.get(index, f"Assignment {index}")
                    if track == ASSIGNMENT_SCORE_TRACK
                    else f"Week {index}"
                )
                metadata[column] = {"track": track, "cycle": cycle_label}
            continue

        for index, column in enumerate(remaining_columns, start=1):
            if track == ASSIGNMENT_SCORE_TRACK:
                cycle_label = {1: "First", 2: "Second", 3: "Third"}.get(index, f"Assignment {index}")
            else:
                sheet_week = extract_week_label(sheet_name)
                cycle_label = sheet_week if sheet_week and len(remaining_columns) == 1 else f"Week {index}"
            metadata[column] = {"track": track, "cycle": cycle_label}
    return metadata


def derive_submission_week_labels(source_columns: List[str], sheet_name: str = "", default_track: str = ASSESSMENT_SCORE_TRACK) -> Dict[str, str]:
    return {
        column: item["cycle"]
        for column, item in derive_submission_cycle_metadata(source_columns, sheet_name, default_track).items()
    }


def detect_technology_labels(value: Any) -> List[str]:
    if is_blank_value(value):
        return []

    lowered = str(value).strip().lower()
    normalized = normalize_column_name(value)
    labels: List[str] = []
    detection_rules = [
        (lambda text, norm: "python" in text or "python" in norm, "Python"),
        (lambda text, norm: "java" in text or "java" in norm, "Java"),
        (lambda text, norm: "typescript" in text or "typescript" in norm, "TypeScript"),
        (lambda text, norm: "javascript" in text or "javascript" in norm, "JavaScript"),
        (lambda text, norm: ".net" in text or "dotnet" in norm or "csharp" in norm or re.search(r"(?<![a-z0-9])c#(?![a-z0-9])", text) is not None, ".NET"),
        (lambda text, norm: "pytest" in text or "pytest" in norm, "Pytest"),
        (lambda text, norm: "robot framework" in text or "robotframework" in norm or re.search(r"(?<![a-z0-9])robot(?![a-z0-9])", text) is not None, "Robot Framework"),
        (lambda text, norm: "playwright" in text or "play wright" in text, "Playwright"),
    ]
    for matcher, label in detection_rules:
        if matcher(lowered, normalized):
            labels.append(label)
    return labels


def infer_declared_technology(sheet_name: str = "", explicit_value: str = "") -> str:
    labels: List[str] = []
    for label in detect_technology_labels(explicit_value):
        if label not in labels:
            labels.append(label)
    for label in detect_technology_labels(sheet_name):
        if label not in labels:
            labels.append(label)
    if labels:
        return ", ".join(labels)

    explicit_text = str(explicit_value or "").strip()
    if explicit_text and explicit_text.lower() not in {"nan", "none", "null", "n/a"}:
        return explicit_text
    return ""


def clean_workbook_label(value: Any, strip_extension: bool = False) -> str:
    if is_blank_value(value):
        return ""
    text = str(value).strip()
    if strip_extension:
        text = Path(text).stem
    return re.sub(r"\s+", " ", text).strip()


def infer_college_name(explicit_value: Any = "", source_label: str = "", sheet_name: str = "") -> str:
    explicit_text = clean_workbook_label(explicit_value)
    if explicit_text:
        return explicit_text
    sheet_text = clean_workbook_label(sheet_name)
    if normalize_column_name(sheet_text) not in {"", "workbook", "sheet1"}:
        return sheet_text
    source_text = clean_workbook_label(source_label, strip_extension=True)
    if source_text:
        return source_text
    return "Unspecified College"


def languages_tested_from_files(files: List[RepositoryFile]) -> str:
    if not files:
        return ""

    labels: List[str] = []
    for item in files:
        suffix = Path(item.path).suffix.lower()
        if suffix == ".ipynb":
            label = "Notebook (Python)"
        else:
            label = item.language
        if label and label not in labels:
            labels.append(label)
    return ", ".join(labels)


def detect_submission_header_row(raw_df: pd.DataFrame) -> int:
    search_limit = min(len(raw_df), WORKBOOK_HEADER_SCAN_ROWS)
    best_row = 0
    best_score = -1
    for row_index in range(search_limit):
        row_values = {
            normalize_column_name(value)
            for value in raw_df.iloc[row_index].tolist()
            if not is_blank_value(value)
        }
        if not row_values:
            continue
        score = 0
        if row_values & SUBMISSION_NAME_KEYS:
            score += 2
        elif any(any(token in value for token in ("name", "candidate", "student", "trainee")) for value in row_values):
            score += 1
        if row_values & SUBMISSION_ID_KEYS:
            score += 1
        elif any(any(token in value for token in ("superset", "roll", "studentid", "candidateid")) for value in row_values):
            score += 1
        if row_values & SUBMISSION_SOURCE_KEYS:
            score += 3
        elif any(any(token in value for token in ("github", "git", "repo", "repository")) for value in row_values):
            score += 2
        if score > best_score:
            best_score = score
            best_row = row_index
    return best_row


def prepare_submission_dataframe(raw_df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame()

    header_row = detect_submission_header_row(raw_df)
    prepared = raw_df.iloc[header_row + 1 :].copy()
    prepared.columns = deduplicate_headers(raw_df.iloc[header_row].tolist())
    prepared = prepared.dropna(how="all").reset_index(drop=True)
    if prepared.empty:
        return pd.DataFrame()
    prepared["Worksheet"] = sheet_name
    return prepared


def annotate_submission_preview(dataframe: pd.DataFrame, sheet_name: str, source_label: str = "") -> pd.DataFrame:
    prepared = dataframe.copy()
    normalized_columns = {normalize_column_name(column): column for column in prepared.columns}
    tech_column = find_submission_column(
        normalized_columns,
        SUBMISSION_TECH_KEYS,
        ("technology", "language", "stack", "framework", "tool"),
    )
    college_column = find_submission_column(
        normalized_columns,
        SUBMISSION_COLLEGE_KEYS,
        ("college",),
    )
    if "Worksheet" not in prepared.columns:
        prepared["Worksheet"] = sheet_name
    prepared["Source File"] = clean_workbook_label(source_label)
    prepared["College"] = prepared.apply(
        lambda row: infer_college_name(row.get(college_column, "") if college_column else "", source_label, row.get("Worksheet", sheet_name)),
        axis=1,
    )
    prepared["Declared Technology"] = prepared.apply(
        lambda row: infer_declared_technology(sheet_name, row.get(tech_column, "") if tech_column else ""),
        axis=1,
    )
    return prepared


def build_workbook_source_preview(
    dataframe: pd.DataFrame,
    selected_week: str = ALL_ASSESSMENTS_LABEL,
    selected_track: str = ALL_GITHUB_TRACKS_LABEL,
) -> pd.DataFrame:
    if dataframe is None or dataframe.empty:
        return pd.DataFrame(columns=["SL NO", "Student Name", "Superset ID", "Git Hub Link"])

    normalized_columns = {normalize_column_name(column): column for column in dataframe.columns}
    serial_column = find_submission_column(
        normalized_columns,
        {"slno", "serialno", "sno", "srno", "slnumber", "serialnumber"},
        ("slno", "serialno", "sno", "srno", "serial", "row"),
    )
    name_column = find_submission_column(
        normalized_columns,
        SUBMISSION_NAME_KEYS,
        ("fullname", "studentname", "candidatename", "trainee"),
    )
    superset_column = find_submission_column(
        normalized_columns,
        SUBMISSION_ID_KEYS,
        ("superset", "studentid", "candidateid", "roll"),
    )
    source_columns = find_submission_source_columns(dataframe.columns)
    sheet_label = str(dataframe.get("Worksheet", pd.Series([""])).iloc[0] if "Worksheet" in dataframe.columns and not dataframe.empty else "")
    source_metadata = derive_submission_cycle_metadata(source_columns, sheet_label)
    if selected_track != ALL_GITHUB_TRACKS_LABEL:
        source_metadata = {
            column: item
            for column, item in source_metadata.items()
            if item.get("track") == normalize_evaluation_track(selected_track)
        }

    preview_data: Dict[str, Any] = {
        "SL NO": dataframe[serial_column] if serial_column else range(1, len(dataframe) + 1),
        "Student Name": dataframe[name_column] if name_column else "",
        "Superset ID": dataframe[superset_column] if superset_column else "",
    }
    week_items = sort_week_labels(item.get("cycle", "") for item in source_metadata.values())
    if not is_all_cycle_label(selected_week):
        week_items = [label for label in week_items if label == selected_week]
    if week_items:
        multiple_tracks = len({item.get("track", ASSESSMENT_SCORE_TRACK) for item in source_metadata.values()}) > 1
        for week_label in week_items:
            source_column = next((column for column, item in source_metadata.items() if item.get("cycle") == week_label), None)
            if source_column:
                column_label = week_label
                if multiple_tracks:
                    metadata = source_metadata.get(source_column, {})
                    column_label = f"{track_display_label(metadata.get('track', ASSESSMENT_SCORE_TRACK))} - {week_label}"
                preview_data[column_label] = dataframe[source_column]
    else:
        source_column = find_submission_column(
            normalized_columns,
            SUBMISSION_SOURCE_KEYS,
            ("github", "git", "repo", "repository", "assessment", "assesment", "assignment", "submission", "project", "solution", "link"),
        )
        preview_data["Git Hub Link"] = dataframe[source_column] if source_column else ""

    preview = pd.DataFrame(preview_data)
    inferred_tracks = {item.get("track", ASSESSMENT_SCORE_TRACK) for item in source_metadata.values()}
    if len(inferred_tracks) == 1:
        preview["Evaluation Track"] = track_display_label(next(iter(inferred_tracks)))
    if "College" in dataframe.columns:
        preview["College"] = dataframe["College"].fillna("")
    if "Source File" in dataframe.columns and dataframe["Source File"].nunique(dropna=True) > 1:
        preview["Source File"] = dataframe["Source File"].fillna("")
    preview = preview.fillna("")
    preview["SL NO"] = preview["SL NO"].replace("", pd.NA)
    preview["SL NO"] = preview["SL NO"].fillna(pd.Series(range(1, len(preview) + 1), index=preview.index))
    return preview


def extract_submissions_from_dataframe(
    dataframe: pd.DataFrame,
    sheet_name: str = "",
    source_label: str = "",
    evaluation_track: str = ASSESSMENT_SCORE_TRACK,
) -> Tuple[List[Dict[str, str]], List[str], List[Dict[str, str]], List[EvaluationResult], Dict[str, Dict[str, Any]]]:
    normalized_columns = {normalize_column_name(column): column for column in dataframe.columns}
    name_column = find_submission_column(normalized_columns, SUBMISSION_NAME_KEYS, ("fullname", "studentname", "candidatename", "trainee"))
    source_columns = find_submission_source_columns(dataframe.columns)
    source_cycle_metadata = derive_submission_cycle_metadata(source_columns, sheet_name, evaluation_track)
    superset_column = find_submission_column(normalized_columns, SUBMISSION_ID_KEYS, ("superset", "studentid", "candidateid", "roll"))
    tech_column = find_submission_column(
        normalized_columns,
        SUBMISSION_TECH_KEYS,
        ("technology", "language", "stack", "framework", "tool"),
    )
    college_column = find_submission_column(
        normalized_columns,
        SUBMISSION_COLLEGE_KEYS,
        ("college",),
    )

    notices: List[str] = []
    manual_candidates: List[Dict[str, str]] = []
    imported_results: List[EvaluationResult] = []
    submission_cell_map: Dict[str, Dict[str, Any]] = {}
    if not source_cycle_metadata:
        label = f"Sheet '{sheet_name}'" if sheet_name else "Workbook"
        notices.append(f"{label}: no GitHub/repository column was detected, so this sheet was skipped.")
        return [], notices, manual_candidates, imported_results, submission_cell_map

    submissions: List[Dict[str, str]] = []
    missing_source_count = 0
    normalized_source_label = clean_workbook_label(source_label)
    for row_number, row in dataframe.iterrows():
        trainee_name = str(row.get(name_column, "")).strip() if name_column else ""
        superset_id = normalize_superset_id(row.get(superset_column, "")) if superset_column else ""
        declared_technology = infer_declared_technology(sheet_name, row.get(tech_column, "") if tech_column else "")
        college = infer_college_name(row.get(college_column, "") if college_column else row.get("College", ""), source_label, sheet_name)
        for source_column, source_meta in source_cycle_metadata.items():
            week_label = source_meta.get("cycle", "")
            column_track = source_meta.get("track", normalize_evaluation_track(evaluation_track))
            submission_id = f"{normalized_source_label or 'Workbook'}::{sheet_name or 'Workbook'}::{row_number + 1}::{week_label}::{source_column}"
            source_value = str(row.get(source_column, "")).strip()
            submission_cell_map[submission_id] = {
                "sheet_name": sheet_name or "Workbook",
                "row_index": row_number,
                "column_name": source_column,
                "track": column_track,
                "cycle": week_label,
            }
            if is_blank_value(source_value) or source_value == "-":
                missing_source_count += 1
                manual_candidates.append(
                    {
                        "submission_id": submission_id,
                        "worksheet": sheet_name or "Workbook",
                        "row_number": str(row_number + 1),
                        "trainee": trainee_name or superset_id or f"Submission {row_number + 1}",
                        "superset_id": superset_id,
                        "declared_technology": declared_technology,
                        "college": college,
                        "source_file": normalized_source_label,
                        "assessment_week": week_label,
                        "evaluation_track": column_track,
                        "reason": f"Missing GitHub link in workbook row for {week_label}.",
                    }
                )
                continue
            submission_status = classify_submission_source_status(source_value)
            imported_score = parse_imported_workbook_score(source_value)
            if imported_score is not None:
                imported_results.append(
                    build_imported_scored_result(
                        trainee_name or superset_id or f"Submission {row_number + 1}",
                        imported_score,
                        "Fresher",
                        100,
                        "Resilient (Recommended)",
                        sheet_name or "Workbook",
                        declared_technology,
                        college,
                        superset_id,
                        week_label,
                        normalized_source_label,
                        column_track,
                        submission_id,
                    )
                )
                continue
            if is_imported_not_evaluated_value(source_value):
                imported_results.append(
                    build_imported_not_evaluated_result(
                        trainee_name or superset_id or f"Submission {row_number + 1}",
                        "Fresher",
                        100,
                        "Resilient (Recommended)",
                        sheet_name or "Workbook",
                        declared_technology,
                        college,
                        superset_id,
                        week_label,
                        normalized_source_label,
                        column_track,
                        submission_id,
                    )
                )
                continue
            if submission_status in SPECIAL_RESULT_STATUSES:
                special_result = build_special_status_result(
                    trainee_name or superset_id or f"Submission {row_number + 1}",
                    source_value,
                    submission_status,
                    "Fresher",
                    100,
                    "Resilient (Recommended)",
                    sheet_name or "Workbook",
                    declared_technology,
                    college,
                    superset_id,
                    week_label,
                    normalized_source_label,
                    column_track,
                )
                special_result.output["SubmissionID"] = submission_id
                special_result.output["ImportedFromWorkbook"] = True
                imported_results.append(special_result)
                continue
            submissions.append(
                {
                    "submission_id": submission_id,
                    "trainee": trainee_name or superset_id or f"Submission {row_number + 1}",
                    "source": source_value,
                    "worksheet": sheet_name or "Workbook",
                    "superset_id": superset_id,
                    "declared_technology": declared_technology,
                    "college": college,
                    "source_file": normalized_source_label,
                    "assessment_week": week_label,
                    "submission_status": submission_status or "Pending Evaluation",
                    "evaluation_track": column_track,
                }
            )
    if missing_source_count:
        label = f"Sheet '{sheet_name}'" if sheet_name else "Workbook"
        notices.append(f"{label}: skipped {missing_source_count} row(s) with no GitHub link.")
    return submissions, notices, manual_candidates, imported_results, submission_cell_map


def build_submission_template_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Superset ID": "QS-1001",
                "Candidate Name": "Student Name",
                "Assessment - Week 1": "https://github.com/username/repository-assessment-week-1",
                "Assessment - Week 2": "ABSENT",
                "Assessment - Week 3": "https://github.com/username/repository-assessment-week-3",
                "Assignment - First": "https://github.com/username/repository-assignment-first",
                "Technology / Framework": "Python / Pytest",
                "Notes": "Optional",
            }
        ]
    )


def load_submissions_from_excel(
    excel_source: Any,
    sheet_name: Optional[str] = None,
    evaluation_track: str = ASSESSMENT_SCORE_TRACK,
) -> Tuple[pd.DataFrame, List[Dict[str, str]], List[str], List[Dict[str, str]], List[EvaluationResult], List[Dict[str, Any]]]:
    preview_frames: List[pd.DataFrame] = []
    submissions: List[Dict[str, str]] = []
    notices: List[str] = []
    manual_candidates: List[Dict[str, str]] = []
    imported_results: List[EvaluationResult] = []
    workbook_contexts: List[Dict[str, Any]] = []
    excel_sources = excel_source if isinstance(excel_source, list) else [excel_source]

    for source_index, current_source in enumerate(excel_sources, start=1):
        source_label = getattr(current_source, "name", "") or (Path(str(current_source)).name if isinstance(current_source, (str, Path)) else f"Workbook {source_index}")
        if hasattr(current_source, "seek"):
            current_source.seek(0)
        raw_workbook = pd.read_excel(current_source, sheet_name=sheet_name or None, header=None)
        if isinstance(raw_workbook, pd.DataFrame):
            raw_sheets = {sheet_name or "Sheet1": raw_workbook}
        else:
            raw_sheets = raw_workbook
        workbook_context: Dict[str, Any] = {
            "source_label": clean_workbook_label(source_label),
            "file_name": Path(str(source_label)).name or f"task1_scored_workbook_{source_index}.xlsx",
            "sheets": [],
        }

        for current_sheet, raw_df in raw_sheets.items():
            prepared = prepare_submission_dataframe(raw_df, str(current_sheet))
            if prepared.empty:
                notices.append(f"Workbook '{clean_workbook_label(source_label)}' sheet '{current_sheet}' did not contain a usable trainee table.")
                continue
            export_frame = prepared.drop(columns=["Worksheet"], errors="ignore").copy()
            annotated_prepared = annotate_submission_preview(prepared, str(current_sheet), source_label)
            annotated_prepared["Evaluation Track"] = evaluation_track
            preview_frames.append(annotated_prepared)
            sheet_submissions, sheet_notices, sheet_manual_candidates, sheet_imported_results, sheet_cell_map = extract_submissions_from_dataframe(
                annotated_prepared,
                str(current_sheet),
                source_label,
                evaluation_track,
            )
            submissions.extend(sheet_submissions)
            notices.extend(sheet_notices)
            manual_candidates.extend(sheet_manual_candidates)
            imported_results.extend(sheet_imported_results)
            workbook_context["sheets"].append(
                {
                    "sheet_name": str(current_sheet),
                    "dataframe": export_frame,
                    "submission_cells": sheet_cell_map,
                }
            )

        if workbook_context["sheets"]:
            workbook_contexts.append(workbook_context)

    preview = pd.concat(preview_frames, ignore_index=True) if preview_frames else pd.DataFrame()
    return preview, submissions, notices, manual_candidates, imported_results, workbook_contexts


def dedupe_workbook_submissions(submissions: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen_pairs = set()
    deduped_submissions: List[Dict[str, str]] = []
    for item in submissions:
        dedupe_key = (
            normalize_superset_id(item.get("superset_id", "")),
            item["trainee"],
            item["source"],
            item.get("worksheet", ""),
            item.get("declared_technology", ""),
            item.get("college", ""),
            item.get("source_file", ""),
            item.get("assessment_week", ""),
            item.get("submission_status", ""),
            item.get("evaluation_track", ASSESSMENT_SCORE_TRACK),
        )
        if dedupe_key in seen_pairs:
            continue
        seen_pairs.add(dedupe_key)
        deduped_submissions.append(item)
    return deduped_submissions


def collect_task1_colleges(submissions: List[Dict[str, str]]) -> List[str]:
    return sorted({str(item.get("college", "")).strip() for item in submissions if str(item.get("college", "")).strip()})


def collect_task1_weeks(submissions: List[Dict[str, str]], selected_track: str = ALL_GITHUB_TRACKS_LABEL) -> List[str]:
    scoped_submissions = submissions if selected_track == ALL_GITHUB_TRACKS_LABEL else [
        item for item in submissions if normalize_evaluation_track(item.get("evaluation_track", "")) == normalize_evaluation_track(selected_track)
    ]
    return sort_week_labels(item.get("assessment_week", "") for item in scoped_submissions)


def collect_task1_tracks(submissions: List[Dict[str, str]]) -> List[str]:
    track_values = {
        normalize_evaluation_track(item.get("evaluation_track", ASSESSMENT_SCORE_TRACK))
        for item in submissions
    }
    return sorted(track_values, key=lambda value: (value != ASSESSMENT_SCORE_TRACK, value))


def group_workbook_submissions_by_college(submissions: List[Dict[str, str]]) -> List[Tuple[str, List[Dict[str, str]]]]:
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for item in submissions:
        college = str(item.get("college", "")).strip() or "Unspecified College"
        grouped.setdefault(college, []).append(item)
    return list(grouped.items())


def get_result_college(result: EvaluationResult) -> str:
    if getattr(result, "output", None) and isinstance(result.output, dict):
        college = clean_workbook_label(result.output.get("College", ""))
        if college:
            return college
    if getattr(result, "source_file", ""):
        inferred = clean_workbook_label(getattr(result, "source_file", ""), strip_extension=True)
        if inferred:
            return inferred
    if getattr(result, "worksheet", ""):
        inferred = infer_college_name("", "", result.worksheet)
        if inferred:
            return inferred
    return "Unspecified College"


def get_result_superset_id(result: EvaluationResult) -> str:
    direct_value = normalize_superset_id(getattr(result, "superset_id", ""))
    if direct_value:
        return direct_value
    if isinstance(result.output, dict):
        return normalize_superset_id(result.output.get("SupersetID", ""))
    return ""


def get_result_week(result: EvaluationResult) -> str:
    direct_value = str(getattr(result, "assessment_week", "")).strip()
    if direct_value:
        return direct_value
    if isinstance(result.output, dict):
        week_value = str(result.output.get("AssessmentWeek", "")).strip()
        if week_value:
            return week_value
    return "Week 1"


def normalize_evaluation_track(value: Any) -> str:
    normalized = normalize_column_name(value)
    if normalized in {"assignmentscore", "assignmentscores", "assignment", "dailyassignment", "dailyassignments"}:
        return ASSIGNMENT_SCORE_TRACK
    return ASSESSMENT_SCORE_TRACK


def get_result_track(result: EvaluationResult) -> str:
    direct_value = normalize_evaluation_track(getattr(result, "evaluation_track", ""))
    if direct_value:
        return direct_value
    if isinstance(result.output, dict):
        return normalize_evaluation_track(result.output.get("EvaluationTrack", ""))
    return ASSESSMENT_SCORE_TRACK


def is_special_status_result(result: EvaluationResult) -> bool:
    return result.status in SPECIAL_RESULT_STATUSES


def is_retryable_result(result: EvaluationResult) -> bool:
    return result.status == "Not Evaluated"


def filter_results_by_assessment_week(results: List[EvaluationResult], selected_week: str) -> List[EvaluationResult]:
    if is_all_cycle_label(selected_week):
        return results
    return [item for item in results if get_result_week(item) == selected_week]


def filter_results_by_track(results: List[EvaluationResult], selected_track: str) -> List[EvaluationResult]:
    if selected_track == ALL_GITHUB_TRACKS_LABEL:
        return results
    return [item for item in results if get_result_track(item) == selected_track]


def build_result_candidate_key(result: EvaluationResult) -> Tuple[str, str, str, str]:
    superset_id = get_result_superset_id(result)
    track = get_result_track(result)
    if superset_id:
        return (superset_id, get_result_college(result), "", track)
    return ("", get_result_college(result), result.trainee.strip().lower(), track)


def group_results_by_candidate(results: List[EvaluationResult]) -> Dict[Tuple[str, str, str], List[EvaluationResult]]:
    grouped: Dict[Tuple[str, str, str], List[EvaluationResult]] = {}
    for item in results:
        grouped.setdefault(build_result_candidate_key(item), []).append(item)
    return grouped


def filter_exception_results(results: List[EvaluationResult], selected_filter: str) -> List[EvaluationResult]:
    if selected_filter == "Not Evaluated":
        matched = [item for item in results if is_retryable_result(item)]
    elif selected_filter in SPECIAL_RESULT_STATUSES:
        matched = [item for item in results if item.status == selected_filter]
    else:
        return []
    return sort_evaluation_results(matched)


def count_exception_candidates(results: List[EvaluationResult], selected_filter: str) -> int:
    matched = filter_exception_results(results, selected_filter)
    return len({build_result_candidate_key(item) for item in matched})


def aggregate_candidate_results(results: List[EvaluationResult], selected_week: str = ALL_WEEKS_LABEL) -> List[EvaluationResult]:
    scoped_results = filter_results_by_assessment_week(results, selected_week)
    grouped = group_results_by_candidate(scoped_results)

    aggregated: List[EvaluationResult] = []
    for _, candidate_results in grouped.items():
        evaluated_results = [item for item in candidate_results if is_evaluated_result(item)]
        if evaluated_results:
            template = sorted(evaluated_results, key=lambda item: (assessment_week_sort_key(get_result_week(item)), item.repository.lower()))[0]
            average_score = round(sum(item.score for item in evaluated_results) / len(evaluated_results))
            averaged_breakdown = {
                bucket: round(sum(item.score_breakdown.get(bucket, 0.0) for item in evaluated_results) / len(evaluated_results), 2)
                for bucket in SCORE_WEIGHTS
            }
            aggregated.append(
                EvaluationResult(
                    trainee=template.trainee,
                    repository=template.repository if not is_all_cycle_label(selected_week) else f"{len(evaluated_results)} cycle(s) evaluated",
                    primary_file=template.primary_file,
                    files_analyzed=int(round(sum(item.files_analyzed for item in evaluated_results) / len(evaluated_results))),
                    score=average_score,
                    classification=widget_score_band(average_score),
                    experience_profile=template.experience_profile,
                    strengths=template.strengths,
                    weaknesses=template.weaknesses,
                    security=template.security,
                    primary_gap=template.primary_gap,
                    static_analysis=template.static_analysis,
                    structural_analysis=template.structural_analysis,
                    ai_review=template.ai_review,
                    file_tree=template.file_tree,
                    primary_snapshot=template.primary_snapshot,
                    reviewed_files=template.reviewed_files,
                    score_breakdown=averaged_breakdown,
                    review_strictness=template.review_strictness,
                    status="Evaluated",
                    issue_type="",
                    issue_message="",
                    output={
                        **(template.output or {}),
                        "AssessmentWeek": selected_week if not is_all_cycle_label(selected_week) else default_cycle_label_for_track(get_result_track(template)),
                        "SupersetID": get_result_superset_id(template),
                        "College": get_result_college(template),
                        "EvaluationTrack": get_result_track(template),
                    },
                    worksheet=template.worksheet,
                    declared_technology=template.declared_technology,
                    languages_tested=template.languages_tested,
                    superset_id=get_result_superset_id(template),
                    assessment_week=selected_week if not is_all_cycle_label(selected_week) else default_cycle_label_for_track(get_result_track(template)),
                    source_file=template.source_file,
                    evaluation_track=get_result_track(template),
                )
            )
            continue

        status_priority = ["Dropout", "No Show", "Absent", "Not Evaluated"]
        template = candidate_results[0]
        selected_status = next((status for status in status_priority if any(item.status == status for item in candidate_results)), "Not Evaluated")
        selected_issue = next((item for item in candidate_results if item.status == "Not Evaluated"), template)
        aggregated.append(
            EvaluationResult(
                trainee=template.trainee,
                repository=template.repository,
                primary_file="",
                files_analyzed=0,
                score=0,
                classification=selected_status if selected_status in SPECIAL_RESULT_STATUSES else NOT_EVALUATED_CLASSIFICATION,
                experience_profile=template.experience_profile,
                strengths=[],
                weaknesses=[] if selected_status in SPECIAL_RESULT_STATUSES else selected_issue.weaknesses,
                security=[],
                primary_gap="" if selected_status in SPECIAL_RESULT_STATUSES else selected_issue.primary_gap,
                static_analysis=StaticAnalysisResult(),
                structural_analysis=StructuralAnalysisResult(),
                ai_review=AIReviewResult(),
                file_tree="",
                primary_snapshot="",
                reviewed_files=[],
                score_breakdown={},
                review_strictness=template.review_strictness,
                status=selected_status,
                issue_type="" if selected_status in SPECIAL_RESULT_STATUSES else selected_issue.issue_type,
                issue_message="" if selected_status in SPECIAL_RESULT_STATUSES else selected_issue.issue_message,
                output={
                    **(template.output or {}),
                    "AssessmentWeek": selected_week if not is_all_cycle_label(selected_week) else default_cycle_label_for_track(get_result_track(template)),
                    "SupersetID": get_result_superset_id(template),
                    "College": get_result_college(template),
                    "Status": selected_status,
                    "EvaluationTrack": get_result_track(template),
                },
                worksheet=template.worksheet,
                declared_technology=template.declared_technology,
                languages_tested="",
                superset_id=get_result_superset_id(template),
                assessment_week=selected_week if not is_all_cycle_label(selected_week) else default_cycle_label_for_track(get_result_track(template)),
                source_file=template.source_file,
                evaluation_track=get_result_track(template),
            )
        )
    return sort_evaluation_results(aggregated)


def build_candidate_week_summary(candidate_results: List[EvaluationResult]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for week_label in sort_week_labels(get_result_week(item) for item in candidate_results):
        aggregated_results = aggregate_candidate_results(candidate_results, week_label)
        if not aggregated_results:
            continue
        item = aggregated_results[0]
        rows.append(
            {
                "GitHub Cycle": week_label,
                "Status": item.status,
                "Score": item.score if is_evaluated_result(item) else None,
                "Grade": item.classification,
                "Repository": item.repository or "No repository source captured.",
                "Issue Type": item.issue_type,
                "Issue": item.issue_message,
            }
        )
    return pd.DataFrame(rows)


def filter_results_by_college(results: List[EvaluationResult], selected_college: str) -> List[EvaluationResult]:
    if selected_college == "All Colleges":
        return results
    return [item for item in results if get_result_college(item) == selected_college]


def filter_workbook_preview_by_college(workbook_preview: Optional[pd.DataFrame], selected_college: str) -> Optional[pd.DataFrame]:
    if workbook_preview is None or workbook_preview.empty or selected_college == "All Colleges" or "College" not in workbook_preview.columns:
        return workbook_preview
    return workbook_preview[workbook_preview["College"].astype(str).str.strip() == selected_college].copy()


def filter_workbook_preview_by_track(workbook_preview: Optional[pd.DataFrame], selected_track: str) -> Optional[pd.DataFrame]:
    return workbook_preview


def build_bucket_table(score_breakdown: Dict[str, float]) -> pd.DataFrame:
    rows = []
    for category, max_points in SCORE_WEIGHTS.items():
        raw_score = float(score_breakdown.get(category, 0.0))
        awarded = round((raw_score / 100.0) * max_points, 2)
        rows.append(
            {
                "Bucket": category,
                "Rating": bucket_rating(raw_score),
                "Raw Score": round(raw_score, 2),
                "Score Awarded": awarded,
                "Max Score": max_points,
                "Display": f"{awarded:.2f} / {max_points}",
            }
        )
    return pd.DataFrame(rows)


def inject_dashboard_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #f4f8fc 0%, #eef3f8 100%);
        }
        .portal-banner {
            background: linear-gradient(135deg, #0f4c81 0%, #1f6fb2 100%);
            padding: 1.4rem 1.6rem;
            border-radius: 18px;
            color: white;
            margin-bottom: 1.2rem;
            box-shadow: 0 12px 28px rgba(15, 76, 129, 0.18);
        }
        .qspiders-banner {
            background: linear-gradient(135deg, #fff8f0 0%, #ffffff 55%, #eef6fb 100%);
            padding: 1rem 1.3rem;
            border-radius: 18px;
            color: #17324d;
            margin-bottom: 1rem;
            box-shadow: 0 12px 28px rgba(29, 59, 87, 0.08);
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            border: 1px solid rgba(15, 76, 129, 0.08);
        }
        .qspiders-brand-wrap {
            display: flex;
            align-items: center;
            gap: 0.95rem;
        }
        .qspiders-logo {
            width: 56px;
            height: auto;
            flex-shrink: 0;
        }
        .qspiders-brand {
            font-size: 1.45rem;
            font-weight: 800;
            letter-spacing: 0.02em;
            margin: 0;
            color: #111827;
        }
        .qspiders-note {
            font-size: 0.94rem;
            color: #5b6f82;
            margin-top: 0.2rem;
        }
        .qspiders-chip {
            border: 1px solid rgba(239, 153, 74, 0.28);
            background: rgba(239, 153, 74, 0.12);
            border-radius: 999px;
            padding: 0.42rem 0.9rem;
            font-size: 0.84rem;
            font-weight: 700;
            white-space: nowrap;
            color: #d97706;
        }
        .portal-banner h1 {
            margin: 0;
            font-size: 2rem;
            font-weight: 700;
        }
        .portal-banner p {
            margin: 0.35rem 0 0;
            font-size: 0.98rem;
            opacity: 0.92;
        }
        .section-card {
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid rgba(15, 76, 129, 0.08);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            box-shadow: 0 10px 26px rgba(29, 59, 87, 0.08);
            margin-bottom: 1rem;
        }
        .section-title {
            font-size: 1.08rem;
            font-weight: 700;
            color: #17324d;
            margin-bottom: 0.25rem;
        }
        .section-subtitle {
            font-size: 0.92rem;
            color: #5b6f82;
            margin-bottom: 0.8rem;
        }
        div[data-testid="stMetric"] {
            background: linear-gradient(180deg, #ffffff 0%, #f7fbff 100%);
            border: 1px solid rgba(15, 76, 129, 0.10);
            border-radius: 16px;
            padding: 0.9rem 0.8rem;
            box-shadow: 0 8px 20px rgba(29, 59, 87, 0.06);
        }
        div[data-testid="stExpander"] {
            border: 1px solid rgba(15, 76, 129, 0.08);
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.85);
        }
        div[data-testid="stTabs"] button {
            font-weight: 600;
        }
        .small-note {
            color: #62788f;
            font-size: 0.88rem;
        }
        div[data-testid="stRadio"] > div {
            gap: 0.75rem;
        }
        [data-testid="collapsedControl"] {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header_banner() -> None:
    st.markdown(
        """
        <div class="portal-banner">
            <h1>Training Code Evaluation Portal</h1>
            <p>AI-assisted repository review for trainee benchmarking, deep code audit, and cohort analytics.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_qspiders_banner() -> None:
    logo_path = Path(__file__).resolve().parent / "assets" / "qspiders_logo.svg"
    logo_data = ""
    if logo_path.exists():
        logo_data = base64.b64encode(logo_path.read_bytes()).decode("utf-8")

    st.markdown(
        f"""
        <div class="qspiders-banner">
            <div class="qspiders-brand-wrap">
                <img class="qspiders-logo" src="data:image/svg+xml;base64,{logo_data}" alt="QSpiders logo" />
                <div>
                    <div class="qspiders-brand">QSpiders Enterprise Analytics</div>
                    <div class="qspiders-note">Unified developer training intelligence across code evaluation, assessments, attendance, and executive correlation.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_intro(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="section-card">
            <div class="section-title">{title}</div>
            <div class="section-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chart_note(message: str) -> None:
    st.caption(message)


def score_band_metadata(experience_profile: str) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
    thresholds = scoring_profile(experience_profile)["thresholds"]
    bands = [
        {
            "label": "Retest Required",
            "lower": 0,
            "upper": thresholds["Needs Attention"],
            "color": "rgba(239, 68, 68, 0.10)",
        },
        {
            "label": "Needs Attention",
            "lower": thresholds["Needs Attention"],
            "upper": thresholds["Good"],
            "color": "rgba(245, 158, 11, 0.10)",
        },
        {
            "label": "Good",
            "lower": thresholds["Good"],
            "upper": thresholds["Excellent"],
            "color": "rgba(37, 99, 235, 0.08)",
        },
        {
            "label": "Excellent",
            "lower": thresholds["Excellent"],
            "upper": 100,
            "color": "rgba(16, 185, 129, 0.08)",
        },
    ]
    return thresholds, bands


def add_score_bands(fig: Any, experience_profile: str) -> None:
    thresholds, bands = score_band_metadata(experience_profile)
    for band in bands:
        fig.add_hrect(y0=band["lower"], y1=band["upper"], fillcolor=band["color"], line_width=0, layer="below")
    fig.add_hline(y=thresholds["Needs Attention"], line_dash="dot", line_color="#dc2626")
    fig.add_hline(y=thresholds["Good"], line_dash="dot", line_color="#d97706")
    fig.add_hline(y=thresholds["Excellent"], line_dash="dot", line_color="#059669")


def render_score_guide(
    results: List[EvaluationResult],
    needs_attention_label: str = "Scope for Improvement",
    retest_required_label: str = "Reassessment Required",
) -> None:
    experience_profile = next((item.experience_profile for item in results if is_evaluated_result(item)), st.session_state.get("task1_experience_profile", "Fresher"))
    st.caption(
        "Grade interpretation "
        f"({experience_profile}): Excellent 90-100, "
        "Good 80-89, "
        f"{needs_attention_label} 70-79, "
        f"{retest_required_label} 0-69. "
        "The strictness slider changes penalty strength; it does not replace the score bands."
    )


def workbook_preview_term(label: str) -> str:
    return {
        "Needs Attention": "Scope for Improvement",
        "Retest Required": "Reassessment Required",
    }.get(str(label or ""), str(label or ""))


def widget_score_band(score: float) -> str:
    if score >= 90:
        return "Excellent"
    if score >= 80:
        return "Good"
    if score >= 70:
        return "Needs Attention"
    return "Retest Required"


def merge_workbook_overrides(submissions: List[Dict[str, str]]) -> List[Dict[str, str]]:
    overrides = list(st.session_state.get("task1_manual_overrides", {}).values())
    return submissions + overrides


def render_module_selector() -> str:
    render_section_intro(
        "Portal Modules",
        "Choose any of the available analytics workspaces from the same landing page. Task-1 keeps its current evaluation flow exactly as it is.",
    )
    return st.radio(
        "Portal Module",
        PORTAL_MODULES,
        horizontal=True,
        label_visibility="collapsed",
        key="portal_module_selector",
    )


def run_pylint(file_paths: List[str]) -> Tuple[float, List[Dict[str, Any]], List[str]]:
    if not file_paths:
        return 0.0, [], []
    if PylintRun is None or JSONReporter is None:
        return 0.0, [], ["pylint is not installed."]

    output = StringIO()
    reporter = JSONReporter(output)
    args = ["--reports=n", *sorted(file_paths)]

    try:
        result = PylintRun(args, reporter=reporter, exit=False)
    except TypeError:
        result = PylintRun(args, reporter=reporter, do_exit=False)
    except Exception as exc:
        return 0.0, [], [f"pylint failed: {exc}"]

    try:
        messages = json.loads(output.getvalue() or "[]")
    except json.JSONDecodeError:
        messages = []

    score_stats = getattr(getattr(result, "linter", None), "stats", None)
    global_note = float(getattr(score_stats, "global_note", 0.0) or 0.0) if score_stats else 0.0

    if score_stats and global_note == 0.0:
        statements = max(float(getattr(score_stats, "statement", 0) or 0), 1.0)
        error_count = float(getattr(score_stats, "error", 0) or 0)
        warning_count = float(getattr(score_stats, "warning", 0) or 0)
        refactor_count = float(getattr(score_stats, "refactor", 0) or 0)
        convention_count = float(getattr(score_stats, "convention", 0) or 0)
        global_note = clamp(
            10.0 - ((5.0 * error_count + warning_count + refactor_count + convention_count) / statements) * 10.0,
            0.0,
            10.0,
        )

    return global_note, messages, []


def run_radon_analysis(files: List[RepositoryFile]) -> Tuple[float, float, List[str]]:
    if cc_visit is None or mi_visit is None or radon_raw_analyze is None:
        return 0.0, 0.0, ["radon is not installed."]

    complexities: List[float] = []
    maintainability: List[float] = []
    notes: List[str] = []

    for repo_file in files:
        if repo_file.language != "Python":
            continue
        try:
            blocks = cc_visit(repo_file.content)
            if blocks:
                complexities.extend(block.complexity for block in blocks)
            maintainability.append(float(mi_visit(repo_file.content, multi=True)))
            radon_raw_analyze(repo_file.content)
        except Exception as exc:
            notes.append(f"radon failed for {repo_file.path}: {exc}")

    average_complexity = sum(complexities) / len(complexities) if complexities else 0.0
    average_mi = sum(maintainability) / len(maintainability) if maintainability else 0.0
    return average_complexity, average_mi, notes


def run_bandit_scan(tmp_dir: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    if bandit_config is None or bandit_manager is None:
        return [], ["bandit is not installed."]

    try:
        conf = bandit_config.BanditConfig()
        manager = bandit_manager.BanditManager(conf, "file", False)
        manager.discover_files([tmp_dir], True)
        manager.run_tests()
        issues = []
        for issue in manager.get_issue_list():
            issues.append(
                {
                    "filename": issue.fname,
                    "line_number": issue.lineno,
                    "severity": issue.severity,
                    "confidence": issue.confidence,
                    "test_id": issue.test_id,
                    "text": issue.text,
                }
            )
        return issues, []
    except Exception as exc:
        return [], [f"bandit failed: {exc}"]


def count_prefixed_comment_lines(language: str, lines: List[str]) -> int:
    if language in {"Java", "JavaScript", "TypeScript", "C#"}:
        prefixes = ("//", "/*", "*")
    else:
        prefixes = ("#",)
    return sum(1 for line in lines if line.lstrip().startswith(prefixes))


def resolve_command(*candidates: str) -> Optional[str]:
    cache_key = tuple(candidates)
    with _cache_lock:
        if cache_key in _command_cache:
            return _command_cache[cache_key]
    resolved = next((shutil.which(candidate) for candidate in candidates if shutil.which(candidate)), None)
    with _cache_lock:
        _command_cache[cache_key] = resolved
    return resolved


def generic_imports(language: str, content: str) -> set[str]:
    if language in {"JavaScript", "TypeScript"}:
        imports = set(re.findall(r"""from\s+['"]([^'"]+)['"]""", content))
        imports.update(re.findall(r"""require\(\s*['"]([^'"]+)['"]\s*\)""", content))
        return {item.split("/")[0] for item in imports}
    if language == "C#":
        imports = set(re.findall(r"\busing\s+([A-Za-z0-9_.]+)\s*;", content))
        return {item.split(".")[0] for item in imports}
    if language == "Java":
        imports = set(re.findall(r"\bimport\s+([A-Za-z0-9_.]+)", content))
        return {item.split(".")[0] for item in imports}
    if language == "Robot Framework":
        imports = set(re.findall(r"(?im)^\s*Library\s+([A-Za-z0-9_.]+)", content))
        return {item.split(".")[0] for item in imports}
    return set()


def analyze_generic_structure(language: str, content: str) -> Dict[str, Any]:
    if language == "Java":
        return analyze_java_structure(content)

    if language == "Robot Framework":
        test_match = re.search(r"(?is)\*\*\*\s*test cases\s*\*\*\*(.*?)(\*\*\*|$)", content or "")
        keyword_match = re.search(r"(?is)\*\*\*\s*keywords\s*\*\*\*(.*?)(\*\*\*|$)", content or "")
        test_cases = len(re.findall(r"(?im)^\S.*$", test_match.group(1) if test_match else ""))
        keywords = len(re.findall(r"(?im)^\S.*$", keyword_match.group(1) if keyword_match else ""))
        return {
            "class_count": 0,
            "function_count": test_cases + keywords,
            "imports": generic_imports(language, content),
            "max_depth": 1 if test_cases or keywords else 0,
        }

    if language in {"JavaScript", "TypeScript"}:
        class_count = len(re.findall(r"\bclass\s+[A-Za-z_]\w*", content))
        function_patterns = [
            r"\bfunction\s+[A-Za-z_]\w*\s*\(",
            r"\b(?:const|let|var)\s+[A-Za-z_]\w*\s*=\s*(?:async\s*)?\([^)]*\)\s*=>",
            r"\b[A-Za-z_]\w*\s*:\s*(?:async\s*)?\([^)]*\)\s*=>",
        ]
        function_count = sum(len(re.findall(pattern, content)) for pattern in function_patterns)
    elif language == "C#":
        class_count = len(re.findall(r"\b(class|record|interface)\s+[A-Za-z_]\w*", content))
        function_count = len(
            re.findall(
                r"\b(public|private|protected|internal)\s+(static\s+)?(async\s+)?[\w<>\[\],?]+\s+[A-Za-z_]\w*\s*\([^;{}]*\)\s*\{",
                content,
            )
        )
    else:
        class_count = 0
        function_count = 0

    depth = 0
    max_depth = 0
    for char in content:
        if char == "{":
            depth += 1
            max_depth = max(max_depth, depth)
        elif char == "}":
            depth = max(0, depth - 1)
    return {
        "class_count": class_count,
        "function_count": function_count,
        "imports": generic_imports(language, content),
        "max_depth": max_depth,
    }


def run_eslint_analysis(files: List[RepositoryFile]) -> Dict[str, Any]:
    js_files = [item for item in files if item.language == "JavaScript"]
    if not js_files:
        return {"executed": False, "notes": []}

    npx_cmd = resolve_command("npx.cmd", "npx")
    if not npx_cmd:
        return {"executed": False, "notes": ["npx is not available, so JavaScript lint analysis was skipped."]}

    with tempfile.TemporaryDirectory(prefix="repo_eval_eslint_") as tmp_dir:
        root = Path(tmp_dir) / "workspace"
        file_paths: List[str] = []
        for repo_file in js_files:
            destination = root / Path(repo_file.path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(repo_file.content, encoding="utf-8")
            file_paths.append(str(destination))

        command = [
            npx_cmd,
            "-y",
            "eslint@9.38.0",
            "--no-config-lookup",
            "--format",
            "json",
            "--rule",
            "no-undef:error",
            "--rule",
            "no-unreachable:error",
            "--rule",
            "no-unused-vars:warn",
            "--rule",
            "no-eval:error",
            "--rule",
            "no-implied-eval:error",
            "--rule",
            "no-new-func:error",
            "--rule",
            "complexity:[\"warn\",12]",
            "--rule",
            "max-depth:[\"warn\",4]",
            "--global",
            "console,window,document,module,require,process,__dirname,__filename,setTimeout,clearTimeout,setInterval,clearInterval,describe,it,test,expect,beforeEach,afterEach,page,browser,context",
            *file_paths,
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "CI": "1"},
        )

    if completed.returncode not in {0, 1}:
        stderr = (completed.stderr or completed.stdout or "").strip()
        return {"executed": False, "notes": [f"eslint failed: {stderr or 'unexpected exit code'}"]}

    try:
        payload = json.loads(completed.stdout or "[]")
    except json.JSONDecodeError:
        return {"executed": False, "notes": ["eslint returned unreadable output and was ignored."]}

    error_count = 0
    warning_count = 0
    complexity_warnings = 0
    depth_warnings = 0
    security_issues: List[Dict[str, Any]] = []
    security_rules = {
        "no-eval": "Dynamic eval usage detected in JavaScript.",
        "no-implied-eval": "Implied eval usage detected in JavaScript.",
        "no-new-func": "Dynamic Function construction detected in JavaScript.",
    }
    for file_result in payload:
        error_count += int(file_result.get("errorCount", 0) or 0)
        warning_count += int(file_result.get("warningCount", 0) or 0)
        for message in file_result.get("messages", []):
            rule_id = str(message.get("ruleId") or "")
            if rule_id == "complexity":
                complexity_warnings += 1
            elif rule_id == "max-depth":
                depth_warnings += 1
            if rule_id in security_rules:
                security_issues.append(
                    {
                        "filename": file_result.get("filePath", ""),
                        "line_number": message.get("line", 1),
                        "severity": "MEDIUM",
                        "confidence": "MEDIUM",
                        "test_id": rule_id,
                        "text": security_rules[rule_id],
                    }
                )

    file_count = max(len(js_files), 1)
    quality_proxy = score_from_penalty(error_count * 8.0 + warning_count * 2.0 + complexity_warnings * 2.5 + depth_warnings * 2.5)
    estimated_complexity = round(2.0 + (complexity_warnings / file_count) * 3.0 + (depth_warnings / file_count) * 1.5, 2)
    maintainability = clamp(
        quality_proxy * 0.72
        + score_from_penalty(max(complexity_warnings - file_count, 0) * 3.0) * 0.18
        + score_from_penalty(max(depth_warnings - file_count, 0) * 4.0) * 0.10,
        0.0,
        100.0,
    )
    return {
        "executed": True,
        "quality_proxy": quality_proxy,
        "estimated_complexity": estimated_complexity,
        "maintainability": maintainability,
        "error_count": error_count,
        "warning_count": warning_count,
        "complexity_warnings": complexity_warnings,
        "depth_warnings": depth_warnings,
        "security_issues": security_issues,
        "notes": [
            f"eslint reviewed {len(js_files)} JavaScript file(s) and reported {error_count} error(s) plus {warning_count} warning(s)."
        ],
    }


def analyze_non_python_file(repo_file: RepositoryFile) -> Dict[str, Any]:
    lines = repo_file.content.splitlines()
    stripped_lines = [line.rstrip() for line in lines if line.strip()]
    comment_lines = count_prefixed_comment_lines(repo_file.language, stripped_lines)
    comment_ratio = comment_lines / max(len(stripped_lines), 1)
    long_line_ratio = sum(1 for line in stripped_lines if len(line) > 120) / max(len(stripped_lines), 1)
    structure = analyze_generic_structure(repo_file.language, repo_file.content)
    decision_points = len(
        re.findall(
            r"\b(if|else\s+if|switch|case|for|while|catch|when|elif|except|try|foreach)\b|&&|\|\|",
            repo_file.content,
            flags=re.IGNORECASE,
        )
    )
    function_basis = max(structure["function_count"], 1)
    estimated_complexity = round(1.0 + (decision_points / function_basis), 2)
    maintainability = clamp(
        100.0
        - max(estimated_complexity - 4.0, 0.0) * 8.0
        - max(structure["max_depth"] - 3, 0) * 7.0
        - long_line_ratio * 30.0
        + min(comment_ratio * 25.0, 10.0),
        0.0,
        100.0,
    )
    quality_proxy = clamp(
        maintainability * 0.55
        + score_from_penalty(max(estimated_complexity - 6.0, 0.0) * 7.0) * 0.25
        + score_from_penalty(long_line_ratio * 100.0 * 0.45) * 0.20,
        0.0,
        100.0,
    )
    security_patterns = [
        (r"\beval\s*\(", "Dynamic evaluation detected and should be reviewed."),
        (r"\bnew\s+Function\s*\(", "Dynamic function construction detected."),
        (r"\bProcess\.Start\s*\(", "Process execution detected and should be validated."),
        (r"\bRuntime\.getRuntime\(\)\.exec\s*\(", "Command execution detected in Java code."),
        (r"\bchild_process\b|\bexec\s*\(|\bspawn\s*\(", "External process execution detected in JavaScript/TypeScript."),
        (r"\binnerHTML\s*=", "Direct HTML injection pattern detected."),
        (r"\bBinaryFormatter\b", "BinaryFormatter usage detected and should be avoided."),
    ]
    security_issues = []
    for pattern, message in security_patterns:
        if re.search(pattern, repo_file.content, flags=re.IGNORECASE):
            security_issues.append(
                {
                    "filename": repo_file.path,
                    "line_number": 1,
                    "severity": "MEDIUM",
                    "confidence": "MEDIUM",
                    "test_id": "GENERIC",
                    "text": message,
                }
            )
    return {
        "quality_proxy": quality_proxy,
        "estimated_complexity": estimated_complexity,
        "maintainability": maintainability,
        "documentation_issue": 1 if comment_ratio < 0.05 else 0,
        "security_issues": security_issues,
        "imports": structure["imports"],
        "function_count": structure["function_count"],
        "class_count": structure["class_count"],
        "max_depth": structure["max_depth"],
        "comment_ratio": comment_ratio,
    }


def analyze_static(repository: Dict[str, Any]) -> StaticAnalysisResult:
    files: List[RepositoryFile] = repository["files"]
    python_files = sorted((item for item in files if item.language == "Python"), key=lambda item: item.path)
    non_python_files = [item for item in files if item.language != "Python"]
    result = StaticAnalysisResult()

    with tempfile.TemporaryDirectory(prefix="repo_eval_") as tmp_dir:
        analysis_root = Path(tmp_dir) / "workspace"
        file_paths: List[str] = []
        for repo_file in python_files:
            analysis_relative = Path(repo_file.path)
            if analysis_relative.suffix.lower() == ".ipynb":
                analysis_relative = analysis_relative.with_suffix(".notebook.py")
            destination = analysis_root / analysis_relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(repo_file.content, encoding="utf-8")
            file_paths.append(str(destination))

        pylint_score, pylint_messages, pylint_notes = run_pylint(file_paths)
        result.pylint_score = pylint_score
        result.pylint_messages = pylint_messages
        result.notes.extend(pylint_notes)

        for message in pylint_messages:
            message_type = message.get("type", "")
            symbol = message.get("symbol", "")
            if symbol in {"invalid-name", "disallowed-name"}:
                result.naming_issues += 1
            if message_type == "error":
                result.error_count += 1
            if message_type == "fatal":
                result.fatal_count += 1
            if "docstring" in symbol:
                result.documentation_issues += 1

        average_complexity, maintainability_index, radon_notes = run_radon_analysis(python_files)
        result.average_complexity = average_complexity
        result.maintainability_index = maintainability_index
        result.notes.extend(radon_notes)

        security_issues, bandit_notes = run_bandit_scan(tmp_dir)
        result.security_issues = security_issues
        result.notes.extend(bandit_notes)

    if non_python_files:
        heuristic_scores: List[float] = []
        heuristic_complexities: List[float] = []
        heuristic_maintainability: List[float] = []
        heuristic_security: List[Dict[str, Any]] = []
        heuristic_doc_issues = 0
        detected_languages = sorted({item.language for item in non_python_files})
        grouped_metrics: Dict[str, List[Dict[str, Any]]] = {}
        grouped_files: Dict[str, List[RepositoryFile]] = {}
        for repo_file in non_python_files:
            metrics = analyze_non_python_file(repo_file)
            grouped_metrics.setdefault(repo_file.language, []).append(metrics)
            grouped_files.setdefault(repo_file.language, []).append(repo_file)

        eslint_result = run_eslint_analysis(grouped_files.get("JavaScript", []))
        if eslint_result.get("notes"):
            result.notes.extend(eslint_result["notes"])
        if eslint_result.get("executed"):
            result.error_count += int(eslint_result.get("error_count", 0) or 0)

        for language, metrics_list in grouped_metrics.items():
            language_count = len(metrics_list)
            avg_quality = sum(metric["quality_proxy"] for metric in metrics_list) / max(language_count, 1)
            avg_complexity = sum(metric["estimated_complexity"] for metric in metrics_list) / max(language_count, 1)
            avg_maintainability = sum(metric["maintainability"] for metric in metrics_list) / max(language_count, 1)
            security_findings = [issue for metric in metrics_list for issue in metric["security_issues"]]
            heuristic_doc_issues += sum(metric["documentation_issue"] for metric in metrics_list)

            if language == "JavaScript" and eslint_result.get("executed"):
                avg_quality = avg_quality * 0.55 + float(eslint_result["quality_proxy"]) * 0.45
                avg_complexity = avg_complexity * 0.7 + float(eslint_result["estimated_complexity"]) * 0.3
                avg_maintainability = avg_maintainability * 0.65 + float(eslint_result["maintainability"]) * 0.35
                security_findings.extend(eslint_result["security_issues"])

            heuristic_scores.extend([avg_quality] * language_count)
            heuristic_complexities.extend([avg_complexity] * language_count)
            heuristic_maintainability.extend([avg_maintainability] * language_count)
            heuristic_security.extend(security_findings)

        if heuristic_scores:
            if python_files:
                python_weight = len(python_files)
                other_weight = len(non_python_files)
                blended_quality = ((result.pylint_score * 10.0) * python_weight + sum(heuristic_scores)) / max(python_weight + other_weight, 1)
                result.pylint_score = round(blended_quality / 10.0, 2)
                result.average_complexity = (
                    (result.average_complexity * python_weight + sum(heuristic_complexities)) / max(python_weight + other_weight, 1)
                )
                result.maintainability_index = (
                    (result.maintainability_index * python_weight + sum(heuristic_maintainability)) / max(python_weight + other_weight, 1)
                )
            else:
                result.pylint_score = round(sum(heuristic_scores) / len(heuristic_scores) / 10.0, 2)
                result.average_complexity = sum(heuristic_complexities) / len(heuristic_complexities)
                result.maintainability_index = sum(heuristic_maintainability) / len(heuristic_maintainability)
            result.documentation_issues += heuristic_doc_issues
            result.security_issues.extend(heuristic_security)
            heuristic_only_languages = list(detected_languages)
            if eslint_result.get("executed") and "JavaScript" in heuristic_only_languages:
                heuristic_only_languages.remove("JavaScript")
                result.notes.append("JavaScript files received ESLint review plus structural heuristics.")
            if heuristic_only_languages:
                result.notes.append(
                    "Generic static analysis heuristics were used for "
                    + ", ".join(heuristic_only_languages)
                    + " because language-specific analyzers are not yet wired into this deployment."
                )

    if not python_files:
        result.notes.append("No Python files were available for pylint/radon/bandit analysis.")

    return result


class PythonStructureVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.function_count = 0
        self.class_count = 0
        self.method_count = 0
        self.max_nesting_depth = 0
        self._current_depth = 0
        self.imports: set[str] = set()
        self.docstring_nodes = 0
        self.total_doc_targets = 0
        self._class_stack = 0

    def generic_visit(self, node: ast.AST) -> None:
        nesting_nodes = [ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith]
        match_node = getattr(ast, "Match", None)
        if match_node is not None:
            nesting_nodes.append(match_node)
        increased = isinstance(node, tuple(nesting_nodes))
        if increased:
            self._current_depth += 1
            self.max_nesting_depth = max(self.max_nesting_depth, self._current_depth)
        super().generic_visit(node)
        if increased:
            self._current_depth -= 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.function_count += 1
        if self._class_stack:
            self.method_count += 1
        self.total_doc_targets += 1
        if ast.get_docstring(node):
            self.docstring_nodes += 1
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.class_count += 1
        self.total_doc_targets += 1
        if ast.get_docstring(node):
            self.docstring_nodes += 1
        self._class_stack += 1
        self.generic_visit(node)
        self._class_stack -= 1

    def visit_Module(self, node: ast.Module) -> None:
        self.total_doc_targets += 1
        if ast.get_docstring(node):
            self.docstring_nodes += 1
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.add(alias.name.split(".")[0])
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            self.imports.add(node.module.split(".")[0])
        self.generic_visit(node)


def analyze_java_structure(content: str) -> Dict[str, Any]:
    class_count = len(re.findall(r"\bclass\s+[A-Za-z_]\w*", content))
    function_count = len(re.findall(r"(public|private|protected)?\s+(static\s+)?[\w<>\[\]]+\s+[A-Za-z_]\w*\s*\([^)]*\)\s*\{", content))
    imports = set(re.findall(r"\bimport\s+([a-zA-Z0-9_.]+)", content))
    depth = 0
    max_depth = 0
    for char in content:
        if char == "{":
            depth += 1
            max_depth = max(max_depth, depth)
        elif char == "}":
            depth = max(0, depth - 1)
    return {
        "class_count": class_count,
        "function_count": function_count,
        "imports": {item.split(".")[0] for item in imports},
        "max_depth": max_depth,
    }


def analyze_structure(repository: Dict[str, Any]) -> StructuralAnalysisResult:
    files: List[RepositoryFile] = repository["files"]
    result = StructuralAnalysisResult()
    doc_targets = 0
    doc_hits = 0
    result.detected_languages = repository.get("detected_languages", [])
    result.detected_frameworks = repository.get("detected_frameworks", [])

    for repo_file in files:
        if repo_file.language == "Python":
            try:
                parsed = ast.parse(repo_file.content)
                visitor = PythonStructureVisitor()
                visitor.visit(parsed)
                result.function_count += visitor.function_count
                result.class_count += visitor.class_count
                result.method_count += visitor.method_count
                result.max_nesting_depth = max(result.max_nesting_depth, visitor.max_nesting_depth)
                result.unique_libraries.extend(sorted(visitor.imports))
                doc_targets += visitor.total_doc_targets
                doc_hits += visitor.docstring_nodes
                result.python_files_analyzed += 1
            except SyntaxError as exc:
                result.notes.append(f"AST parse failed for {repo_file.path}: {exc}")
        else:
            generic_metrics = analyze_generic_structure(repo_file.language, repo_file.content)
            result.function_count += generic_metrics["function_count"]
            result.class_count += generic_metrics["class_count"]
            result.max_nesting_depth = max(result.max_nesting_depth, generic_metrics["max_depth"])
            result.unique_libraries.extend(sorted(generic_metrics["imports"]))
            if repo_file.language == "Java":
                result.java_files_analyzed += 1
            else:
                result.other_files_analyzed += 1
            doc_targets += 1
            if count_prefixed_comment_lines(repo_file.language, repo_file.content.splitlines()) > 0:
                doc_hits += 1

    unique_libraries = sorted({item for item in result.unique_libraries if item})
    result.unique_libraries = unique_libraries
    result.docstring_ratio = doc_hits / doc_targets if doc_targets else 0.0

    file_count = max(len(files), 1)
    function_density = result.function_count / file_count
    class_density = result.class_count / file_count
    density_score = score_from_penalty(abs(function_density - 4.0) * 8.0)
    class_score = score_from_penalty(abs(class_density - 1.0) * 10.0)
    nesting_score = score_from_penalty(max(result.max_nesting_depth - 3, 0) * 12.0)
    import_score = normalize_score(len(unique_libraries), 2.0, 15.0)
    result.modularity_score = round((density_score * 0.35) + (class_score * 0.2) + (nesting_score * 0.3) + (import_score * 0.15), 2)

    return result


def prepare_ai_context(repository: Dict[str, Any], static_result: StaticAnalysisResult, structural_result: StructuralAnalysisResult) -> str:
    files: List[RepositoryFile] = repository["files"]
    selected_files = sorted(files, key=lambda item: (item.path != repository["primary_file"], -item.size, item.path))[:MAX_AI_FILES]

    metrics_summary = {
        "repository": repository["repository"],
        "primary_file": repository["primary_file"],
        "files_analyzed": len(files),
        "pylint_score": static_result.pylint_score,
        "average_complexity": static_result.average_complexity,
        "maintainability_index": static_result.maintainability_index,
        "security_issue_count": len(static_result.security_issues),
        "function_count": structural_result.function_count,
        "class_count": structural_result.class_count,
        "max_nesting_depth": structural_result.max_nesting_depth,
        "modularity_score": structural_result.modularity_score,
        "docstring_ratio": structural_result.docstring_ratio,
        "libraries": structural_result.unique_libraries[:20],
        "detected_languages": structural_result.detected_languages,
        "detected_frameworks": structural_result.detected_frameworks,
    }

    prompt = [
        "Evaluate the following code repository from a trainee developer.",
        "",
        "Assess:",
        "1. Code readability",
        "2. Algorithmic logic",
        "3. Structure and modularization",
        "4. Security risks",
        "5. Documentation quality",
        "",
        "Return JSON only with this schema:",
        json.dumps(
            {
                "scores": {
                    "readability": 0,
                    "algorithmic_logic": 0,
                    "structure_modularization": 0,
                    "security_risks": 0,
                    "documentation_quality": 0,
                    "overall": 0,
                },
                "strengths": ["..."],
                "weaknesses": ["..."],
                "suggested_improvements": ["..."],
                "security": ["..."],
                "summary": "...",
            },
            indent=2,
        ),
        "",
        "Repository metrics:",
        json.dumps(metrics_summary, indent=2),
        "",
        "Source files:",
    ]

    current_length = len("\n".join(prompt))
    for repo_file in selected_files:
        snippet = repo_file.content[:5000]
        block = f"\n### {repo_file.path}\n```{repo_file.language.lower()}\n{snippet}\n```\n"
        if current_length + len(block) > MAX_AI_CHARS:
            break
        prompt.append(block)
        current_length += len(block)

    return "\n".join(prompt)


def extract_text_from_ai_response(provider: str, payload: Any) -> str:
    if provider in {"groq", "openrouter"}:
        return payload["choices"][0]["message"]["content"]
    if provider == "huggingface":
        if isinstance(payload, list) and payload:
            candidate = payload[0]
            if isinstance(candidate, dict):
                return candidate.get("generated_text") or candidate.get("summary_text") or json.dumps(candidate)
            return str(candidate)
        if isinstance(payload, dict):
            return payload.get("generated_text") or payload.get("summary_text") or json.dumps(payload)
    return json.dumps(payload)


def call_ai_provider(provider: str, model: str, api_key: str, prompt: str) -> str:
    headers = {"Content-Type": "application/json"}
    if provider == "groq":
        headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": model,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": "You are a senior code reviewer. Respond with valid JSON only."},
                {"role": "user", "content": prompt},
            ],
        }
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
    elif provider == "openrouter":
        headers["Authorization"] = f"Bearer {api_key}"
        headers["HTTP-Referer"] = "https://github.com"
        headers["X-Title"] = "GitHub Evaluation Engine"
        payload = {
            "model": model,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": "You are a senior code reviewer. Respond with valid JSON only."},
                {"role": "user", "content": prompt},
            ],
        }
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
    elif provider == "huggingface":
        headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "inputs": f"You are a senior code reviewer. Respond with valid JSON only.\n\n{prompt}",
            "parameters": {"max_new_tokens": 900, "temperature": 0.0, "return_full_text": False},
        }
        response = requests.post(f"https://api-inference.huggingface.co/models/{model}", headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
    else:
        raise ValueError("Unsupported AI provider selected.")

    response.raise_for_status()
    return extract_text_from_ai_response(provider, response.json())


def ai_review(
    repository: Dict[str, Any],
    static_result: StaticAnalysisResult,
    structural_result: StructuralAnalysisResult,
    provider: str,
    model: str,
    api_key: str,
) -> AIReviewResult:
    if not provider or not model or not api_key:
        return AIReviewResult()

    prompt = prepare_ai_context(repository, static_result, structural_result)
    cache_payload = {
        "provider": provider,
        "model": model,
        "repository": repository["repository"],
        "primary_file": repository["primary_file"],
        "prompt": prompt,
    }
    cache_key = hashlib.sha256(json.dumps(cache_payload, sort_keys=True).encode("utf-8")).hexdigest()
    with _cache_lock:
        cached_result = _ai_review_cache.get(cache_key)
    if cached_result:
        return cached_result

    try:
        raw_response = call_ai_provider(provider, model, api_key, prompt)
        payload = safe_json_loads(raw_response)
        scores = payload.get("scores", {})
        result = AIReviewResult(
            provider=provider,
            model=model,
            strengths=[str(item) for item in payload.get("strengths", [])][:8],
            weaknesses=[str(item) for item in payload.get("weaknesses", [])][:8],
            suggested_improvements=[str(item) for item in payload.get("suggested_improvements", [])][:8],
            security=[str(item) for item in payload.get("security", [])][:8],
            scores={key: float(value) for key, value in scores.items()},
            summary=str(payload.get("summary", "")),
            raw_response=raw_response,
        )
        with _cache_lock:
            _ai_review_cache[cache_key] = result
        return result
    except Exception as exc:
        return AIReviewResult(provider=provider, model=model, error=str(exc))


def is_evaluated_result(result: EvaluationResult) -> bool:
    return result.status == "Evaluated"


def build_special_status_result(
    trainee_name: str,
    repository_source: str,
    status: str,
    experience_profile: str,
    review_strictness: int,
    validation_profile: str,
    worksheet: str = "",
    declared_technology: str = "",
    college: str = "",
    superset_id: str = "",
    assessment_week: str = "",
    source_file: str = "",
    evaluation_track: str = ASSESSMENT_SCORE_TRACK,
) -> EvaluationResult:
    return EvaluationResult(
        trainee=trainee_name,
        repository=repository_source,
        primary_file="",
        files_analyzed=0,
        score=0,
        classification=status,
        experience_profile=experience_profile,
        strengths=[],
        weaknesses=[],
        security=[],
        primary_gap="",
        static_analysis=StaticAnalysisResult(),
        structural_analysis=StructuralAnalysisResult(),
        ai_review=AIReviewResult(),
        file_tree="",
        primary_snapshot="",
        reviewed_files=[],
        score_breakdown={},
        review_strictness=review_strictness,
        status=status,
        issue_type="",
        issue_message="",
        output={
            "Score": None,
            "Classification": status,
            "ExperienceProfile": experience_profile,
            "ReviewStrictness": review_strictness,
            "Strengths": [],
            "Weaknesses": [],
            "Security": [],
            "FilesAnalyzed": 0,
            "ScoreBreakdown": {},
            "Status": status,
            "IssueType": "",
            "IssueMessage": "",
            "ValidationProfile": validation_profile,
            "College": college,
            "SupersetID": superset_id,
            "AssessmentWeek": assessment_week,
            "EvaluationTrack": evaluation_track,
        },
        worksheet=worksheet,
        declared_technology=declared_technology,
        languages_tested="",
        superset_id=superset_id,
        assessment_week=assessment_week,
        source_file=source_file,
        evaluation_track=evaluation_track,
    )


def build_imported_scored_result(
    trainee_name: str,
    score: int,
    experience_profile: str,
    review_strictness: int,
    validation_profile: str,
    worksheet: str = "",
    declared_technology: str = "",
    college: str = "",
    superset_id: str = "",
    assessment_week: str = "",
    source_file: str = "",
    evaluation_track: str = ASSESSMENT_SCORE_TRACK,
    submission_id: str = "",
) -> EvaluationResult:
    normalized_score = int(round(clamp(float(score), 0.0, 100.0)))
    return EvaluationResult(
        trainee=trainee_name,
        repository="Imported from scored workbook",
        primary_file="",
        files_analyzed=0,
        score=normalized_score,
        classification=widget_score_band(normalized_score),
        experience_profile=experience_profile,
        strengths=["Imported from a previously scored workbook."],
        weaknesses=[],
        security=[],
        primary_gap="",
        static_analysis=StaticAnalysisResult(notes=["Imported from scored workbook"]),
        structural_analysis=StructuralAnalysisResult(notes=["Imported from scored workbook"]),
        ai_review=AIReviewResult(summary="Imported from scored workbook"),
        file_tree="",
        primary_snapshot="",
        reviewed_files=[],
        score_breakdown={},
        review_strictness=review_strictness,
        status="Evaluated",
        issue_type="",
        issue_message="",
        output={
            "Score": normalized_score,
            "Classification": widget_score_band(normalized_score),
            "ExperienceProfile": experience_profile,
            "ReviewStrictness": review_strictness,
            "Strengths": ["Imported from a previously scored workbook."],
            "Weaknesses": [],
            "Security": [],
            "FilesAnalyzed": 0,
            "ScoreBreakdown": {},
            "Status": "Evaluated",
            "IssueType": "",
            "IssueMessage": "",
            "ValidationProfile": validation_profile,
            "College": college,
            "SupersetID": superset_id,
            "AssessmentWeek": assessment_week,
            "EvaluationTrack": evaluation_track,
            "SubmissionID": submission_id,
            "ImportedFromWorkbook": True,
        },
        worksheet=worksheet,
        declared_technology=declared_technology,
        languages_tested="",
        superset_id=superset_id,
        assessment_week=assessment_week,
        source_file=source_file,
        evaluation_track=evaluation_track,
    )


def build_imported_not_evaluated_result(
    trainee_name: str,
    experience_profile: str,
    review_strictness: int,
    validation_profile: str,
    worksheet: str = "",
    declared_technology: str = "",
    college: str = "",
    superset_id: str = "",
    assessment_week: str = "",
    source_file: str = "",
    evaluation_track: str = ASSESSMENT_SCORE_TRACK,
    submission_id: str = "",
) -> EvaluationResult:
    issue_message = "Imported from a previously scored workbook as Not Evaluated."
    return EvaluationResult(
        trainee=trainee_name,
        repository="Imported from scored workbook",
        primary_file="",
        files_analyzed=0,
        score=0,
        classification=NOT_EVALUATED_CLASSIFICATION,
        experience_profile=experience_profile,
        strengths=[],
        weaknesses=[issue_message],
        security=[],
        primary_gap=issue_message,
        static_analysis=StaticAnalysisResult(notes=[issue_message]),
        structural_analysis=StructuralAnalysisResult(notes=[issue_message]),
        ai_review=AIReviewResult(),
        file_tree="",
        primary_snapshot="",
        reviewed_files=[],
        score_breakdown={},
        review_strictness=review_strictness,
        status="Not Evaluated",
        issue_type="Imported Workbook Status",
        issue_message=issue_message,
        output={
            "Score": None,
            "Classification": NOT_EVALUATED_CLASSIFICATION,
            "ExperienceProfile": experience_profile,
            "ReviewStrictness": review_strictness,
            "Strengths": [],
            "Weaknesses": [issue_message],
            "Security": [],
            "FilesAnalyzed": 0,
            "ScoreBreakdown": {},
            "Status": "Not Evaluated",
            "IssueType": "Imported Workbook Status",
            "IssueMessage": issue_message,
            "ValidationProfile": validation_profile,
            "College": college,
            "SupersetID": superset_id,
            "AssessmentWeek": assessment_week,
            "EvaluationTrack": evaluation_track,
            "SubmissionID": submission_id,
            "ImportedFromWorkbook": True,
        },
        worksheet=worksheet,
        declared_technology=declared_technology,
        languages_tested="",
        superset_id=superset_id,
        assessment_week=assessment_week,
        source_file=source_file,
        evaluation_track=evaluation_track,
    )


def classify_repository_issue(message: str) -> Tuple[str, str]:
    normalized = unquote(str(message or "")).strip()
    lowered = normalized.lower()
    if "repository input is empty" in lowered or "repository must be in owner/repo format" in lowered:
        return "Invalid Link Format", "The repository link is incomplete. Use a full GitHub repository URL or owner/repo value."
    if "not a github project or profile page" in lowered:
        return "Unsupported Link", "The source must be a GitHub repository or repository subfolder, not a GitHub project or profile page."
    if "only github.com repositories are supported" in lowered or "unsupported characters" in lowered:
        return "Unsupported Link", "The source must be a valid github.com repository or folder URL."
    if "local path does not exist" in lowered or "local path must be a folder" in lowered:
        return "Invalid Local Path", normalized
    if "rate limit exceeded" in lowered:
        return "GitHub Rate Limit", "GitHub temporarily blocked requests. Add a token or retry later."
    if "archive download failed" in lowered or "404" in lowered or "401" in lowered or "403" in lowered:
        return "Repository Access Issue", "The repository, branch, or folder path could not be accessed. Verify the GitHub link and permissions."
    if "no supported code files were found" in lowered:
        return "No Supported Code Files", f"No supported source files were found in the selected repository or subfolder. The current engine reads {SUPPORTED_STACKS_MESSAGE} files."
    return "Evaluation Issue", normalized or "The repository could not be evaluated because an unexpected issue occurred."


def build_failed_evaluation_result(
    trainee_name: str,
    repository_source: str,
    experience_profile: str,
    review_strictness: int,
    exc: Exception,
    validation_profile: str,
    worksheet: str = "",
    declared_technology: str = "",
    college: str = "",
    superset_id: str = "",
    assessment_week: str = "",
    source_file: str = "",
    evaluation_track: str = ASSESSMENT_SCORE_TRACK,
) -> EvaluationResult:
    issue_type, issue_message = classify_repository_issue(str(exc))
    return EvaluationResult(
        trainee=trainee_name,
        repository=repository_source,
        primary_file="",
        files_analyzed=0,
        score=0,
        classification=NOT_EVALUATED_CLASSIFICATION,
        experience_profile=experience_profile,
        strengths=[],
        weaknesses=[issue_message],
        security=[],
        primary_gap=issue_message,
        static_analysis=StaticAnalysisResult(notes=[str(exc)]),
        structural_analysis=StructuralAnalysisResult(notes=[str(exc)]),
        ai_review=AIReviewResult(),
        file_tree="",
        primary_snapshot="",
        reviewed_files=[],
        score_breakdown={},
        review_strictness=review_strictness,
        status="Not Evaluated",
        issue_type=issue_type,
        issue_message=issue_message,
        worksheet=worksheet,
        declared_technology=declared_technology,
        languages_tested="",
        output={
            "Score": None,
            "Classification": NOT_EVALUATED_CLASSIFICATION,
            "ExperienceProfile": experience_profile,
            "ReviewStrictness": review_strictness,
            "Strengths": [],
            "Weaknesses": [issue_message],
            "Security": [],
            "FilesAnalyzed": 0,
            "ScoreBreakdown": {},
            "Status": "Not Evaluated",
            "IssueType": issue_type,
            "IssueMessage": issue_message,
            "ValidationProfile": validation_profile,
            "College": college,
            "SupersetID": superset_id,
            "AssessmentWeek": assessment_week,
            "EvaluationTrack": evaluation_track,
        },
        superset_id=superset_id,
        assessment_week=assessment_week,
        source_file=source_file,
        evaluation_track=evaluation_track,
    )


def compute_score(
    static_result: StaticAnalysisResult,
    structural_result: StructuralAnalysisResult,
    ai_result: AIReviewResult,
    review_strictness: int = 50,
    experience_profile: str = "Fresher",
) -> Tuple[int, str, Dict[str, float]]:
    profile = scoring_profile(experience_profile)
    penalty_factor = strictness_multiplier(review_strictness) * profile["penalty_scale"]
    pylint_component = clamp(profile["pylint_base"] + static_result.pylint_score * profile["pylint_scale"], 0.0, 100.0)
    complexity_component = score_from_penalty(
        max(static_result.average_complexity - profile["complexity_threshold"], 0.0) * profile["complexity_penalty"] * penalty_factor
    )
    maintainability_component = clamp(static_result.maintainability_index, 0.0, 100.0)
    error_penalty = (
        static_result.error_count * profile["error_weight"]
        + static_result.fatal_count * profile["fatal_weight"]
        + static_result.naming_issues * profile["naming_weight"]
    )
    error_component = score_from_penalty(error_penalty * penalty_factor)

    static_quality = (
        pylint_component * 0.28
        + complexity_component * 0.22
        + maintainability_component * 0.30
        + error_component * 0.20
    )

    modularity_component = clamp(structural_result.modularity_score, 0.0, 100.0)
    nesting_component = score_from_penalty(
        max(structural_result.max_nesting_depth - profile["nesting_threshold"], 0.0) * profile["nesting_penalty"] * penalty_factor
    )
    library_component = normalize_score(len(structural_result.unique_libraries), 1.0, 12.0)
    ast_structure = modularity_component * 0.55 + nesting_component * 0.25 + library_component * 0.2

    ai_readability = clamp(ai_result.scores.get("readability", static_quality / 10.0) * 10.0, 0.0, 100.0)
    ai_structure = clamp(ai_result.scores.get("structure_modularization", ast_structure / 10.0) * 10.0, 0.0, 100.0)
    ai_logic_proxy = clamp(complexity_component * 0.45 + error_component * 0.30 + maintainability_component * 0.25, 0.0, 100.0)
    ai_logic = clamp(ai_result.scores.get("algorithmic_logic", ai_logic_proxy / 10.0) * 10.0, 0.0, 100.0)
    code_quality = static_quality * 0.90 + ai_readability * 0.10
    structure = ast_structure * 0.85 + ai_structure * 0.15
    logic_correctness = ai_logic * 0.65 + complexity_component * 0.20 + error_component * 0.15

    bandit_penalty = sum(security_penalty_weight(issue) for issue in static_result.security_issues) * penalty_factor
    bandit_component = score_from_penalty(bandit_penalty)
    ai_security = clamp(ai_result.scores.get("security_risks", bandit_component / 10.0) * 10.0, 0.0, 100.0)
    security = bandit_component * 0.7 + ai_security * 0.3

    doc_component = clamp(structural_result.docstring_ratio * 100.0, 0.0, 100.0)
    doc_penalty = min(static_result.documentation_issues, profile["doc_penalty_cap"]) * profile["doc_penalty"] * penalty_factor
    pylint_doc_component = score_from_penalty(doc_penalty)
    ai_docs = clamp(ai_result.scores.get("documentation_quality", ((doc_component * 0.55 + pylint_doc_component * 0.45) / 10.0)) * 10.0, 0.0, 100.0)
    documentation = doc_component * 0.50 + pylint_doc_component * 0.30 + ai_docs * 0.20

    raw_composite = (
        code_quality * 0.25
        + structure * 0.20
        + logic_correctness * 0.20
        + security * 0.15
        + documentation * 0.05
    )
    ai_overall = clamp(ai_result.scores.get("overall", raw_composite / 10.0) * 10.0, 0.0, 100.0)
    weighted_score = raw_composite + ai_overall * 0.15
    final_score = profile["baseline"] + (weighted_score - profile["baseline_center"]) * profile["baseline_scale"]
    final_score = clamp(final_score, 0.0, 100.0)

    rounded_score = int(round(final_score))
    thresholds = profile["thresholds"]
    if rounded_score >= thresholds["Excellent"]:
        classification = "Excellent"
    elif rounded_score >= thresholds["Good"]:
        classification = "Good"
    elif rounded_score >= thresholds["Needs Attention"]:
        classification = "Needs Attention"
    else:
        classification = "Retest Required"

    breakdown = {
        "Code Quality": round(code_quality, 2),
        "Structure": round(structure, 2),
        "Logic Correctness": round(logic_correctness, 2),
        "Security": round(security, 2),
        "Documentation": round(documentation, 2),
        "AI Evaluation": round(ai_overall, 2),
    }
    return rounded_score, classification, breakdown


def derive_static_strengths(
    static_result: StaticAnalysisResult,
    structural_result: StructuralAnalysisResult,
    files_analyzed: int = 0,
) -> List[str]:
    strengths: List[str] = []
    python_focused = structural_result.python_files_analyzed > 0
    detected_languages = ", ".join(structural_result.detected_languages) or "the detected stack"
    if files_analyzed:
        strengths.append(f"Evaluation covered {files_analyzed} supported source file(s) after ignoring non-code artifacts.")
    if static_result.pylint_score >= 8.0:
        if python_focused:
            strengths.append(f"Pylint score is healthy at {static_result.pylint_score:.1f}, suggesting comparatively stable coding hygiene.")
        else:
            strengths.append(f"Heuristic quality score is healthy at {static_result.pylint_score * 10:.0f}/100 across {detected_languages}.")
    if static_result.error_count == 0 and static_result.fatal_count == 0 and files_analyzed and python_focused:
        strengths.append("No pylint error or fatal findings were raised in the analyzed code files.")
    if static_result.average_complexity and static_result.average_complexity <= 5.0:
        strengths.append(f"Average cyclomatic complexity is controlled at {static_result.average_complexity:.2f}, which supports easier debugging.")
    if static_result.maintainability_index >= 70.0:
        strengths.append(f"Maintainability index is {static_result.maintainability_index:.1f}, which indicates the code should be easier to extend.")
    if structural_result.modularity_score >= 65.0:
        strengths.append(f"Repository structure scored {structural_result.modularity_score:.1f} for modularity, indicating acceptable separation of logic.")
    if structural_result.docstring_ratio >= 0.4:
        doc_label = "measured Python targets" if python_focused else "measured source files"
        strengths.append(f"Documentation coverage is present across {structural_result.docstring_ratio:.0%} of {doc_label}.")
    if structural_result.max_nesting_depth <= 3:
        strengths.append(f"Control-flow nesting peaks at depth {structural_result.max_nesting_depth}, which stays within a manageable range.")
    if not static_result.security_issues:
        if python_focused:
            strengths.append("Bandit did not report any direct security findings in the analyzed Python files.")
        else:
            strengths.append("No direct security-risk patterns were detected in the analyzed source files.")
    return strengths[:5]


def derive_static_weaknesses(
    static_result: StaticAnalysisResult,
    structural_result: StructuralAnalysisResult,
    files_analyzed: int = 0,
) -> List[str]:
    weaknesses: List[str] = []
    python_focused = structural_result.python_files_analyzed > 0
    if files_analyzed == 0:
        weaknesses.append("No supported code files were found after filtering the repository contents.")
        return weaknesses
    if static_result.documentation_issues:
        if python_focused:
            weaknesses.append(f"{static_result.documentation_issues} docstring-related lint issue(s) lowered documentation quality.")
        else:
            weaknesses.append(f"{static_result.documentation_issues} documentation/comment coverage issue(s) lowered maintainability confidence.")
    if static_result.naming_issues:
        weaknesses.append(f"{static_result.naming_issues} naming convention issue(s) were detected by lint analysis.")
    if (static_result.error_count or static_result.fatal_count) and python_focused:
        weaknesses.append(
            f"Pylint reported {static_result.error_count} error(s) and {static_result.fatal_count} fatal issue(s), which points to correctness risk."
        )
    if static_result.average_complexity > 8.0:
        weaknesses.append(f"Average cyclomatic complexity is {static_result.average_complexity:.2f}, which is higher than the target range for freshers.")
    if structural_result.max_nesting_depth > 3:
        weaknesses.append(f"Maximum nesting depth reached {structural_result.max_nesting_depth}, making parts of the logic harder to follow.")
    if structural_result.modularity_score < 45.0:
        weaknesses.append(f"Modularity score is {structural_result.modularity_score:.1f}, which suggests the repository is not well decomposed yet.")
    if structural_result.docstring_ratio < 0.2:
        weaknesses.append(f"Measured documentation coverage is only {structural_result.docstring_ratio:.0%}, so implementation intent is often left implicit.")
    if static_result.security_issues:
        signal_label = "Bandit and heuristic analysis" if python_focused else "Heuristic security analysis"
        weaknesses.append(f"{signal_label} reported {len(static_result.security_issues)} security finding(s) that should be reviewed before deployment.")
    parse_notes = [note for note in static_result.notes + structural_result.notes if "failed" in note.lower()]
    for note in parse_notes[:2]:
        weaknesses.append(note)
    return weaknesses[:5]


def evaluate_repository(
    repo_input: str,
    github_token: str,
    provider: str,
    model: str,
    api_key: str,
    source_mode: str = "auto",
    review_strictness: int = 50,
    experience_profile: str = "Fresher",
    validation_profile: str = "Resilient (Recommended)",
) -> EvaluationResult:
    log_task1_event(
        "task1_repository_analysis_start",
        repository_input=repo_input,
        source_mode=source_mode,
        ai_enabled=bool(provider and model and api_key),
        review_strictness=review_strictness,
        experience_profile=experience_profile,
    )
    repository = fetch_repository(repo_input, github_token, source_mode)
    log_task1_event(
        "task1_repository_fetch_complete",
        repository=repository["repository"],
        file_count=len(repository["files"]),
        languages=",".join(repository.get("detected_languages", [])),
        frameworks=",".join(repository.get("detected_frameworks", [])),
    )
    static_result = analyze_static(repository)
    log_task1_event(
        "task1_repository_static_complete",
        repository=repository["repository"],
        average_complexity=round(static_result.average_complexity, 2),
        security_findings=len(static_result.security_issues),
        pylint_score=round(static_result.pylint_score, 2),
    )
    structural_result = analyze_structure(repository)
    log_task1_event(
        "task1_repository_structure_complete",
        repository=repository["repository"],
        function_count=structural_result.function_count,
        class_count=structural_result.class_count,
        modularity_score=round(structural_result.modularity_score, 2),
        max_nesting_depth=structural_result.max_nesting_depth,
    )
    ai_result = ai_review(repository, static_result, structural_result, provider, model, api_key)
    log_task1_event(
        "task1_repository_ai_complete",
        repository=repository["repository"],
        provider=provider,
        model=model,
        ai_score_keys=",".join(sorted(ai_result.scores.keys())),
    )
    score, classification, score_breakdown = compute_score(
        static_result,
        structural_result,
        ai_result,
        review_strictness,
        experience_profile,
    )

    strengths = ai_result.strengths or derive_static_strengths(static_result, structural_result, len(repository["files"])) or ["No strong positive signal could be derived from the measured repository metrics."]
    weaknesses = ai_result.weaknesses or derive_static_weaknesses(static_result, structural_result, len(repository["files"])) or ["No dominant risk signal was produced beyond the measured repository metrics."]
    security = ai_result.security or [issue["text"] for issue in static_result.security_issues][:5]
    primary_gap = (security[0] if security else weaknesses[0]) if (security or weaknesses) else "None"

    output = {
        "Score": score,
        "Classification": classification,
        "Status": "Evaluated",
        "IssueType": "",
        "IssueMessage": "",
        "ValidationProfile": validation_profile,
        "ExperienceProfile": experience_profile,
        "ReviewStrictness": review_strictness,
        "Complexity": round(static_result.average_complexity, 2),
        "Strengths": strengths,
        "Weaknesses": weaknesses,
        "Security": security,
        "FilesAnalyzed": len(repository["files"]),
        "DetectedLanguages": repository.get("detected_languages", []),
        "DetectedFrameworks": repository.get("detected_frameworks", []),
        "LanguagesTested": languages_tested_from_files(repository["files"]),
        "ScoreBreakdown": score_breakdown,
    }

    trainee_name = Path(repository["repository"]).name or repository["repository"].split("/")[-1]
    log_task1_event(
        "task1_repository_analysis_complete",
        repository=repository["repository"],
        trainee=trainee_name,
        score=score,
        classification=classification,
        files_analyzed=len(repository["files"]),
    )

    return EvaluationResult(
        trainee=trainee_name,
        repository=repository["repository"],
        primary_file=repository["primary_file"],
        files_analyzed=len(repository["files"]),
        score=score,
        classification=classification,
        experience_profile=experience_profile,
        strengths=strengths,
        weaknesses=weaknesses,
        security=security,
        primary_gap=primary_gap,
        static_analysis=static_result,
        structural_analysis=structural_result,
        ai_review=ai_result,
        file_tree=repository["file_tree"],
        primary_snapshot=repository["primary_snapshot"][:5000],
        reviewed_files=repository["files"],
        score_breakdown=score_breakdown,
        review_strictness=review_strictness,
        status="Evaluated",
        issue_type="",
        issue_message="",
        output=output,
        languages_tested=languages_tested_from_files(repository["files"]),
    )


def evaluate_submission(
    trainee_name: str,
    repo_input: str,
    github_token: str,
    provider: str,
    model: str,
    api_key: str,
    source_mode: str = "auto",
    review_strictness: int = 50,
    experience_profile: str = "Fresher",
    validation_profile: str = "Resilient (Recommended)",
    worksheet: str = "",
    declared_technology: str = "",
    college: str = "",
    superset_id: str = "",
    assessment_week: str = "",
    source_file: str = "",
    submission_status: str = "Pending Evaluation",
    evaluation_track: str = ASSESSMENT_SCORE_TRACK,
) -> EvaluationResult:
    if submission_status in SPECIAL_RESULT_STATUSES:
        return build_special_status_result(
            trainee_name.strip() or superset_id or "Unknown Submission",
            repo_input,
            submission_status,
            experience_profile,
            review_strictness,
            validation_profile,
            worksheet,
            declared_technology,
            college,
            normalize_superset_id(superset_id),
            assessment_week,
            source_file,
            evaluation_track,
        )
    result = evaluate_repository(
        repo_input,
        github_token,
        provider,
        model,
        api_key,
        source_mode,
        review_strictness,
        experience_profile,
        validation_profile,
    )
    result.trainee = trainee_name.strip() or result.trainee
    result.worksheet = worksheet
    result.declared_technology = declared_technology
    result.superset_id = normalize_superset_id(superset_id)
    result.assessment_week = assessment_week or "Week 1"
    result.source_file = source_file
    result.evaluation_track = normalize_evaluation_track(evaluation_track)
    result.output["College"] = college
    result.output["SupersetID"] = result.superset_id
    result.output["AssessmentWeek"] = result.assessment_week
    result.output["EvaluationTrack"] = result.evaluation_track
    return result


def evaluate_repositories(
    repo_inputs: List[str],
    github_token: str,
    provider: str,
    model: str,
    api_key: str,
    source_mode: str = "auto",
    review_strictness: int = 50,
    experience_profile: str = "Fresher",
    validation_profile: str = "Resilient (Recommended)",
) -> List[EvaluationResult]:
    results: List[EvaluationResult] = []
    max_workers = min(8, max(len(repo_inputs), 1))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                evaluate_repository,
                repo,
                github_token,
                provider,
                model,
                api_key,
                source_mode,
                review_strictness,
                experience_profile,
                validation_profile,
            ): repo
            for repo in repo_inputs
        }
        for future in as_completed(future_map):
            repo = future_map[future]
            try:
                results.append(future.result())
            except Exception as exc:
                trainee_name = Path(repo).name or repo.split("/")[-1]
                results.append(
                    build_failed_evaluation_result(
                        trainee_name,
                        repo,
                        experience_profile,
                        review_strictness,
                        exc,
                        validation_profile,
                        "",
                        "",
                    )
                )
    return sort_evaluation_results(results)


def evaluate_submissions(
    submissions: List[Dict[str, str]],
    github_token: str,
    provider: str,
    model: str,
    api_key: str,
    source_mode: str = "auto",
    review_strictness: int = 50,
    experience_profile: str = "Fresher",
    validation_profile: str = "Resilient (Recommended)",
) -> List[EvaluationResult]:
    results: List[EvaluationResult] = []
    max_workers = min(8, max(len(submissions), 1))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                evaluate_submission,
                submission["trainee"],
                submission["source"],
                github_token,
                provider,
                model,
                api_key,
                source_mode,
                review_strictness,
                experience_profile,
                validation_profile,
                submission.get("worksheet", ""),
                submission.get("declared_technology", ""),
                submission.get("college", ""),
                submission.get("superset_id", ""),
                submission.get("assessment_week", "Week 1"),
                submission.get("source_file", ""),
                submission.get("submission_status", "Pending Evaluation"),
            ): submission
            for submission in submissions
        }
        for future in as_completed(future_map):
            submission = future_map[future]
            try:
                results.append(future.result())
            except Exception as exc:
                results.append(
                    build_failed_evaluation_result(
                        submission["trainee"],
                        submission["source"],
                        experience_profile,
                        review_strictness,
                        exc,
                        validation_profile,
                        submission.get("worksheet", ""),
                        submission.get("declared_technology", ""),
                        submission.get("college", ""),
                        submission.get("superset_id", ""),
                        submission.get("assessment_week", ""),
                        submission.get("source_file", ""),
                    )
                )
    return sort_evaluation_results(results)


def render_summary(
    results: List[EvaluationResult],
    raw_results: Optional[List[EvaluationResult]] = None,
    selected_week: str = ALL_WEEKS_LABEL,
) -> None:
    total = len(results)
    evaluated = [item for item in results if is_evaluated_result(item)]
    excellent = sum(widget_score_band(item.score) == "Excellent" for item in evaluated)
    good = sum(widget_score_band(item.score) == "Good" for item in evaluated)
    needs_attention = sum(widget_score_band(item.score) == "Needs Attention" for item in evaluated)
    retest_required = sum(widget_score_band(item.score) == "Retest Required" for item in evaluated)
    if is_all_cycle_label(selected_week) and raw_results is not None:
        absent = count_exception_candidates(raw_results, "Absent")
        dropout = count_exception_candidates(raw_results, "Dropout")
        no_show = count_exception_candidates(raw_results, "No Show")
        not_evaluated = count_exception_candidates(raw_results, "Not Evaluated")
    else:
        absent = sum(item.status == "Absent" for item in results)
        dropout = sum(item.status == "Dropout" for item in results)
        no_show = sum(item.status == "No Show" for item in results)
        not_evaluated = sum(is_retryable_result(item) for item in results)
    average_score = round(sum(item.score for item in evaluated) / len(evaluated), 1) if evaluated else 0.0

    top_metrics = st.columns(6)
    top_metrics[0].metric("Total Rows", total)
    top_metrics[1].metric("Evaluated", len(evaluated), delta=f"Avg {average_score}")
    top_metrics[2].metric("Excellent", excellent)
    top_metrics[3].metric("Good", good)
    top_metrics[4].metric("Scope for Improvement", needs_attention)
    top_metrics[5].metric("Reassessment Required", retest_required)

    st.markdown("**Exception Candidates**")
    bottom_metrics = st.columns(4)
    bottom_metrics[0].metric("Absent", absent)
    bottom_metrics[1].metric("Dropout", dropout)
    bottom_metrics[2].metric("No Show", no_show)
    bottom_metrics[3].metric("Not Evaluated", not_evaluated)
    if is_all_cycle_label(selected_week):
        st.caption("All-cycle exception counts flag any student who had that status in at least one cycle. Use the View buttons below for detailed cycle-level data.")
        st.caption("For `Not Evaluated`, click `View Not Evaluated` to see which GitHub cycle data was missing or failed.")
    else:
        st.caption("Use the View buttons below for detailed exception data.")

    top_filter_buttons = [
        ("All", "View All"),
        ("Evaluated", "View Evaluated"),
        ("Excellent", "View Excellent"),
        ("Good", "View Good"),
        ("Needs Attention", "View Scope for Improvement"),
        ("Retest Required", "View Reassessment Required"),
    ]
    top_button_columns = st.columns(len(top_filter_buttons))
    for index, (filter_value, button_label) in enumerate(top_filter_buttons):
        with top_button_columns[index]:
            if st.button(button_label, key=f"task1_summary_filter_{normalize_column_name(filter_value)}", use_container_width=True):
                st.session_state["task1_result_filter"] = filter_value
                st.rerun()

    bottom_filter_buttons = [
        ("Absent", "View Absent"),
        ("Dropout", "View Dropout"),
        ("No Show", "View No Show"),
        ("Not Evaluated", "View Not Evaluated"),
    ]
    bottom_button_columns = st.columns(len(bottom_filter_buttons))
    for index, (filter_value, button_label) in enumerate(bottom_filter_buttons):
        with bottom_button_columns[index]:
            if st.button(button_label, key=f"task1_summary_filter_{normalize_column_name(filter_value)}", use_container_width=True):
                st.session_state["task1_result_filter"] = filter_value
                st.rerun()


def filter_task1_results(
    results: List[EvaluationResult],
    selected_filter: str,
    raw_results: Optional[List[EvaluationResult]] = None,
    selected_week: str = ALL_WEEKS_LABEL,
) -> List[EvaluationResult]:
    if selected_filter == "All":
        return results
    if is_all_cycle_label(selected_week) and raw_results is not None and (selected_filter == "Not Evaluated" or selected_filter in SPECIAL_RESULT_STATUSES):
        return filter_exception_results(raw_results, selected_filter)
    if selected_filter == "Evaluated":
        return [item for item in results if is_evaluated_result(item)]
    if selected_filter == "Not Evaluated":
        return [item for item in results if is_retryable_result(item)]
    if selected_filter in SPECIAL_RESULT_STATUSES:
        return [item for item in results if item.status == selected_filter]
    return [item for item in results if is_evaluated_result(item) and widget_score_band(item.score) == selected_filter]


def render_leaderboard(results: List[EvaluationResult]) -> None:
    rows = [
            {
                "Trainee": item.trainee,
                "Superset ID": get_result_superset_id(item),
                "Evaluation Track": get_result_track(item),
                "GitHub Cycle": get_result_week(item),
                "Worksheet": item.worksheet or "Manual Entry",
                "Declared Tech": item.declared_technology or "Not Declared",
                "Languages Tested": item.languages_tested or "Not Detected",
            "Score": item.score if is_evaluated_result(item) else None,
            "Status": item.status,
            "Classification": workbook_preview_term(item.classification),
            "Files Analyzed": item.files_analyzed,
        }
        for item in results
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_issue_table(results: List[EvaluationResult]) -> None:
    issue_rows = pd.DataFrame(
        [
            {
                "Trainee": item.trainee,
                "Evaluation Track": get_result_track(item),
                "GitHub Cycle": get_result_week(item),
                "Worksheet": item.worksheet or "Manual Entry",
                "Declared Tech": item.declared_technology or "Not Declared",
                "Repository": item.repository,
                "Issue Type": item.issue_type,
                "Issue": item.issue_message,
            }
            for item in results
            if is_retryable_result(item)
        ]
    )
    if issue_rows.empty:
        st.success("No repository issues are pending trainer correction.")
    else:
        st.dataframe(issue_rows, use_container_width=True, hide_index=True)


def render_repository_issue_recovery(
    results: List[EvaluationResult],
    config: Dict[str, str],
    source_mode: str,
    review_strictness: int,
    experience_profile: str,
    show_heading: bool = True,
) -> None:
    blocked_results = [item for item in results if is_retryable_result(item)]
    if not blocked_results:
        return

    if show_heading:
        render_section_intro(
            "Retry Repository Exceptions",
            "Select a not-evaluated row, paste the corrected repository source, and rerun only that student without resetting the full session.",
        )
    else:
        st.markdown("**Repository Evaluation Failures**")
        st.caption("These rows reached evaluation but could not be scored. Correct the repository source and retry only the affected student.")

    render_issue_table(blocked_results)
    option_labels = {
        " | ".join(
            segment
            for segment in [
                item.trainee,
                get_result_track(item),
                get_result_week(item),
                item.worksheet or "Manual Entry",
                item.declared_technology or "",
                item.issue_type,
            ]
            if segment
        ): item
        for item in blocked_results
    }
    selected_label = st.selectbox(
        "Not Evaluated Submission",
        list(option_labels.keys()),
        key="task1_issue_recovery_selection",
    )
    corrected_source = st.text_input(
        "Corrected Repository Source",
        key="task1_issue_recovery_source",
        placeholder="https://github.com/owner/repository or C:\\local\\project",
    ).strip()
    if st.button("Retry Selected Repository", key="task1_issue_recovery_retry", type="primary"):
        if not corrected_source:
            st.error("Enter a corrected repository source before retrying the evaluation.")
        else:
            selected = option_labels[selected_label]
            try:
                retried = evaluate_submission(
                    selected.trainee,
                    corrected_source,
                    config["github_token"],
                    config["provider"],
                    config["model"],
                    config["api_key"],
                    source_mode,
                    review_strictness,
                    experience_profile,
                    config["validation_profile"],
                    selected.worksheet,
                    selected.declared_technology,
                    selected.output.get("College", "") if isinstance(selected.output, dict) else "",
                    get_result_superset_id(selected),
                    get_result_week(selected),
                    getattr(selected, "source_file", ""),
                    "Pending Evaluation",
                    get_result_track(selected),
                )
            except Exception as exc:
                st.error(f"Retry failed: {exc}")
            else:
                updated_results = []
                replaced = False
                for item in results:
                    if (
                        not replaced
                        and item.trainee == selected.trainee
                        and item.repository == selected.repository
                        and item.worksheet == selected.worksheet
                        and get_result_week(item) == get_result_week(selected)
                        and get_result_superset_id(item) == get_result_superset_id(selected)
                        and get_result_track(item) == get_result_track(selected)
                    ):
                        updated_results.append(retried)
                        replaced = True
                    else:
                        updated_results.append(item)
                st.session_state["evaluation_results"] = sort_evaluation_results(updated_results)
                sync_task1_results_to_shared_eval()
                st.session_state["task1_issue_recovery_source"] = ""
                st.session_state["task1_result_filter"] = "All"
                st.success(f"Retry completed for {selected.trainee}.")
                st.rerun()


def render_task1_exception_center(
    manual_candidates: List[Dict[str, str]],
    results: List[EvaluationResult],
    controls_disabled: bool,
    config: Dict[str, str],
    source_mode: str,
    review_strictness: int,
    experience_profile: str,
) -> None:
    overrides = st.session_state.get("task1_manual_overrides", {})
    unresolved_candidates = [candidate for candidate in manual_candidates if candidate["submission_id"] not in overrides]
    blocked_results = [item for item in results if is_retryable_result(item)]

    if not unresolved_candidates and not overrides and not blocked_results:
        return

    render_section_intro(
        "GitHub Exception Center",
        "Missing or blank GitHub links and repository evaluation failures are handled together here. `Absent`, `Dropout`, and `No Show` remain final status rows, not correction items.",
    )

    overview_frame = build_task1_exception_overview(manual_candidates, results)
    if not overview_frame.empty:
        st.dataframe(overview_frame, use_container_width=True, hide_index=True)

    if unresolved_candidates or overrides:
        render_missing_link_recovery(manual_candidates, controls_disabled, show_heading=False)

    if blocked_results:
        render_repository_issue_recovery(
            results,
            config,
            source_mode,
            review_strictness,
            experience_profile,
            show_heading=False,
        )


def render_deep_audit(results: List[EvaluationResult]) -> None:
    render_section_intro(
        "Deep Audit View",
        "Inspect repository structure, reviewed files, scoring buckets, and coaching notes for an individual trainee.",
    )
    if not results:
        st.info("No repositories match the current college filter.")
        return

    candidate_groups = group_results_by_candidate(results)

    name_counts: Dict[str, int] = {}
    for candidate_results in candidate_groups.values():
        name_counts[candidate_results[0].trainee] = name_counts.get(candidate_results[0].trainee, 0) + 1

    option_map: Dict[str, Tuple[str, str, str]] = {}
    for candidate_key, candidate_results in candidate_groups.items():
        candidate = candidate_results[0]
        base_label = candidate.trainee
        superset_id = get_result_superset_id(candidate)
        display_label = f"{base_label} ({superset_id})" if superset_id and name_counts.get(base_label, 0) > 1 else base_label
        sibling_tracks = {
            get_result_track(item)
            for other_key, other_results in candidate_groups.items()
            if other_results and other_results[0].trainee == candidate.trainee and get_result_superset_id(other_results[0]) == superset_id
            for item in other_results[:1]
        }
        if len(sibling_tracks) > 1:
            display_label = f"{display_label} | {track_display_label(get_result_track(candidate))}"
        option_map[display_label] = candidate_key

    selector_left, selector_right = st.columns(2)
    with selector_left:
        selected_label = st.selectbox("Select Trainee Repository", sorted(option_map.keys()))
    selected_candidate_results = candidate_groups[option_map[selected_label]]
    candidate_week_options = sort_week_labels(get_result_week(item) for item in selected_candidate_results)
    candidate_track = get_result_track(selected_candidate_results[0]) if selected_candidate_results else ASSESSMENT_SCORE_TRACK
    default_all_cycle = default_cycle_label_for_track(candidate_track)
    week_options = [default_all_cycle] + candidate_week_options if len(candidate_week_options) > 1 else candidate_week_options
    default_week = default_all_cycle if len(candidate_week_options) > 1 else (candidate_week_options[0] if candidate_week_options else default_all_cycle)
    if st.session_state.get("task1_deep_audit_week_filter", "") not in week_options:
        st.session_state["task1_deep_audit_week_filter"] = default_week
    with selector_right:
        selected_week = st.selectbox("Select GitHub Cycle", week_options, key="task1_deep_audit_week_filter")

    summary_results = aggregate_candidate_results(selected_candidate_results, selected_week)
    summary_selected = summary_results[0] if summary_results else None
    if summary_selected is None:
        st.info("No deep-audit data is available for the selected trainee and week.")
        return

    selected = summary_selected
    detailed_week = selected_week
    if is_all_cycle_label(selected_week):
        week_summary_frame = build_candidate_week_summary(selected_candidate_results)
        weeks_evaluated = int(week_summary_frame["Status"].eq("Evaluated").sum()) if not week_summary_frame.empty else 0
        exception_weeks = int(week_summary_frame["Status"].ne("Evaluated").sum()) if not week_summary_frame.empty else 0

        summary_metric1, summary_metric2, summary_metric3, summary_metric4, summary_metric5 = st.columns(5)
        summary_metric1.metric("Profile", summary_selected.experience_profile)
        summary_metric2.metric("Average Score", f"{summary_selected.score}/100" if is_evaluated_result(summary_selected) else "N/A")
        summary_metric3.metric("Average Grade", summary_selected.classification)
        summary_metric4.metric("Weeks Evaluated", weeks_evaluated)
        summary_metric5.metric("Weeks with Exceptions", exception_weeks)

        st.markdown("**All-Cycle Summary**")
        if not week_summary_frame.empty:
            st.dataframe(week_summary_frame, use_container_width=True, hide_index=True)

        if candidate_week_options:
            if st.session_state.get("task1_deep_audit_detail_week_filter", "") not in candidate_week_options:
                preferred_detail_week = next(
                    (row["GitHub Cycle"] for _, row in week_summary_frame.iterrows() if row.get("Status") == "Evaluated"),
                    candidate_week_options[0],
                )
                st.session_state["task1_deep_audit_detail_week_filter"] = preferred_detail_week
            detailed_week = st.selectbox(
                "Select Cycle for Detailed Evidence",
                candidate_week_options,
                key="task1_deep_audit_detail_week_filter",
            )
            detailed_results = aggregate_candidate_results(selected_candidate_results, detailed_week)
            selected = detailed_results[0] if detailed_results else None
        else:
            selected = None

        if selected is None:
            st.info("No detailed cycle-level evidence is available for the selected trainee.")
            return
        st.caption(f"Detailed repository evidence below reflects `{detailed_week}`.")
    else:
        st.session_state["task1_deep_audit_detail_week_filter"] = ""

    if is_special_status_result(selected):
        st.info(f"This submission is marked as `{selected.status}` for {get_result_week(selected)}.")
        status_summary = pd.DataFrame(
            [
                {
                    "Trainee": selected.trainee,
                    "Superset ID": get_result_superset_id(selected),
                    "GitHub Cycle": get_result_week(selected),
                    "Worksheet": selected.worksheet or "Manual Entry",
                    "Declared Tech": selected.declared_technology or "Not Declared",
                    "Repository": selected.repository or "No repository source captured.",
                    "Status": selected.status,
                }
            ]
        )
        st.dataframe(status_summary, use_container_width=True, hide_index=True)
        return
    if not is_evaluated_result(selected):
        st.warning(f"This submission was not evaluated. {selected.issue_type}: {selected.issue_message}")
        issue_summary = pd.DataFrame(
            [
                {
                    "Trainee": selected.trainee,
                    "Superset ID": get_result_superset_id(selected),
                    "GitHub Cycle": get_result_week(selected),
                    "Worksheet": selected.worksheet or "Manual Entry",
                    "Declared Tech": selected.declared_technology or "Not Declared",
                    "Repository": selected.repository or "No repository source captured.",
                    "Issue Type": selected.issue_type,
                    "Issue": selected.issue_message,
                }
            ]
        )
        st.dataframe(issue_summary, use_container_width=True, hide_index=True)
        st.info("Use the retry section in Workbook Preview to correct the repository source and rerun this submission.")
        return
    bucket_table = build_bucket_table(selected.score_breakdown)
    expand_snapshots = st.session_state.get("expand_code_snapshots", False)

    if is_all_cycle_label(selected_week):
        st.markdown(f"**Detailed Evidence for {detailed_week}**")

    metric1, metric2, metric3, metric4, metric5, metric6, metric7 = st.columns(7)
    metric1.metric("Profile", selected.experience_profile)
    metric2.metric("Review Strictness", str(selected.review_strictness))
    metric3.metric("Selected Cycle Score" if is_all_cycle_label(selected_week) else "Average Score", f"{selected.score}/100")
    metric4.metric("Avg Complexity", f"{selected.static_analysis.average_complexity:.2f}")
    metric5.metric("Max Nesting", selected.structural_analysis.max_nesting_depth)
    metric6.metric("Functions", selected.structural_analysis.function_count)
    metric7.metric("Classes", selected.structural_analysis.class_count)

    top_left, top_right = st.columns([1.15, 0.85])
    with top_left:
        st.markdown("**Weighted Scoring Parameters**")
        bucket_display = bucket_table[["Bucket", "Display", "Rating", "Raw Score"]].rename(columns={"Display": "Score"})
        st.dataframe(bucket_display, use_container_width=True, hide_index=True)
    with top_right:
        radar_frame = pd.DataFrame(
            {
                "Bucket": list(selected.score_breakdown.keys()) + [next(iter(selected.score_breakdown.keys()), "Code Quality")],
                "Score": list(selected.score_breakdown.values()) + [next(iter(selected.score_breakdown.values()), 0.0)],
            }
        )
        if not radar_frame.empty:
            radar_chart = px.line_polar(
                radar_frame,
                r="Score",
                theta="Bucket",
                line_close=True,
                range_r=[0, 100],
                title="Scoring Radar",
            )
            radar_chart.update_traces(fill="toself")
            radar_chart.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(radar_chart, use_container_width=True)

    with st.expander("Repository Structure", expanded=True):
        structure_left, structure_right = st.columns([1.05, 0.95])
        with structure_left:
            st.code(selected.file_tree or "No source files discovered.", language="text")
        with structure_right:
            st.markdown("**Structure Reviewed**")
            if selected.structural_analysis.detected_languages:
                st.write(f"- Languages detected: {', '.join(selected.structural_analysis.detected_languages)}")
            if selected.structural_analysis.detected_frameworks:
                st.write(f"- Frameworks detected: {', '.join(selected.structural_analysis.detected_frameworks)}")
            st.write(f"- Functions reviewed: {selected.structural_analysis.function_count}")
            st.write(f"- Classes reviewed: {selected.structural_analysis.class_count}")
            st.write(f"- Methods reviewed: {selected.structural_analysis.method_count}")
            st.write(f"- Max nesting depth: {selected.structural_analysis.max_nesting_depth}")
            st.write(f"- Modularity score: {selected.structural_analysis.modularity_score}")
            st.write(f"- Docstring ratio: {selected.structural_analysis.docstring_ratio:.2f}")
            st.write(f"- Python files reviewed: {selected.structural_analysis.python_files_analyzed}")
            st.write(f"- Java files reviewed: {selected.structural_analysis.java_files_analyzed}")
            st.write(f"- Other files reviewed: {selected.structural_analysis.other_files_analyzed}")
            st.markdown("**Library Usage**")
            if selected.structural_analysis.unique_libraries:
                for library in selected.structural_analysis.unique_libraries[:30]:
                    st.write(f"- {library}")
            else:
                st.write("No imports detected.")

    with st.expander("Strengths", expanded=True):
        for item in selected.strengths or ["No strengths returned."]:
            st.write(f"- {item}")

    with st.expander("Areas for Improvement", expanded=True):
        st.markdown("**Weaknesses**")
        for item in selected.weaknesses or ["No weaknesses returned."]:
            st.write(f"- {item}")

    with st.expander("Code Snapshot", expanded=expand_snapshots):
        file_options = [repo_file.path for repo_file in selected.reviewed_files]
        if file_options:
            default_index = file_options.index(selected.primary_file) if selected.primary_file in file_options else 0
            chosen_file = st.selectbox("Reviewed File", file_options, index=default_index)
            chosen = next(repo_file for repo_file in selected.reviewed_files if repo_file.path == chosen_file)
            st.caption(f"{chosen.path} | {chosen.language} | {chosen.size} bytes")
            code_language = {
                "Python": "python",
                "Java": "java",
                "JavaScript": "javascript",
                "TypeScript": "typescript",
                "C#": "csharp",
                "Robot Framework": "text",
            }.get(chosen.language, "text")
            st.code(chosen.content[:12000] or "No code snapshot available.", language=code_language)
        else:
            st.code("No code snapshot available.", language="text")

    with st.expander("Scoring Breakdown", expanded=False):
        render_chart_note("How to read: each bar shows the weighted points awarded to this repository in that review bucket. Higher bars indicate stronger performance in that dimension.")
        breakdown_chart = px.bar(
            bucket_table,
            x="Bucket",
            y="Score Awarded",
            color="Rating",
            text="Display",
            title="Scoring Bucket Contribution",
        )
        breakdown_chart.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis_title="Review Bucket",
            yaxis_title="Weighted Points Awarded",
        )
        st.plotly_chart(breakdown_chart, use_container_width=True)
        metrics_left, metrics_right = st.columns(2)
        with metrics_left:
            st.markdown("**Static Metrics**")
            st.json(asdict(selected.static_analysis))
        with metrics_right:
            st.markdown("**Structural Metrics**")
            st.json(asdict(selected.structural_analysis))

    with st.expander("Evaluation Payload", expanded=False):
        st.markdown("**Evaluation Output**")
        st.json(selected.output)


def render_batch_analytics(results: List[EvaluationResult]) -> None:
    render_section_intro(
        "Batch Analytics",
        "Review cohort-wide trends, weakest review buckets, and trainees who need targeted coaching.",
    )
    evaluated_results = [item for item in results if is_evaluated_result(item)]
    blocked_results = [item for item in results if is_retryable_result(item)]
    experience_profile = next((item.experience_profile for item in evaluated_results), st.session_state.get("task1_experience_profile", "Fresher"))

    if not evaluated_results:
        st.warning("No repositories were fully evaluated yet. Fix the repository/link issues listed below and rerun.")
    else:
        render_score_guide(evaluated_results)
    frame = pd.DataFrame(
        [
            {
                "Trainee": item.trainee,
                "Repository": item.repository,
                "Score": item.score,
                "Classification": item.classification,
                "Classification Display": workbook_preview_term(item.classification),
                "FilesAnalyzed": item.files_analyzed,
                "Complexity": round(item.static_analysis.average_complexity, 2),
                "Lowest Bucket": min(item.score_breakdown, key=item.score_breakdown.get) if item.score_breakdown else "N/A",
                "Primary Gap": item.primary_gap,
            }
            for item in evaluated_results
        ]
    )
    bucket_rows = []
    for item in evaluated_results:
        for bucket, score in item.score_breakdown.items():
            bucket_rows.append(
                {
                    "Trainee": item.trainee,
                    "Repository": item.repository,
                    "Bucket": bucket,
                    "RawScore": score,
                    "ScoreAwarded": round((score / 100.0) * SCORE_WEIGHTS[bucket], 2),
                    "Rating": bucket_rating(score),
                }
            )
    bucket_frame = pd.DataFrame(bucket_rows)
    if not frame.empty:
        top_left, top_right = st.columns(2)
        with top_left:
            render_chart_note("How to read: this donut shows the share of evaluated students in each score band. Use it to understand cohort distribution at a glance.")
            counts = frame.groupby("Classification Display", as_index=False).size()
            pie = px.pie(
                counts,
                names="Classification Display",
                values="size",
                hole=0.45,
                color="Classification Display",
                title="Classification Distribution",
            )
            pie.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20), legend_title_text="")
            st.plotly_chart(pie, use_container_width=True)
        with top_right:
            render_chart_note("How to read: bars show each trainee's final engine score. The horizontal bands represent the active score thresholds for the selected scoring profile.")
            score_bar = px.bar(
                frame.sort_values("Score", ascending=False),
                x="Trainee",
                y="Score",
                color="Classification Display",
                text="Score",
                title="Trainee Score Overview",
            )
            add_score_bands(score_bar, experience_profile)
            score_bar.update_layout(
                height=350,
                margin=dict(l=20, r=20, t=50, b=20),
                xaxis_title="Trainee",
                yaxis_title="Final Engine Score",
            )
            st.plotly_chart(score_bar, use_container_width=True)

        if not bucket_frame.empty:
            render_chart_note("How to read: each bar shows the cohort average weighted points in a review bucket. Lower bars identify the weakest shared competency areas.")
            avg_bucket = bucket_frame.groupby("Bucket", as_index=False)["ScoreAwarded"].mean().sort_values("ScoreAwarded")
            avg_chart = px.bar(
                avg_bucket,
                x="Bucket",
                y="ScoreAwarded",
                color="Bucket",
                title="Average Score by Review Bucket",
                text_auto=".2f",
            )
            avg_chart.update_layout(
                height=350,
                margin=dict(l=20, r=20, t=50, b=20),
                showlegend=False,
                xaxis_title="Review Bucket",
                yaxis_title="Average Weighted Points",
            )
            st.plotly_chart(avg_chart, use_container_width=True)

        trainer_left, trainer_right = st.columns([1.05, 0.95])
        with trainer_left:
            st.markdown("**Students Requiring Trainer Attention**")
            action_frame = frame[frame["Classification"].isin(["Needs Attention", "Retest Required"])].copy()
            if action_frame.empty:
                st.success("No students currently fall into the scope-for-improvement or reassessment-required buckets.")
            else:
                st.dataframe(
                    action_frame[["Trainee", "Score", "Classification Display", "Lowest Bucket", "Primary Gap"]].rename(
                        columns={"Classification Display": "Classification"}
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
        with trainer_right:
            if not bucket_frame.empty:
                render_chart_note("How to read: for each review bucket, this chart shows the lowest weighted score awarded to any trainee. It highlights the deepest single-point coaching need.")
                weak_bucket_frame = (
                    bucket_frame.sort_values(["Bucket", "ScoreAwarded", "Trainee"])
                    .groupby("Bucket", as_index=False)
                    .first()
                    .rename(columns={"Trainee": "Weakest Trainee", "ScoreAwarded": "Lowest Awarded Score"})
                )
                weakest_chart = px.bar(
                    weak_bucket_frame,
                    x="Bucket",
                    y="Lowest Awarded Score",
                    color="Weakest Trainee",
                    title="Lowest Performing Trainee by Bucket",
                    text_auto=".2f",
                )
                weakest_chart.update_layout(
                    height=350,
                    margin=dict(l=20, r=20, t=50, b=20),
                    xaxis_title="Review Bucket",
                    yaxis_title="Lowest Weighted Points",
                )
                st.plotly_chart(weakest_chart, use_container_width=True)

    if blocked_results:
        with st.expander("Repository Issues Requiring Trainer Fix", expanded=False):
            issue_rows = pd.DataFrame(
                [
                    {
                        "Trainee": item.trainee,
                        "GitHub Cycle": get_result_week(item),
                        "Repository": item.repository,
                        "Status": item.status,
                        "Issue Type": item.issue_type,
                        "Issue": item.issue_message,
                    }
                    for item in blocked_results
                ]
            )
            st.dataframe(issue_rows, use_container_width=True, hide_index=True)


def render_sidebar(disabled: bool = False) -> Dict[str, str]:
    st.sidebar.header("Configuration")
    github_token = st.sidebar.text_input(
        "GitHub Token",
        type="password",
        key="task1_github_token",
        disabled=disabled,
        help="Only needed when evaluating GitHub repositories, especially private repos or higher API limits.",
    )
    provider = st.sidebar.selectbox("AI Provider", ["", "groq", "openrouter", "huggingface"], key="task1_ai_provider", disabled=disabled)
    validation_profile = st.sidebar.radio(
        "Validation Rules",
        list(VALIDATION_RULE_PROFILES.keys()),
        index=0,
        key="task1_validation_profile",
        disabled=disabled,
    )

    default_models = {
        "": "",
        "groq": "llama-3.3-70b-versatile",
        "openrouter": "openai/gpt-4o-mini",
        "huggingface": "mistralai/Mistral-7B-Instruct-v0.3",
    }
    model = st.sidebar.text_input("AI Model", value=default_models.get(provider, ""), key="task1_ai_model", disabled=disabled)
    api_key = st.sidebar.text_input("AI API Key", type="password", key="task1_ai_api_key", disabled=disabled)

    st.sidebar.markdown("**Security Controls**")
    st.sidebar.write(f"- Max file size: {MAX_FILE_SIZE_BYTES // 1000} KB")
    st.sidebar.write(f"- Max notebook size: {MAX_NOTEBOOK_SIZE_BYTES // 1000} KB")
    st.sidebar.write(f"- Max files per repository: {MAX_FILES_PER_REPOSITORY}")
    st.sidebar.write("- External code execution disabled")
    st.sidebar.write(f"- Source files currently read: {SUPPORTED_STACKS_MESSAGE}")
    st.sidebar.markdown("**Active Validation Rules**")
    profile_rules = VALIDATION_RULE_PROFILES[validation_profile]
    st.sidebar.write(f"- Blank GitHub links: {'Skip row' if profile_rules['skip_blank_links'] else 'Error'}")
    st.sidebar.write(f"- Encoded GitHub folder paths: {'Decode automatically' if profile_rules['decode_github_subpaths'] else 'Literal match'}")
    st.sidebar.write("- Invalid links / missing code: mark as Not Evaluated")
    st.sidebar.write("- Failed evaluations are excluded from cohort averages")

    return {
        "github_token": github_token.strip(),
        "provider": provider.strip(),
        "model": model.strip(),
        "api_key": api_key.strip(),
        "validation_profile": validation_profile,
    }


def ensure_task1_state() -> None:
    defaults: Dict[str, Any] = {
        "task1_evaluation_in_progress": False,
        "task1_pending_evaluation": False,
        "task1_last_error": "",
        "task1_manual_candidates": [],
        "task1_manual_overrides": {},
        "task1_missing_link_selection": None,
        "task1_manual_link_input": "",
        "task1_issue_recovery_selection": None,
        "task1_issue_recovery_source": "",
        "task1_repo_text": "",
        "task1_assessment_excel_path": "",
        "task1_assignment_excel_path": "",
        "task1_input_mode": "Manual Entry",
        "task1_experience_profile": "Fresher",
        "task1_review_strictness": 100,
        "task1_source_mode_label": "Auto Detect",
        "task1_upload_nonce": 0,
        "task1_result_filter": "All",
        "task1_college_filter": "All Colleges",
        "task1_week_filter": ALL_ASSESSMENTS_LABEL,
        "task1_track_filter": ALL_GITHUB_TRACKS_LABEL,
        "task1_deep_audit_week_filter": "",
        "task1_deep_audit_detail_week_filter": "",
        "task1_workbook_export_contexts": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_task1_state(clear_inputs: bool = True) -> None:
    upload_nonce = st.session_state.get("task1_upload_nonce", 0)
    upload_key = f"task1_uploaded_excel_{upload_nonce}"
    log_task1_event(
        "task1_state_reset",
        clear_inputs=clear_inputs,
        had_results=len(st.session_state.get("evaluation_results", [])),
        had_pending=bool(st.session_state.get("task1_pending_evaluation", False)),
        had_in_progress=bool(st.session_state.get("task1_evaluation_in_progress", False)),
    )
    for key in [
        "evaluation_results",
        "task1_shared_eval_results",
        "workbook_preview",
        "workbook_import_notices",
        "task1_pending_evaluation",
        "task1_evaluation_in_progress",
        "task1_last_error",
        "task1_manual_candidates",
        "task1_manual_overrides",
        "task1_missing_link_selection",
        "task1_manual_link_input",
        "task1_issue_recovery_selection",
        "task1_issue_recovery_source",
        "task1_result_filter",
        "task1_college_filter",
        "task1_week_filter",
        "task1_track_filter",
        "task1_deep_audit_week_filter",
        "task1_deep_audit_detail_week_filter",
        "task1_workbook_export_contexts",
    ]:
        st.session_state.pop(key, None)
    if clear_inputs:
        st.session_state["task1_repo_text"] = ""
        st.session_state["task1_assessment_excel_path"] = ""
        st.session_state["task1_assignment_excel_path"] = ""
        st.session_state["task1_input_mode"] = "Manual Entry"
        st.session_state["task1_experience_profile"] = "Fresher"
        st.session_state["task1_review_strictness"] = 100
        st.session_state["task1_source_mode_label"] = "Auto Detect"
        st.session_state["task1_upload_nonce"] = upload_nonce + 1
        st.session_state["task1_result_filter"] = "All"
        st.session_state["task1_college_filter"] = "All Colleges"
        st.session_state["task1_week_filter"] = ALL_ASSESSMENTS_LABEL
        st.session_state["task1_track_filter"] = ALL_GITHUB_TRACKS_LABEL
        st.session_state["task1_deep_audit_week_filter"] = ""
        st.session_state["task1_deep_audit_detail_week_filter"] = ""
        st.session_state.pop(f"{upload_key}_assessment", None)
        st.session_state.pop(f"{upload_key}_assignment", None)


def reset_task2_state() -> None:
    for key in [
        "tracker_data",
        "eval_results",
        "assessment_notices",
        "table_filter",
        "current_file_selection",
        "current_batch_selection",
        "current_week_selection",
        "nlp_candidate_view",
        "show_append_uploader",
        "student_360_selected_student",
    ]:
        st.session_state.pop(key, None)


def reset_task3_state() -> None:
    for key in [
        "att_data",
        "weekly_data",
        "ind_weekly_data",
        "roster_filter",
        "scroll_to_roster",
        "selected_week_drilldown",
        "num_uploaders",
    ]:
        st.session_state.pop(key, None)


def reset_task4_state() -> None:
    for key in [
        "task4_college",
        "task4_batch",
    ]:
        st.session_state.pop(key, None)


def reset_task5_state() -> None:
    for key in [
        "task5_college",
        "task5_batch",
        "task5_population_scope",
        "task5_quadrant_filter",
    ]:
        st.session_state.pop(key, None)


def reset_task6_state() -> None:
    for key in [
        "task6_current_file_selection",
        "task6_current_batch_selection",
        "task6_current_week_selection",
        "task6_table_filter",
        "task6_selected_student",
        "task6_show_append_uploader",
    ]:
        st.session_state.pop(key, None)


def reset_task7_state() -> None:
    for key in [
        "task7_current_batch_selection",
        "task7_current_week_selection",
        "student_360_selected_student",
    ]:
        st.session_state.pop(key, None)


def reset_current_module_state(selected_module: str) -> None:
    if selected_module == "GitHub Evaluation Engine":
        reset_task1_state(clear_inputs=True)
    elif selected_module == "Assessment & Feedback Analytics":
        reset_task2_state()
    elif selected_module == "Executive Attendance Intelligence":
        reset_task3_state()
    elif selected_module == "Top Brains Performance":
        reset_task6_state()
    elif selected_module == "Performance Magic Quadrant":
        reset_task4_state()
    else:
        reset_task5_state()


def reset_all_portal_state() -> None:
    reset_task1_state(clear_inputs=True)
    reset_task2_state()
    reset_task3_state()
    reset_task4_state()
    reset_task5_state()
    reset_task6_state()
    reset_task7_state()


def render_portal_reset_toolbar(selected_module: str) -> None:
    current_label = "Reset Current View" if selected_module in {"Performance Magic Quadrant", "Performance Magic Quadrant Plus"} else "Reset Current Module"
    with st.container(border=True):
        info_col, current_col, all_col = st.columns([1.8, 0.9, 1.0])
        with info_col:
            st.markdown("**Portal Reset Controls**")
            st.caption(
                "Current module reset clears only the active workspace state. "
                "Portal reset clears Task-1, Task-2, Task-3, and executive filter state together."
            )
        with current_col:
            if st.button(current_label, key=f"portal_reset_current_{selected_module}", use_container_width=True):
                reset_current_module_state(selected_module)
                st.rerun()
        with all_col:
            if st.button("Reset All Portal Data", key="portal_reset_all", use_container_width=True):
                reset_all_portal_state()
                st.rerun()


def sort_evaluation_results(results: List[EvaluationResult]) -> List[EvaluationResult]:
    return sorted(
        results,
        key=lambda item: (
            not is_evaluated_result(item),
            item.trainee.lower(),
            get_result_track(item),
            assessment_week_sort_key(get_result_week(item)),
            -item.score,
            item.repository.lower(),
        ),
    )


def render_missing_link_recovery(
    manual_candidates: List[Dict[str, str]],
    controls_disabled: bool,
    show_heading: bool = True,
) -> None:
    overrides = st.session_state.get("task1_manual_overrides", {})
    unresolved_candidates = [candidate for candidate in manual_candidates if candidate["submission_id"] not in overrides]

    if not manual_candidates:
        return

    if show_heading:
        render_section_intro(
            "Missing GitHub Link Recovery",
            "Use this only for workbook rows where the GitHub link was blank or missing. Save the corrected link here and it will be included in the next evaluation run.",
        )
    else:
        st.markdown("**Missing or Blank GitHub Links**")
        st.caption("These workbook rows were skipped because the GitHub link cell was blank or missing. Save a corrected link and include it in the next evaluation run.")

    if unresolved_candidates:
        option_labels = {
            " | ".join(
                segment
                for segment in [
                    candidate["trainee"],
                    candidate.get("assessment_week", ""),
                    candidate.get("declared_technology", ""),
                    candidate["worksheet"],
                    f"Row {candidate['row_number']}",
                ]
                if segment
            ): candidate["submission_id"]
            for candidate in unresolved_candidates
        }
        selected_label = st.selectbox(
            "Rows with no GitHub link",
            list(option_labels.keys()),
            key="task1_missing_link_selection",
            disabled=controls_disabled,
        )
        manual_link = st.text_input(
            "Manual GitHub Link",
            key="task1_manual_link_input",
            placeholder="https://github.com/owner/repository",
            disabled=controls_disabled,
        ).strip()
        if st.button("Save Manual Link", use_container_width=True, disabled=controls_disabled):
            if not manual_link:
                st.error("Enter a GitHub repository or folder URL before saving the manual link.")
            else:
                try:
                    parse_github_reference(manual_link)
                except Exception as exc:
                    st.error(f"Invalid GitHub link: {exc}")
                else:
                    candidate = next(item for item in unresolved_candidates if item["submission_id"] == option_labels[selected_label])
                    overrides[candidate["submission_id"]] = {
                        "submission_id": candidate["submission_id"],
                        "trainee": candidate["trainee"],
                        "source": manual_link,
                        "worksheet": candidate.get("worksheet", "Workbook"),
                        "superset_id": candidate.get("superset_id", ""),
                        "declared_technology": candidate.get("declared_technology", ""),
                        "college": candidate.get("college", ""),
                        "source_file": candidate.get("source_file", ""),
                        "assessment_week": candidate.get("assessment_week", "Week 1"),
                        "submission_status": "Pending Evaluation",
                        "evaluation_track": candidate.get("evaluation_track", ASSESSMENT_SCORE_TRACK),
                    }
                    st.session_state["task1_manual_overrides"] = overrides
                    st.session_state["task1_manual_link_input"] = ""
                    st.success(f"Saved manual GitHub link for {candidate['trainee']}.")
                    st.rerun()
    else:
        st.success("All workbook rows with missing GitHub links already have manual overrides saved.")

    if overrides:
        saved_override_frame = pd.DataFrame(
            [
                {
                    "Trainee": value["trainee"],
                    "GitHub Cycle": value.get("assessment_week", "Week 1"),
                    "Evaluation Track": value.get("evaluation_track", ASSESSMENT_SCORE_TRACK),
                    "College": value.get("college", "") or "Unspecified College",
                    "Worksheet": value.get("worksheet", "Workbook"),
                    "Declared Technology": value.get("declared_technology", "") or "Not Declared",
                    "Manual GitHub Link": value["source"],
                }
                for value in overrides.values()
            ]
        )
        st.markdown("**Saved Manual Overrides**")
        st.dataframe(saved_override_frame, use_container_width=True, hide_index=True)


def build_task1_exception_overview(
    manual_candidates: List[Dict[str, str]],
    results: List[EvaluationResult],
) -> pd.DataFrame:
    overrides = st.session_state.get("task1_manual_overrides", {})
    unresolved_candidates = [candidate for candidate in manual_candidates if candidate["submission_id"] not in overrides]
    rows: List[Dict[str, Any]] = []

    for candidate in unresolved_candidates:
        rows.append(
            {
                "Exception Category": "Missing GitHub Link",
                "Trainee": candidate.get("trainee", ""),
                "Superset ID": candidate.get("superset_id", ""),
                "Evaluation Track": candidate.get("evaluation_track", ASSESSMENT_SCORE_TRACK),
                "GitHub Cycle": candidate.get("assessment_week", ""),
                "Worksheet": candidate.get("worksheet", "Workbook"),
                "Declared Tech": candidate.get("declared_technology", "") or "Not Declared",
                "Repository": "",
                "Issue Type": "Input Missing",
                "Issue": candidate.get("reason", "Missing GitHub link in workbook row."),
                "Next Action": "Save manual GitHub link",
            }
        )

    for item in filter_exception_results(results, "Not Evaluated"):
        rows.append(
            {
                "Exception Category": "Repository Evaluation Failed",
                "Trainee": item.trainee,
                "Superset ID": get_result_superset_id(item),
                "Evaluation Track": get_result_track(item),
                "GitHub Cycle": get_result_week(item),
                "Worksheet": item.worksheet or "Manual Entry",
                "Declared Tech": item.declared_technology or "Not Declared",
                "Repository": item.repository,
                "Issue Type": item.issue_type or "Evaluation Error",
                "Issue": item.issue_message,
                "Next Action": "Retry with corrected repository",
            }
        )

    return pd.DataFrame(rows)


def render_workbook_preview(
    results: List[EvaluationResult],
    raw_results: Optional[List[EvaluationResult]] = None,
    workbook_preview: Optional[pd.DataFrame] = None,
    selected_week: str = ALL_WEEKS_LABEL,
) -> None:
    render_section_intro(
        "Workbook Preview",
        "Review cohort KPIs and trainee-level scoring before drilling into repository evidence.",
    )
    render_summary(results, raw_results, selected_week)
    render_score_guide(
        results,
        needs_attention_label="Scope for Improvement",
        retest_required_label="Reassessment Required",
    )
    selected_filter = st.session_state.get("task1_result_filter", "All")
    filtered_results = filter_task1_results(results, selected_filter, raw_results, selected_week)
    is_exception_detail_view = is_all_cycle_label(selected_week) and selected_filter in SPECIAL_RESULT_STATUSES.union({"Not Evaluated"})

    table_heading = "Exception Detail Table" if is_exception_detail_view else "Student Scoring Table"
    st.markdown(f"**{table_heading}**  \nCurrent filter: `{workbook_preview_term(selected_filter)}`")
    if is_all_cycle_label(selected_week) and selected_filter == "Not Evaluated":
        st.caption("This table shows the exact cycle-level rows that were not evaluated, so you can see which GitHub cycle data was missing or failed.")
    render_leaderboard(filtered_results)

    with st.expander("Workbook Source Preview", expanded=False):
        if workbook_preview is not None and not workbook_preview.empty:
            st.dataframe(
                build_workbook_source_preview(
                    workbook_preview,
                    selected_week,
                    st.session_state.get("task1_track_filter", ALL_GITHUB_TRACKS_LABEL),
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.markdown('<div class="small-note">No workbook preview available for the current evaluation run.</div>', unsafe_allow_html=True)


def build_export_frame(results: List[EvaluationResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Trainee": item.trainee,
                "SupersetID": get_result_superset_id(item),
                "AssessmentWeek": get_result_week(item),
                "EvaluationTrack": get_result_track(item),
                "College": get_result_college(item),
                "Worksheet": item.worksheet,
                "DeclaredTechnology": item.declared_technology,
                "LanguagesTested": item.languages_tested,
                "DetectedFrameworks": ", ".join(item.structural_analysis.detected_frameworks) if item.structural_analysis.detected_frameworks else "",
                "SourceFile": getattr(item, "source_file", ""),
                "Repository": item.repository,
                "Score": item.score if is_evaluated_result(item) else None,
                "Classification": item.classification,
                "Status": item.status,
                "IssueType": item.issue_type,
                "IssueMessage": item.issue_message,
                "ExperienceProfile": item.experience_profile,
                "ReviewStrictness": item.review_strictness,
                "FilesAnalyzed": item.files_analyzed,
                "PrimaryGap": item.primary_gap,
                **{bucket: item.score_breakdown.get(bucket, 0.0) for bucket in SCORE_WEIGHTS},
            }
            for item in results
        ]
    )


def sanitize_export_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    return cleaned.strip("._") or "candidate"


def build_weighted_parameter_export(results: List[EvaluationResult]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for item in results:
        for bucket, raw_score in item.score_breakdown.items():
            rows.append(
                {
                    "Trainee": item.trainee,
                    "SupersetID": get_result_superset_id(item),
                    "AssessmentWeek": get_result_week(item),
                    "EvaluationTrack": get_result_track(item),
                    "Bucket": bucket,
                    "RawScore": round(raw_score, 2),
                    "ScoreAwarded": round((raw_score / 100.0) * SCORE_WEIGHTS[bucket], 2),
                    "Rating": bucket_rating(raw_score),
                }
            )
    return pd.DataFrame(rows)


def build_deep_audit_export_bundle(raw_results: List[EvaluationResult], display_results: List[EvaluationResult]) -> bytes:
    zip_buffer = BytesIO()
    summary_frame = build_export_frame(display_results)
    raw_frame = build_export_frame(raw_results)
    weighted_frame = build_weighted_parameter_export(raw_results)
    payload_rows = [
            {
                "Trainee": item.trainee,
                "SupersetID": get_result_superset_id(item),
                "AssessmentWeek": get_result_week(item),
                "EvaluationTrack": get_result_track(item),
                "Payload": item.output,
            }
        for item in raw_results
    ]
    audit_rows = [
        {
            "Trainee": item.trainee,
            "SupersetID": get_result_superset_id(item),
            "AssessmentWeek": get_result_week(item),
            "EvaluationTrack": get_result_track(item),
            "College": get_result_college(item),
            "Repository": item.repository,
            "Status": item.status,
            "Classification": item.classification,
            "RepositoryStructure": {
                "FileTree": item.file_tree,
                "PrimaryFile": item.primary_file,
                "DetectedLanguages": item.structural_analysis.detected_languages,
                "DetectedFrameworks": item.structural_analysis.detected_frameworks,
                "FunctionCount": item.structural_analysis.function_count,
                "ClassCount": item.structural_analysis.class_count,
                "MethodCount": item.structural_analysis.method_count,
                "MaxNestingDepth": item.structural_analysis.max_nesting_depth,
                "ModularityScore": item.structural_analysis.modularity_score,
                "DocstringRatio": item.structural_analysis.docstring_ratio,
            },
            "Strengths": item.strengths,
            "AreasOfImprovement": item.weaknesses,
            "ScoringBreakdown": item.score_breakdown,
            "EvaluationPayload": item.output,
        }
        for item in raw_results
    ]

    with ZipFile(zip_buffer, "w") as zip_file:
        zip_file.writestr("summary/current_view_summary.csv", summary_frame.to_csv(index=False))
        zip_file.writestr("summary/week_level_results.csv", raw_frame.to_csv(index=False))
        if not weighted_frame.empty:
            zip_file.writestr("details/weighted_scoring_parameters.csv", weighted_frame.to_csv(index=False))
        zip_file.writestr("details/deep_audit_records.json", json.dumps(audit_rows, indent=2, default=str))
        zip_file.writestr("details/evaluation_payloads.json", json.dumps(payload_rows, indent=2, default=str))

        for item in raw_results:
            if item.primary_snapshot:
                superset_suffix = sanitize_export_filename(get_result_superset_id(item))
                file_name = (
                    f"snapshots/{sanitize_export_filename(item.trainee)}"
                    f"{f'__{superset_suffix}' if superset_suffix else ''}"
                    f"__{sanitize_export_filename(get_result_week(item))}"
                    f"__primary_snapshot.txt"
                )
                zip_file.writestr(file_name, item.primary_snapshot)

    return zip_buffer.getvalue()


def workbook_export_value_for_result(result: EvaluationResult) -> Any:
    if is_evaluated_result(result):
        return int(round(result.score))
    if result.status in SPECIAL_RESULT_STATUSES:
        return result.status
    if result.status == "Not Evaluated":
        return NOT_EVALUATED_CLASSIFICATION
    return ""


def build_scored_workbook_bytes(workbook_contexts: List[Dict[str, Any]], results: List[EvaluationResult]) -> Optional[Tuple[bytes, str]]:
    if not workbook_contexts or len(workbook_contexts) != 1:
        return None

    workbook_context = workbook_contexts[0]
    result_lookup = {
        get_result_submission_id(item): item
        for item in results
        if get_result_submission_id(item)
    }
    if not result_lookup:
        return None

    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        for sheet in workbook_context.get("sheets", []):
            export_frame = sheet.get("dataframe", pd.DataFrame()).copy()
            for submission_id, cell_meta in sheet.get("submission_cells", {}).items():
                mapped_result = result_lookup.get(submission_id)
                if mapped_result is None:
                    continue
                row_index = int(cell_meta.get("row_index", -1))
                column_name = str(cell_meta.get("column_name", "")).strip()
                if row_index < 0 or not column_name or column_name not in export_frame.columns or row_index >= len(export_frame):
                    continue
                export_frame.at[row_index, column_name] = workbook_export_value_for_result(mapped_result)
            export_frame.to_excel(writer, index=False, sheet_name=str(sheet.get("sheet_name", "Sheet1"))[:31] or "Sheet1")

        workbook = writer.book
        for worksheet in workbook.worksheets:
            worksheet.protection.sheet = True
            worksheet.protection.insertColumns = True
            worksheet.protection.insertRows = True
            worksheet.protection.formatColumns = True
            worksheet.protection.formatRows = True
            worksheet.protection.sort = True
            worksheet.protection.autoFilter = True

    file_name = Path(str(workbook_context.get("file_name", "task1_scored_workbook.xlsx"))).stem
    return excel_buffer.getvalue(), f"{sanitize_export_filename(file_name)}__scored.xlsx"


def render_export_tools(
    raw_results: List[EvaluationResult],
    display_results: Optional[List[EvaluationResult]] = None,
    full_session_results: Optional[List[EvaluationResult]] = None,
) -> None:
    export_results = display_results if display_results is not None else raw_results
    export_frame = build_export_frame(export_results)
    csv_bytes = export_frame.to_csv(index=False).encode("utf-8")
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        export_frame.to_excel(writer, index=False, sheet_name="Evaluations")
    excel_bytes = excel_buffer.getvalue()
    deep_audit_zip_bytes = build_deep_audit_export_bundle(raw_results, export_results)
    scored_workbook_payload = build_scored_workbook_bytes(
        st.session_state.get("task1_workbook_export_contexts", []),
        full_session_results if full_session_results is not None else raw_results,
    )

    with st.expander("Evaluation Export Tools", expanded=False):
        action_left, action_right, action_workbook, action_zip, action_toggle = st.columns([0.8, 0.8, 1.1, 1.15, 1.1])
        with action_left:
            st.download_button(
                "Export Results to CSV",
                data=csv_bytes,
                file_name="evaluation_results.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with action_right:
            st.download_button(
                "Export Results to Excel",
                data=excel_bytes,
                file_name="evaluation_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        with action_workbook:
            if scored_workbook_payload is None:
                st.caption("Scored workbook export is available for a single imported workbook after evaluation.")
            else:
                workbook_bytes, workbook_name = scored_workbook_payload
                st.download_button(
                    "Export Scored Workbook",
                    data=workbook_bytes,
                    file_name=workbook_name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
        with action_zip:
            st.download_button(
                "Export Deep Audit Report (ZIP)",
                data=deep_audit_zip_bytes,
                file_name="evaluation_deep_audit_bundle.zip",
                mime="application/zip",
                use_container_width=True,
            )
        with action_toggle:
            expand_snapshots = st.toggle(
                "Expand code snapshots by default",
                value=st.session_state.get("expand_code_snapshots", False),
                help="Controls whether the Code Snapshot panel opens expanded in Deep Audit View.",
            )
            st.session_state["expand_code_snapshots"] = expand_snapshots


def run_dashboard_evaluation(
    input_mode: str,
    submissions: List[Dict[str, str]],
    imported_results: Optional[List[EvaluationResult]],
    repo_text: str,
    config: Dict[str, str],
    source_mode: str,
    review_strictness: int,
    experience_profile: str,
) -> None:
    ai_enabled = bool(config["provider"] and config["model"] and config["api_key"])
    status_placeholder = st.empty()
    progress_bar = st.progress(0.0, text="Preparing evaluation run...")
    log_task1_event(
        "task1_run_start",
        input_mode=input_mode,
        source_mode=source_mode,
        review_strictness=review_strictness,
        experience_profile=experience_profile,
        submission_count=len(submissions),
        manual_repo_count=len([line.strip() for line in repo_text.splitlines() if line.strip()]) if repo_text else 0,
        ai_enabled=ai_enabled,
    )

    def run_queue(
        items: List[Any],
        label_builder: Callable[[Any], str],
        evaluator: Callable[[Any], EvaluationResult],
        completed_before: int = 0,
        overall_total: Optional[int] = None,
        status_prefix: str = "",
    ) -> List[EvaluationResult]:
        results: List[EvaluationResult] = []
        total = overall_total or max(len(items), 1)
        for index, item in enumerate(items, start=1):
            current_index = completed_before + index
            prefix = f"{status_prefix} | " if status_prefix else ""
            item_label = label_builder(item)
            status_placeholder.info(f"{prefix}Reviewing {current_index} of {total}: {item_label}")
            if isinstance(item, dict):
                log_task1_event(
                    "task1_item_start",
                    progress=f"{current_index}/{total}",
                    status_prefix=status_prefix,
                    trainee=item.get("trainee", ""),
                    track=item.get("evaluation_track", ASSESSMENT_SCORE_TRACK),
                    cycle=item.get("assessment_week", ""),
                    worksheet=item.get("worksheet", ""),
                    source=item.get("source", ""),
                )
            else:
                log_task1_event(
                    "task1_item_start",
                    progress=f"{current_index}/{total}",
                    status_prefix=status_prefix,
                    source=str(item),
                )
            try:
                evaluated_result = evaluator(item)
                if isinstance(item, dict) and item.get("submission_id") and isinstance(evaluated_result.output, dict):
                    evaluated_result.output["SubmissionID"] = item.get("submission_id", "")
                results.append(evaluated_result)
                log_task1_event(
                    "task1_item_complete",
                    progress=f"{current_index}/{total}",
                    status_prefix=status_prefix,
                    trainee=evaluated_result.trainee,
                    track=get_result_track(evaluated_result),
                    cycle=get_result_week(evaluated_result),
                    status=evaluated_result.status,
                    score=evaluated_result.score if is_evaluated_result(evaluated_result) else None,
                    issue_type=evaluated_result.issue_type,
                )
            except Exception as exc:
                if isinstance(item, dict):
                    trainee_name = str(item.get("trainee", "")).strip() or "Unknown Submission"
                    repository_source = str(item.get("source", "")).strip()
                else:
                    repository_source = str(item).strip()
                    trainee_name = Path(repository_source).name or repository_source.split("/")[-1]
                log_task1_event(
                    "task1_item_exception",
                    progress=f"{current_index}/{total}",
                    status_prefix=status_prefix,
                    trainee=trainee_name,
                    source=repository_source,
                    error=str(exc),
                )
                failed_result = (
                    build_failed_evaluation_result(
                        trainee_name,
                        repository_source,
                        experience_profile,
                        review_strictness,
                        exc,
                        config["validation_profile"],
                        item.get("worksheet", "") if isinstance(item, dict) else "",
                        item.get("declared_technology", "") if isinstance(item, dict) else "",
                        item.get("college", "") if isinstance(item, dict) else "",
                        item.get("superset_id", "") if isinstance(item, dict) else "",
                        item.get("assessment_week", "") if isinstance(item, dict) else "",
                        item.get("source_file", "") if isinstance(item, dict) else "",
                        item.get("evaluation_track", ASSESSMENT_SCORE_TRACK) if isinstance(item, dict) else ASSESSMENT_SCORE_TRACK,
                    )
                )
                if isinstance(item, dict) and item.get("submission_id") and isinstance(failed_result.output, dict):
                    failed_result.output["SubmissionID"] = item.get("submission_id", "")
                results.append(failed_result)
            progress_bar.progress(current_index / total, text=f"Completed {current_index} of {total}")
        return sort_evaluation_results(results)

    if input_mode == "Excel Import":
        historical_results = sort_evaluation_results(imported_results or [])
        if not submissions and not historical_results:
            raise ValueError("No valid submissions were found in the Excel workbook.")
        deduped_submissions = dedupe_workbook_submissions(submissions)
        grouped_submissions = group_workbook_submissions_by_college(deduped_submissions)
        total_submissions = len(deduped_submissions)
        total_colleges = len(grouped_submissions)
        log_task1_event(
            "task1_excel_mode_prepared",
            total_submissions=total_submissions,
            total_colleges=total_colleges,
            batch_size=TASK1_WORKBOOK_BATCH_SIZE,
        )
        progress_bar.progress(
            0.0,
            text=(
                f"Evaluating {total_colleges} college group(s) in internal batch(es) of up to {TASK1_WORKBOOK_BATCH_SIZE} rows with static analysis, structural analysis, and AI review..."
                if ai_enabled
                else f"Evaluating {total_colleges} college group(s) in internal batch(es) of up to {TASK1_WORKBOOK_BATCH_SIZE} rows with static analysis and structural analysis..."
            ),
        )
        aggregated_results: List[EvaluationResult] = list(historical_results)
        st.session_state["evaluation_results"] = sort_evaluation_results(aggregated_results)
        sync_task1_results_to_shared_eval()
        completed_count = 0
        completed_colleges: List[str] = []
        if not deduped_submissions:
            status_placeholder.success(
                f"Loaded {len(historical_results)} previously scored workbook result(s). No new GitHub links required evaluation."
            )
            log_task1_event(
                "task1_run_complete",
                input_mode=input_mode,
                total_results=len(st.session_state.get("evaluation_results", [])),
                total_submissions=0,
                imported_results=len(historical_results),
            )
            return
        for college_index, (college_name, college_items) in enumerate(grouped_submissions, start=1):
            college_batches = max((len(college_items) + TASK1_WORKBOOK_BATCH_SIZE - 1) // TASK1_WORKBOOK_BATCH_SIZE, 1)
            log_task1_event(
                "task1_college_start",
                college_index=college_index,
                total_colleges=total_colleges,
                college_name=college_name,
                submission_count=len(college_items),
                batch_count=college_batches,
            )
            try:
                for batch_number, start_index in enumerate(range(0, len(college_items), TASK1_WORKBOOK_BATCH_SIZE), start=1):
                    batch_items = college_items[start_index : start_index + TASK1_WORKBOOK_BATCH_SIZE]
                    log_task1_event(
                        "task1_batch_start",
                        college_name=college_name,
                        batch_number=batch_number,
                        batch_count=college_batches,
                        batch_size=len(batch_items),
                        completed_before=completed_count + start_index,
                        total_submissions=total_submissions,
                    )
                    batch_results = run_queue(
                        batch_items,
                        lambda item: " | ".join(
                            segment
                            for segment in [
                                item["trainee"],
                                item.get("evaluation_track", ASSESSMENT_SCORE_TRACK),
                                item.get("assessment_week", ""),
                                item.get("declared_technology", ""),
                                item.get("worksheet", ""),
                                item["source"],
                            ]
                            if segment
                        ),
                        lambda item: evaluate_submission(
                            item["trainee"],
                            item["source"],
                            config["github_token"],
                            config["provider"],
                            config["model"],
                            config["api_key"],
                            source_mode,
                            review_strictness,
                            experience_profile,
                            config["validation_profile"],
                            item.get("worksheet", ""),
                            item.get("declared_technology", ""),
                            item.get("college", ""),
                            item.get("superset_id", ""),
                            item.get("assessment_week", "Week 1"),
                            item.get("source_file", ""),
                            item.get("submission_status", "Pending Evaluation"),
                            item.get("evaluation_track", ASSESSMENT_SCORE_TRACK),
                        ),
                        completed_before=completed_count + start_index,
                        overall_total=total_submissions,
                        status_prefix=f"College {college_index} of {total_colleges} | {college_name} | Batch {batch_number} of {college_batches}",
                    )
                    aggregated_results.extend(batch_results)
                    st.session_state["evaluation_results"] = sort_evaluation_results(aggregated_results)
                    sync_task1_results_to_shared_eval()
                    log_task1_event(
                        "task1_batch_complete",
                        college_name=college_name,
                        batch_number=batch_number,
                        stored_results=len(st.session_state.get("evaluation_results", [])),
                    )
            except Exception as exc:
                st.session_state["evaluation_results"] = sort_evaluation_results(aggregated_results)
                sync_task1_results_to_shared_eval()
                completed_text = ", ".join(completed_colleges) if completed_colleges else "none"
                log_task1_event(
                    "task1_college_exception",
                    college_name=college_name,
                    completed_colleges=completed_text,
                    error=str(exc),
                )
                raise RuntimeError(
                    f"College '{college_name}' could not be completed. Results for completed college(s): {completed_text}. "
                    f"Re-upload the failed college data and run again. Root issue: {exc}"
                ) from exc
            completed_count += len(college_items)
            completed_colleges.append(college_name)
            log_task1_event(
                "task1_college_complete",
                college_name=college_name,
                completed_count=completed_count,
                total_submissions=total_submissions,
            )
        status_placeholder.success(
            f"Completed evaluation for {total_submissions} workbook submission(s) across {total_colleges} college group(s)."
        )
        log_task1_event(
            "task1_run_complete",
            input_mode=input_mode,
            total_results=len(st.session_state.get("evaluation_results", [])),
            total_submissions=total_submissions,
            imported_results=len(historical_results),
        )
        return

    repo_inputs = [line.strip() for line in repo_text.splitlines() if line.strip()]
    if not repo_inputs:
        raise ValueError("At least one repository is required.")

    unique_inputs = list(dict.fromkeys(repo_inputs))
    progress_bar.progress(
        0.0,
        text=(
            "Evaluating repositories with static analysis, structural analysis, and AI review..."
            if ai_enabled
            else "Evaluating repositories with static analysis and structural analysis..."
        ),
    )
    st.session_state["evaluation_results"] = run_queue(
        unique_inputs,
        lambda item: item,
        lambda item: evaluate_repository(
            item,
            config["github_token"],
            config["provider"],
            config["model"],
            config["api_key"],
            source_mode,
            review_strictness,
            experience_profile,
            config["validation_profile"],
        ),
    )
    sync_task1_results_to_shared_eval()
    status_placeholder.success(f"Completed evaluation for {len(unique_inputs)} repository source(s).")
    log_task1_event(
        "task1_run_complete",
        input_mode=input_mode,
        total_results=len(st.session_state.get("evaluation_results", [])),
        total_repositories=len(unique_inputs),
    )


def render_task1_dashboard() -> None:
    ensure_task1_state()
    run_in_progress = st.session_state.get("task1_evaluation_in_progress", False)
    has_completed_results = bool(st.session_state.get("evaluation_results", []))
    controls_disabled = run_in_progress or has_completed_results
    last_error = st.session_state.get("task1_last_error", "")
    if last_error:
        st.error(last_error)
    config = render_sidebar(disabled=controls_disabled)
    experience_profile = st.radio(
        "Scoring Profile",
        ["Fresher", "Experienced"],
        horizontal=True,
        key="task1_experience_profile",
        disabled=controls_disabled,
    )
    review_strictness = st.select_slider(
        "Code Review Strictness",
        options=[25, 50, 75, 100],
        key="task1_review_strictness",
        disabled=controls_disabled,
    )
    source_mode_label = st.radio(
        "Source Type",
        ["Auto Detect", "Local Folders", "GitHub Repositories"],
        horizontal=True,
        key="task1_source_mode_label",
        disabled=controls_disabled,
    )
    source_mode = {
        "Auto Detect": "auto",
        "Local Folders": "local",
        "GitHub Repositories": "github",
    }[source_mode_label]

    placeholder = (
        "C:\\path\\to\\project\nC:\\path\\to\\another-project"
        if source_mode == "local"
        else "owner/repo\nhttps://github.com/example/project"
        if source_mode == "github"
        else "C:\\path\\to\\project\nowner/repo"
    )
    help_text = (
        "Enter one local project folder per line."
        if source_mode == "local"
        else "Enter one GitHub repository per line."
        if source_mode == "github"
        else "Enter local project folders or GitHub repositories. Existing local paths are detected automatically."
    )

    input_mode = st.radio(
        "Submission Input",
        ["Manual Entry", "Excel Import"],
        horizontal=True,
        key="task1_input_mode",
        disabled=controls_disabled,
    )
    repo_text = ""
    if input_mode == "Manual Entry":
        repo_text = st.text_area(
            "Project Sources",
            height=180,
            placeholder=placeholder,
            help=help_text,
            key="task1_repo_text",
            disabled=controls_disabled,
        )
    assessment_excel_path = ""
    assignment_excel_path = ""
    uploaded_assessment_excel: Any = None
    uploaded_assignment_excel: Any = None
    preview_frame: Optional[pd.DataFrame] = None
    submissions: List[Dict[str, str]] = []
    imported_results: List[EvaluationResult] = []
    workbook_export_contexts: List[Dict[str, Any]] = st.session_state.get("task1_workbook_export_contexts", [])
    manual_candidates: List[Dict[str, str]] = st.session_state.get("task1_manual_candidates", [])
    import_notices: List[str] = st.session_state.get("workbook_import_notices", [])
    upload_key = f"task1_uploaded_excel_{st.session_state.get('task1_upload_nonce', 0)}"
    assessment_upload_key = f"{upload_key}_assessment"
    assignment_upload_key = f"{upload_key}_assignment"

    if input_mode == "Excel Import":
        assessment_col, assignment_col = st.columns(2)
        with assessment_col:
            st.markdown("**Assessment GitHub Workbook Imports**")
            assessment_excel_path = st.text_input(
                "Assessment GitHub Excel Path",
                key="task1_assessment_excel_path",
                help="Provide an absolute path to a GitHub workbook. Assessment and assignment columns are detected automatically.",
                disabled=controls_disabled,
            ).strip()
            uploaded_assessment_excel = st.file_uploader(
                "Upload GitHub Workbook(s)",
                type=["xlsx", "xls", "xlsm"],
                key=assessment_upload_key,
                accept_multiple_files=True,
                disabled=controls_disabled,
            )
        with assignment_col:
            st.markdown("**Additional GitHub Workbook Imports**")
            assignment_excel_path = st.text_input(
                "Additional GitHub Excel Path",
                key="task1_assignment_excel_path",
                help="Use this when you want to append another workbook. Assessment and assignment columns are detected automatically.",
                disabled=controls_disabled,
            ).strip()
            uploaded_assignment_excel = st.file_uploader(
                "Upload Additional GitHub Workbook(s)",
                type=["xlsx", "xls", "xlsm"],
                key=assignment_upload_key,
                accept_multiple_files=True,
                disabled=controls_disabled,
            )

        with st.expander("Recommended Trainer Workbook Format", expanded=False):
            template_frame = build_submission_template_frame()
            st.dataframe(template_frame, use_container_width=True, hide_index=True)
            st.download_button(
                "Download Template CSV",
                data=template_frame.to_csv(index=False).encode("utf-8"),
                file_name="github_evaluation_template.csv",
                mime="text/csv",
            )
            st.caption(
                "Minimum reliable columns: Superset ID, Candidate Name, and one or more week-specific GitHub link columns "
                "(for example: Assessment - Week 1, Assessment - Week 2, Assignment - First). Optional: Technology / Framework. "
                "Rows with blank GitHub links are skipped for that cycle, not failed. Assessment and assignment columns are detected automatically from the imported workbook."
            )

        try:
            preview_frames: List[pd.DataFrame] = []
            submissions = []
            import_notices = []
            manual_candidates = []
            imported_results = []
            workbook_export_contexts = []

            assessment_source = uploaded_assessment_excel if uploaded_assessment_excel else assessment_excel_path
            assignment_source = uploaded_assignment_excel if uploaded_assignment_excel else assignment_excel_path

            if assessment_source:
                assessment_preview, assessment_submissions, assessment_notices, assessment_manual_candidates, assessment_imported_results, assessment_workbook_contexts = load_submissions_from_excel(
                    assessment_source,
                    evaluation_track=ASSESSMENT_SCORE_TRACK,
                )
                if assessment_preview is not None and not assessment_preview.empty:
                    preview_frames.append(assessment_preview)
                submissions.extend(assessment_submissions)
                import_notices.extend(assessment_notices)
                manual_candidates.extend(assessment_manual_candidates)
                imported_results.extend(assessment_imported_results)
                workbook_export_contexts.extend(assessment_workbook_contexts)

            if assignment_source:
                assignment_preview, assignment_submissions, assignment_notices, assignment_manual_candidates, assignment_imported_results, assignment_workbook_contexts = load_submissions_from_excel(
                    assignment_source,
                    evaluation_track=ASSIGNMENT_SCORE_TRACK,
                )
                if assignment_preview is not None and not assignment_preview.empty:
                    preview_frames.append(assignment_preview)
                submissions.extend(assignment_submissions)
                import_notices.extend(assignment_notices)
                manual_candidates.extend(assignment_manual_candidates)
                imported_results.extend(assignment_imported_results)
                workbook_export_contexts.extend(assignment_workbook_contexts)

            if assessment_source or assignment_source:
                preview_frame = pd.concat(preview_frames, ignore_index=True) if preview_frames else pd.DataFrame()
                st.session_state["workbook_preview"] = preview_frame
                st.session_state["workbook_import_notices"] = import_notices
                st.session_state["task1_manual_candidates"] = manual_candidates
                st.session_state["task1_workbook_export_contexts"] = workbook_export_contexts
                assessment_count = sum(1 for item in submissions if normalize_evaluation_track(item.get("evaluation_track", "")) == ASSESSMENT_SCORE_TRACK)
                assignment_count = sum(1 for item in submissions if normalize_evaluation_track(item.get("evaluation_track", "")) == ASSIGNMENT_SCORE_TRACK)
                imported_cycle_count = len(imported_results)
                st.success("GitHub workbook(s) uploaded successfully. Existing scored cells will be read, and only fresh GitHub links will be evaluated when you press Evaluate Repositories.")
                st.caption(
                    f"Detected {assessment_count} new assessment submission(s), {assignment_count} new assignment submission(s), "
                    f"and {imported_cycle_count} previously scored/status cycle(s) from the imported workbook source(s)."
                )
            else:
                st.session_state["workbook_import_notices"] = []
                st.session_state["task1_manual_candidates"] = []
                st.session_state["task1_workbook_export_contexts"] = []
                import_notices = []
                manual_candidates = []
                imported_results = []
                workbook_export_contexts = []
        except Exception as exc:
            st.session_state["workbook_import_notices"] = [str(exc)]
            st.session_state["task1_manual_candidates"] = []
            st.session_state["task1_workbook_export_contexts"] = []
            st.error(f"Excel import failed: {exc}")

        for notice in import_notices:
            st.warning(notice)

    workbook_submissions = merge_workbook_overrides(submissions) if input_mode == "Excel Import" else submissions
    deduped_workbook_submissions = dedupe_workbook_submissions(workbook_submissions) if input_mode == "Excel Import" else []
    available_colleges = collect_task1_colleges(deduped_workbook_submissions) if input_mode == "Excel Import" else []
    available_tracks = collect_task1_tracks(deduped_workbook_submissions) if input_mode == "Excel Import" else []
    current_track_filter = st.session_state.get("task1_track_filter", ALL_GITHUB_TRACKS_LABEL)
    available_weeks = collect_task1_weeks(deduped_workbook_submissions, current_track_filter) if input_mode == "Excel Import" else []
    if st.session_state.get("task1_college_filter", "All Colleges") not in ["All Colleges"] + available_colleges:
        st.session_state["task1_college_filter"] = "All Colleges"
    if input_mode == "Excel Import" and available_tracks:
        if st.session_state.get("task1_track_filter", "") not in available_tracks:
            st.session_state["task1_track_filter"] = ASSESSMENT_SCORE_TRACK if ASSESSMENT_SCORE_TRACK in available_tracks else available_tracks[0]
        current_track_filter = st.session_state.get("task1_track_filter", available_tracks[0])
        available_weeks = collect_task1_weeks(deduped_workbook_submissions, current_track_filter) if input_mode == "Excel Import" else []
    default_cycle_filter = default_cycle_label_for_track(current_track_filter if current_track_filter != ALL_GITHUB_TRACKS_LABEL else ASSESSMENT_SCORE_TRACK)
    if st.session_state.get("task1_week_filter", default_cycle_filter) not in [default_cycle_filter] + available_weeks:
        st.session_state["task1_week_filter"] = default_cycle_filter
    has_ready_input = bool(repo_text.strip()) if input_mode == "Manual Entry" else bool(
        assessment_excel_path or uploaded_assessment_excel or assignment_excel_path or uploaded_assignment_excel
    )
    can_evaluate = has_ready_input and (input_mode == "Manual Entry" or bool(workbook_submissions) or bool(imported_results))

    if input_mode == "Excel Import" and manual_candidates:
        unresolved_count = len([item for item in manual_candidates if item["submission_id"] not in st.session_state.get("task1_manual_overrides", {})])
        if unresolved_count:
            st.caption(f"{unresolved_count} workbook row(s) still have no GitHub link. They will remain skipped unless you save a manual override.")
    if input_mode == "Excel Import" and deduped_workbook_submissions:
        batch_count = max((len(deduped_workbook_submissions) + TASK1_WORKBOOK_BATCH_SIZE - 1) // TASK1_WORKBOOK_BATCH_SIZE, 1)
        if batch_count > 1:
            st.caption(
                f"This workbook will be processed in {batch_count} internal batch(es) of up to {TASK1_WORKBOOK_BATCH_SIZE} submissions. "
                "Combined results are stored in the current session until you reset the module or the session ends."
            )
    action_columns = st.columns([0.28, 0.18, 0.54])
    with action_columns[0]:
        evaluate_clicked = st.button("Evaluate Repositories", type="primary", disabled=controls_disabled or not can_evaluate, use_container_width=True)
    with action_columns[1]:
        reset_clicked = st.button("Reset Evaluation", disabled=run_in_progress, use_container_width=True)
    with action_columns[2]:
        if run_in_progress:
            st.info("Evaluation is running. Inputs are locked until the current run completes.")
        elif has_completed_results:
            st.info("This evaluation session is locked. Use Reset Evaluation or Reset Current Module to start a new run.")
        elif not has_ready_input:
            st.caption("Provide repository sources or import an Excel workbook to enable evaluation.")
        elif input_mode == "Excel Import" and not workbook_submissions and not imported_results:
            st.caption("No valid workbook submissions are ready yet. Add a manual GitHub link only for the rare blank-link exceptions.")
        else:
            st.caption("The current input set is ready for evaluation.")

    if reset_clicked:
        reset_task1_state(clear_inputs=True)
        st.rerun()

    if evaluate_clicked:
        validation_error = ""
        if input_mode == "Excel Import":
            if not (assessment_excel_path or uploaded_assessment_excel or assignment_excel_path or uploaded_assignment_excel):
                validation_error = "Provide at least one GitHub workbook before running the evaluation."
            elif not workbook_submissions and not imported_results:
                validation_error = "No valid submissions or scored cycles were found in the Excel workbook."
        elif not repo_text.strip():
            validation_error = "At least one repository is required."

        if validation_error:
            st.error(validation_error)
        else:
            st.session_state["task1_last_error"] = ""
            st.session_state["task1_pending_evaluation"] = True
            st.session_state["task1_evaluation_in_progress"] = True
            log_task1_event(
                "task1_run_queued",
                input_mode=input_mode,
                submission_count=len(workbook_submissions),
                manual_repo_count=len([line.strip() for line in repo_text.splitlines() if line.strip()]) if repo_text else 0,
            )
            if preview_frame is not None:
                st.session_state["workbook_preview"] = preview_frame
            st.rerun()

    if st.session_state.get("task1_pending_evaluation", False):
        try:
            run_dashboard_evaluation(
                input_mode,
                workbook_submissions,
                imported_results,
                repo_text,
                config,
                source_mode,
                review_strictness,
                experience_profile,
            )
            if preview_frame is not None:
                st.session_state["workbook_preview"] = preview_frame
        except Exception as exc:
            st.session_state["task1_last_error"] = f"Evaluation failed: {exc}"
            log_task1_event("task1_run_failed", error=str(exc))
        finally:
            st.session_state["task1_pending_evaluation"] = False
            st.session_state["task1_evaluation_in_progress"] = False
            log_task1_event(
                "task1_run_finished",
                has_results=bool(st.session_state.get("evaluation_results")),
                stored_results=len(st.session_state.get("evaluation_results", [])),
                last_error=st.session_state.get("task1_last_error", ""),
            )
        st.rerun()

    results = st.session_state.get("evaluation_results", [])
    workbook_preview = st.session_state.get("workbook_preview")
    if not results:
        render_task1_exception_center(
            manual_candidates,
            [],
            controls_disabled,
            config,
            source_mode,
            review_strictness,
            experience_profile,
        )
        if controls_disabled:
            st.info("Evaluation is in progress. Inputs are locked until the current run completes.")
        else:
            st.info("Provide project sources or import an Excel workbook, then click Evaluate Repositories.")
        return

    result_colleges = sorted({get_result_college(item) for item in results if get_result_college(item)})
    if st.session_state.get("task1_college_filter", "All Colleges") not in ["All Colleges"] + result_colleges:
        st.session_state["task1_college_filter"] = "All Colleges"
    selected_college = st.session_state.get("task1_college_filter", "All Colleges")
    result_tracks = sorted({get_result_track(item) for item in results}, key=lambda value: (value != ASSESSMENT_SCORE_TRACK, value))
    if st.session_state.get("task1_track_filter", "") not in result_tracks:
        st.session_state["task1_track_filter"] = ASSESSMENT_SCORE_TRACK if ASSESSMENT_SCORE_TRACK in result_tracks else (result_tracks[0] if result_tracks else ALL_GITHUB_TRACKS_LABEL)
    selected_track = st.session_state.get("task1_track_filter", ASSESSMENT_SCORE_TRACK)
    result_weeks = sort_week_labels(get_result_week(item) for item in results if get_result_track(item) == selected_track)
    default_cycle_filter = default_cycle_label_for_track(selected_track)
    if st.session_state.get("task1_week_filter", default_cycle_filter) not in [default_cycle_filter] + result_weeks:
        st.session_state["task1_week_filter"] = default_cycle_filter
    if input_mode == "Excel Import":
        filter_columns = st.columns([1.0, 1.0, 1.0])
        with filter_columns[0]:
            if len(result_colleges) > 1:
                st.selectbox(
                    "Select College",
                    ["All Colleges"] + result_colleges,
                    key="task1_college_filter",
                    disabled=run_in_progress,
                )
        with filter_columns[1]:
            if result_tracks:
                st.radio(
                    "Select GitHub Type",
                    result_tracks,
                    key="task1_track_filter",
                    format_func=track_display_label,
                    horizontal=True,
                    disabled=run_in_progress,
                )
        with filter_columns[2]:
            if result_weeks:
                st.selectbox(
                    "Select Cycle",
                    [default_cycle_filter] + result_weeks,
                    key="task1_week_filter",
                    disabled=run_in_progress,
                )

    selected_week = st.session_state.get(
        "task1_week_filter",
        default_cycle_label_for_track(st.session_state.get("task1_track_filter", ASSESSMENT_SCORE_TRACK)),
    )
    scoped_results = filter_results_by_track(filter_results_by_college(results, selected_college), selected_track)
    cycle_filter = selected_week
    week_scoped_raw_results = filter_results_by_assessment_week(scoped_results, cycle_filter)
    display_results = aggregate_candidate_results(scoped_results, cycle_filter)
    scoped_workbook_preview = filter_workbook_preview_by_track(filter_workbook_preview_by_college(workbook_preview, selected_college), selected_track)

    workbook_tab, deep_audit_tab, analytics_tab = st.tabs(["Workbook Preview", "Deep Audit View", "Batch Analytics"])

    with workbook_tab:
        render_workbook_preview(display_results, week_scoped_raw_results, scoped_workbook_preview, cycle_filter)
    with deep_audit_tab:
        render_deep_audit(scoped_results)
    with analytics_tab:
        render_batch_analytics(display_results)

    render_task1_exception_center(
        manual_candidates,
        week_scoped_raw_results,
        controls_disabled,
        config,
        source_mode,
        review_strictness,
        experience_profile,
    )
    render_export_tools(week_scoped_raw_results, display_results, results)


def generate_dashboard() -> None:
    st.set_page_config(page_title="GitHub Evaluation Engine", layout="wide")
    inject_dashboard_styles()
    render_qspiders_banner()
    render_header_banner()
    selected_module = render_module_selector()
    render_portal_reset_toolbar(selected_module)

    if selected_module == "GitHub Evaluation Engine":
        render_task1_dashboard()
    elif selected_module == "Assessment & Feedback Analytics":
        task2_assessment.run()
    elif selected_module == "Executive Attendance Intelligence":
        task3_attendance.run()
    elif selected_module == "Top Brains Performance":
        task6_top_brains.run()
    elif selected_module == "Performance Magic Quadrant":
        task4_correlation.run()
    else:
        task5_correlation_plus.run()


def main() -> None:
    generate_dashboard()


if __name__ == "__main__":
    main()
