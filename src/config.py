"""Central configuration: paths, topic maps, label encodings."""

from pathlib import Path

# ============ Paths ============
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Inputs / outputs
RAW_QUESTIONS_CSV = INTERIM_DIR / "questions_raw.csv"
LABELLED_QUESTIONS_CSV = PROCESSED_DIR / "questions.csv"
SAMPLE_QUESTIONS_CSV = PROCESSED_DIR / "questions_sample.csv"
FEATURES_CSV = PROCESSED_DIR / "questions_features.csv"

# Models
TOPIC_MODEL_PATH = MODELS_DIR / "topic_model.joblib"
DIFFICULTY_MODEL_PATH = MODELS_DIR / "difficulty_model.joblib"

# ============ Folder → Topic hint ============
# Lower-cased folder names in data/raw/ mapped to a canonical topic label.
# Extraction uses this as a *hint* only; final labels are set by hand in questions.csv.
FOLDER_TOPIC_HINTS = {
    "land law": "Land Law",
    "land dispute": "Land Disputes",
    "land disputes": "Land Disputes",
    "technical guidelines": "Technical Guidelines",
    "ls-course-notes": "Other",
    "ethics": "Ethics",
    "oral interview": "Ethics",
    "photogrammetry": "Photogrammetry",
    "instrumentation": "Instrumentation",
}

# File-name substrings → topic hint (for loose files at the top of raw/)
FILENAME_TOPIC_HINTS = {
    "instrumentation": "Instrumentation",
    "photogrammetry": "Photogrammetry",
    "technical guideline": "Technical Guidelines",
    "land law": "Land Law",
    "land dispute": "Land Disputes",
    "preparatory": "Other",
    "past questions": "Other",
    "financial_management": "Other",
}

# ============ Label space ============
TOPIC_LABELS = [
    "Land Law",
    "Land Disputes",
    "Technical Guidelines",
    "Ethics",
    "Instrumentation",
    "Photogrammetry",
    "Other",
]

DIFFICULTY_LABELS = ["Easy", "Medium", "Hard"]

# ============ Feature configuration ============
FEATURE_COLUMNS = [
    # Structural
    "word_count",
    "char_count",
    "sentence_count",
    # Content
    "has_statute_reference",
    "has_year",
    "has_number",
    "has_definition_keyword",
    "has_question_mark",
    "question_type_encoded",
    # Topic keywords (the strongest signal for the topic classifier)
    "kw_land_law",
    "kw_land_disputes",
    "kw_photogrammetry",
    "kw_instrumentation",
    "kw_ethics",
    "kw_technical",
]

QUESTION_TYPE_MAP = {
    "definition": 0,
    "scenario": 1,
    "computation": 2,
    "case_law": 3,
    "other": 4,
}

# ============ TabPFN ============
TABPFN_N_ESTIMATORS = 8
TABPFN_DEVICE = "cpu"  # switch to "cuda" or "mps" if you have a GPU
TABPFN_RANDOM_STATE = 42

# ============ Extraction heuristics ============
MIN_QUESTION_CHARS = 20
MAX_QUESTION_CHARS = 2000

# Regex that matches the start of a question at the beginning of a line.
# Handles: "Question 1:", "QUESTION 2", "Q3)", "Q.4", "1.", "(1)", "a)", etc.
# NOTE: This regex is compiled with re.IGNORECASE in extract_questions.py, so
#       "QUESTION", "Question", and "question" all match.
QUESTION_START_REGEX = (
    r"(?m)^\s*(?:"
    r"Question\s+\d+(?:\s*[:\.\)])?\s+"   # "Question 1:" OR "Question 1 " (newline)
    r"|Q(?:uestion)?\s*[\.\:\-\)]?\s*\d+\s*[:\.\)]\s*"   # "Q1:" / "Q.2)"
    r"|\(\d+\)\s*"                       # "(1) "
    r"|\d+\s*[\.\)]\s+"                  # "1. " / "2) "
    r"|[a-z]\s*[\)\.]\s+"                # "a) " / "b. "
    r")"
)

# ============ Question quality filter ============
# A block is kept only if it looks like an exam question.
QUESTION_COMMAND_WORDS = (
    "define", "state", "discuss", "explain", "calculate", "list",
    "outline", "describe", "compare", "identify", "name", "give",
    "mention", "what", "why", "how", "when", "where", "which",
    "who", "is", "are", "can", "does", "do", "shall", "may",
    "differentiate", "distinguish", "illustrate", "compute",
    "determine", "find", "solve", "draw", "sketch",
)

# Reject blocks that start with these patterns (legal clauses, section numbers)
NON_QUESTION_START_REGEX = r"^\s*(?:\d+\.\d+|\d+\.\d+\.\d+)\s"  # e.g. "11.1 ..."

# ============ Extraction exclusions ============
# Case-insensitive substrings matched against a file's path relative to
# data/raw/. Any file whose path contains one of these is skipped entirely,
# because its content is statute prose, court party lists, or lecture slides —
# none of which are exam questions.
EXCLUDE_PATH_SUBSTRINGS = (
    # Court judgments — party names, not questions
    "narh  bi",
    "dr_eric_graham",
    # Statutes and regulations — prose, not questions
    "act 127 rev ed",
    "selected sections of land act",
    "selected_sections_of_survey_act_1962_act_127",
    "selected_sections_of_survey_act",
    "survey act.pdf",
    # Lectures and admin material
    "photogrammetry/lecture",
    "lecture 1",
    "lecture 2",
    "lecture 3",
    "time table",
    "financial_management",
    "map and plan making tech guidelines",
)


def ensure_dirs() -> None:
    """Create all required directories if they don't exist."""
    for d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)