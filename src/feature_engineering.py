"""Turn questions.csv into questions_features.csv for TabPFN.

We engineer three families of features:

1. **Structural** — word count, char count, sentence count. Cheap and
   surprisingly informative: "Define X" is short, scenario questions are long.

2. **Content** — regex flags for statute references, years, numbers,
   definition keywords, question marks, and a categorical `question_type`
   (definition / scenario / computation / case_law / other).

3. **Topic keywords** — one binary flag per topic class, firing when the
   question text mentions characteristic terms for that topic (allodial,
   beacon, orthophoto, theodolite, ethics, etc.). This is the strongest
   signal we can give TabPFN for the topic classification task.

The output file is what TabPFN consumes; the target columns (`topic`,
`difficulty_label`) are passed alongside as `y` during training.
"""

from __future__ import annotations

import re

import pandas as pd

from src.config import (
    FEATURES_CSV,
    LABELLED_QUESTIONS_CSV,
    QUESTION_TYPE_MAP,
    SAMPLE_QUESTIONS_CSV,
)
from src.utils import clean_text, ensure_parent, setup_logger

logger = setup_logger("features")


# ============================================================================
# Content regexes
# ============================================================================
STATUTE_RE = re.compile(
    r"\b(act|li|legislative instrument|section|s\.|regulation|guideline|tg)\b",
    re.IGNORECASE,
)
YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")
NUMBER_RE = re.compile(r"\b\d+(?:[\.,]\d+)?\b")
DEFINITION_RE = re.compile(
    r"\b(means|refers to|is defined as|definition of|what is|state the)\b",
    re.IGNORECASE,
)
SCENARIO_RE = re.compile(
    r"\b(scenario|case|a client|suppose|assume|given that|in the event)\b",
    re.IGNORECASE,
)
COMPUTATION_RE = re.compile(
    r"\b(calculate|compute|find|determine the|coordinate|bearing|azimuth|area|distance)\b",
    re.IGNORECASE,
)
CASE_LAW_RE = re.compile(
    r"\b(supreme court|court of appeal|high court|precedent|judgment|versus|v\.)\b",
    re.IGNORECASE,
)


# ============================================================================
# Topic keyword regexes
# ============================================================================
# One compiled regex per topic class. Each fires a binary feature that
# directly correlates with the topic label — the primary signal for the
# topic classifier.
TOPIC_KEYWORDS = {
    "kw_land_law": re.compile(
        r"\b(allodial|usufructuary|stool|skin|customary|freehold|leasehold|"
        r"conveyance|title registration|deed|caveat|easement|act 1036|"
        r"land act|li 1444|act 127|survey act|beacon|boundary mark|"
        r"licensed surveyor|survey regulation)\b",
        re.IGNORECASE,
    ),
    "kw_land_disputes": re.compile(
        r"\b(dispute|plaintiff|defendant|litigation|supreme court|"
        r"court order|composite plan|adverse claim|trespass|"
        r"judgment|boundary conflict|overlapping|claim)\b",
        re.IGNORECASE,
    ),
    "kw_photogrammetry": re.compile(
        r"\b(photogrammetry|stereo|aerial|relief displacement|parallax|"
        r"orthophoto|gsd|gcp|uav|dsm|dtm|nadir|stereopair|flight|"
        r"image|rectification|differential rectification)\b",
        re.IGNORECASE,
    ),
    "kw_instrumentation": re.compile(
        r"\b(instrument|theodolite|total station|calibration|edm|level|"
        r"tripod|measuring band|precision|gnss receiver|optical)\b",
        re.IGNORECASE,
    ),
    "kw_ethics": re.compile(
        r"\b(ethic|integrity|professional conduct|conflict of interest|"
        r"public interest|supervision|responsibility|code of conduct|"
        r"fidelity|confidentiality|impartiality|ethical|disrepute|"
        r"signing|sealing|signature|seal|certification|honest|"
        r"trustworthy|dishonour)\b",
        re.IGNORECASE,
    ),
    "kw_technical": re.compile(
        r"\b(technical guideline|tg-2008|map preparation|plan preparation|"
        r"parcel plan|map scale|gnss|ground control point|accuracy|"
        r"datum|projection|coordinate system|scale)\b",
        re.IGNORECASE,
    ),
}


# ============================================================================
# Helpers
# ============================================================================
def pick_input_csv() -> pd.DataFrame:
    """Return the labelled dataset if present, else the synthetic sample."""
    if LABELLED_QUESTIONS_CSV.exists():
        logger.info("Using labelled dataset: %s", LABELLED_QUESTIONS_CSV)
        return pd.read_csv(LABELLED_QUESTIONS_CSV)
    logger.warning(
        "Labelled dataset not found. Falling back to sample: %s",
        SAMPLE_QUESTIONS_CSV,
    )
    return pd.read_csv(SAMPLE_QUESTIONS_CSV)


def classify_question_type(text: str) -> int:
    """Return an integer code for the type of question."""
    if CASE_LAW_RE.search(text):
        return QUESTION_TYPE_MAP["case_law"]
    if COMPUTATION_RE.search(text):
        return QUESTION_TYPE_MAP["computation"]
    if SCENARIO_RE.search(text):
        return QUESTION_TYPE_MAP["scenario"]
    if DEFINITION_RE.search(text):
        return QUESTION_TYPE_MAP["definition"]
    return QUESTION_TYPE_MAP["other"]


# ============================================================================
# Feature builder
# ============================================================================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the full feature matrix from a questions dataframe."""
    df = df.copy()
    df["question_text"] = df["question_text"].astype(str).map(clean_text)

    # --- Structural features ---
    df["word_count"] = df["question_text"].str.split().str.len().fillna(0).astype(int)
    df["char_count"] = df["question_text"].str.len().fillna(0).astype(int)
    df["sentence_count"] = (
        df["question_text"].str.count(r"[.!?]").fillna(0).astype(int) + 1
    )

    # --- Content features ---
    df["has_statute_reference"] = (
        df["question_text"].str.contains(STATUTE_RE, regex=True).astype(int)
    )
    df["has_year"] = (
        df["question_text"].str.contains(YEAR_RE, regex=True).astype(int)
    )
    df["has_number"] = (
        df["question_text"].str.contains(NUMBER_RE, regex=True).astype(int)
    )
    df["has_definition_keyword"] = (
        df["question_text"].str.contains(DEFINITION_RE, regex=True).astype(int)
    )
    df["has_question_mark"] = (
        df["question_text"].str.contains(r"\?", regex=True).astype(int)
    )
    df["question_type_encoded"] = df["question_text"].map(classify_question_type)

    # --- Topic keyword features ---
    for feat_name, pattern in TOPIC_KEYWORDS.items():
        df[feat_name] = (
            df["question_text"].str.contains(pattern, regex=True).astype(int)
        )

    return df


# ============================================================================
# Entry point
# ============================================================================
def main() -> None:
    df = pick_input_csv()
    logger.info("Loaded %d rows", len(df))
    featured = build_features(df)
    ensure_parent(FEATURES_CSV)
    featured.to_csv(FEATURES_CSV, index=False)
    logger.info("Wrote features to %s", FEATURES_CSV)


if __name__ == "__main__":
    main()