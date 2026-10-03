"""
Bootstrap questions.csv from questions_raw.csv.

- Reads data/interim/questions_raw.csv
- Infers `topic` from question content (fallback to topic_hint)
- Guesses `difficulty_label` from text heuristics
- Writes data/processed/questions.csv

Run after `python -m src.extract_questions` (or via scripts/run_pipeline.sh).
"""

from __future__ import annotations

import re

import pandas as pd

from src.config import (
    LABELLED_QUESTIONS_CSV,
    RAW_QUESTIONS_CSV,
    TOPIC_LABELS,
)


def infer_topic(text: str, hint: str) -> str:
    """Assign a topic label using content keywords, falling back to the hint."""
    t = text.lower()

    if any(k in t for k in (
        "land dispute", "boundary dispute", "court order", "composite plan",
        "litigation", "supreme court", "judgment", "adverse claim",
        "trespass", "plaintiff", "defendant",
    )):
        return "Land Disputes"

    if any(k in t for k in (
        "photogrammetry", "stereo", "aerial", "relief displacement",
        "parallax", "orthophoto", "gsd", "gcp", "uav", "dsm", "dtm",
        "nadir", "stereopair",
    )):
        return "Photogrammetry"

    if any(k in t for k in (
        "instrument", "theodolite", "total station", "calibration",
        "edm", "level", "tripod", "measuring band",
    )):
        return "Instrumentation"

    if any(k in t for k in (
        "act 1036", "land act", "allodial", "usufructuary", "stool",
        "skin", "customary", "freehold", "leasehold", "conveyance",
        "title registration", "deed", "caveat", "easement",
    )):
        return "Land Law"

    if any(k in t for k in (
        "act 127", "survey act", "li 1444", "regulation", "beacon",
        "licensed surveyor", "boundary mark", "survey regulation",
        "regulation 1",
    )):
        return "Land Law"

    if any(k in t for k in (
        "technical guideline", "tg-2008", "map preparation",
        "plan preparation", "parcel plan", "map scale", "gnss",
        "ground control point",
    )):
        return "Technical Guidelines"

    if any(k in t for k in (
        "ethic", "integrity", "professional conduct",
        "conflict of interest", "public interest", "supervision",
        "responsibility", "code of conduct", "fidelity",
        "confidentiality", "impartiality", "ethical", "disrepute",
        "signing", "sealing", "signature", "seal", "certification",
    )):
        return "Ethics"

    if hint in TOPIC_LABELS:
        return hint
    return "Other"


def guess_difficulty(text: str) -> str:
    """Guess difficulty from question-text heuristics."""
    t = text.lower().strip()

    hard = (
        "scenario", "a client", "suppose", "assume", "given that",
        "advise", "discuss", "critically", "analyze", "analyse",
        "evaluate", "compare and contrast", "case study",
        "explain", "outline", "state the investigation",
        "during retracement", "you recover", "a surveyor proposes",
        "why are", "why is", "why do",
    )
    easy = (
        "define", "what is", "what are", "list ", "name ", "give ",
        "mention ", "identify the", "state five", "state four",
        "state three", "state two", "name four", "name five", "state the",
    )

    statutes = len(re.findall(r"act\s+\d+|li\s+\d+", t))
    words = len(t.split())

    score = (
        sum(1 for m in hard if m in t)
        + (1 if statutes >= 2 else 0)
        + (1 if words > 120 else 0)
        - sum(1 for m in easy if m in t)
    )

    if score >= 2:
        return "Hard"
    if score <= 0:
        return "Easy"
    return "Medium"


def main() -> None:
    if not RAW_QUESTIONS_CSV.exists():
        raise FileNotFoundError(
            f"{RAW_QUESTIONS_CSV} not found. "
            f"Run `python -m src.extract_questions` first."
        )

    df = pd.read_csv(RAW_QUESTIONS_CSV)
    print(f"Loaded {len(df)} raw rows from {RAW_QUESTIONS_CSV.name}")

    for col in ("topic", "difficulty_label", "notes"):
        if col not in df.columns:
            df[col] = ""

    df["topic"] = df.apply(
        lambda r: infer_topic(str(r["question_text"]), str(r.get("topic_hint", ""))),
        axis=1,
    )
    df["difficulty_label"] = df["question_text"].map(
        lambda x: guess_difficulty(str(x))
    )

    out_cols = [
        "question_text", "topic", "difficulty_label",
        "source_file", "topic_hint", "notes",
    ]
    df = df[out_cols]
    df.to_csv(LABELLED_QUESTIONS_CSV, index=False)

    print(f"✅ Wrote {len(df)} bootstrapped rows to {LABELLED_QUESTIONS_CSV.name}")
    print("\nTopic distribution:")
    print(df["topic"].value_counts().to_string())
    print("\nDifficulty distribution:")
    print(df["difficulty_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
