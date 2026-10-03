"""
Promote Easy → Medium/Hard using stronger heuristics.

This only moves rows UP the difficulty scale; it never demotes.
Run after `scripts/bootstrap_labels.py` to sharpen the distribution
so the difficulty classifier has enough signal for each class.

Reads and overwrites data/processed/questions.csv.
"""

from __future__ import annotations

import re

import pandas as pd

from src.config import LABELLED_QUESTIONS_CSV


HARD_PATTERNS = (
    r"\bdiscuss\b", r"\bcritically\b", r"\banaly[sz]e\b", r"\bevaluate\b",
    r"\bcompare and contrast\b", r"\bcase study\b",
    r"a client", r"a surveyor proposes", r"suppose ", r"assume ",
    r"given that", r"during retracement", r"you recover",
    r"court order", r"composite plan", r"litigation",
    r"state the investigation", r"outline, in an orderly",
    r"explain, in order", r"what are the consequences",
    r"\bscenario\b", r"what is the .* scenario",
)

MEDIUM_PATTERNS = (
    r"\bexplain\b", r"\bdescribe\b", r"\bwhy\b",
    r"\bhow (does|can|do|is|are|should)\b",
    r"\bdistinguish\b", r"\billustrate\b", r"\bdifferentiate\b",
    r"\bjustify\b", r"\boutline\b",
    r"\bstate the (role|purpose|function|difference|requirements|procedure|reasons|consequences)\b",
    r"\bwhat (are the|is the) (role|purpose|function|difference|requirements|procedure|consequences)\b",
    r"\bwhat does regulation\b",
    r"\bwhat must\b",
    r"\bhow should\b",
    r"\bwhat (are the most|is the core|are the two|is the .* perspective|are the .* questions)\b",
    r"\bstate the .* (rules|standards|values|gates|checks|principles|framework)\b",
    r"\bhow (to|do|can) (handle|respond|deal)\b",
    r"\bis there an ethical\b",
    r"\bdo you have an ethical\b",
)


def promote(text: str, current: str) -> str:
    t = text.lower()
    hard_hits = sum(1 for p in HARD_PATTERNS if re.search(p, t))
    med_hits = sum(1 for p in MEDIUM_PATTERNS if re.search(p, t))
    statutes = len(re.findall(r"act\s+\d+|li\s+\d+", t))
    words = len(t.split())

    # Never demote — trust existing Hard/Medium labels
    if current in ("Hard", "Medium"):
        return current

    if hard_hits >= 1:
        return "Hard"
    if statutes >= 2 and words > 40:
        return "Hard"
    if words > 200:
        return "Hard"

    if med_hits >= 1:
        return "Medium"
    if statutes >= 1 and words > 30:
        return "Medium"
    if words > 80:
        return "Medium"

    return "Easy"


def main() -> None:
    if not LABELLED_QUESTIONS_CSV.exists():
        raise FileNotFoundError(
            f"{LABELLED_QUESTIONS_CSV} not found. "
            f"Run `scripts/bootstrap_labels.py` first."
        )

    df = pd.read_csv(LABELLED_QUESTIONS_CSV)
    before = df["difficulty_label"].value_counts().to_dict()

    df["difficulty_label"] = df.apply(
        lambda r: promote(str(r["question_text"]), str(r["difficulty_label"])),
        axis=1,
    )
    df.to_csv(LABELLED_QUESTIONS_CSV, index=False)

    after = df["difficulty_label"].value_counts().to_dict()
    print("Difficulty BEFORE:")
    for label in ("Easy", "Medium", "Hard"):
        print(f"  {label:<7} {before.get(label, 0)}")
    print("Difficulty AFTER:")
    for label in ("Easy", "Medium", "Hard"):
        print(f"  {label:<7} {after.get(label, 0)}")


if __name__ == "__main__":
    main()
