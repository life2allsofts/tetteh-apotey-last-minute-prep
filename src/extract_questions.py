"""Extract questions from raw .docx and .pdf files into questions_raw.csv.

Strategy:
    1. Read every .docx and .pdf under data/raw/ recursively, skipping any
       file whose path matches an entry in EXCLUDE_PATH_SUBSTRINGS.
    2. Split each document's text into candidate blocks using a question-start
       regex (matches "Question 1:", "QUESTION 2", "Q1.", "(1)", "1.", "a)").
    3. Cap very long blocks so answers don't dominate the feature space.
    4. Filter each block through `looks_like_question()` to reject statute
       clauses, lecture notes, and other non-question content.
    5. Deduplicate by normalised question text.
    6. Preserve the folder name as a `topic_hint` so downstream labelling is
       fast (the user confirms or corrects the hint in questions.csv).
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable

from docx import Document
from pypdf import PdfReader

from src.config import (
    EXCLUDE_PATH_SUBSTRINGS,
    FILENAME_TOPIC_HINTS,
    FOLDER_TOPIC_HINTS,
    MAX_QUESTION_CHARS,
    MIN_QUESTION_CHARS,
    NON_QUESTION_START_REGEX,
    QUESTION_COMMAND_WORDS,
    QUESTION_START_REGEX,
    RAW_DIR,
    RAW_QUESTIONS_CSV,
    ensure_dirs,
)
from src.utils import clean_text, ensure_parent, safe_filename, setup_logger

logger = setup_logger("extract")

# ============================================================================
# Compiled regexes (module-level, built once)
# ============================================================================
QUESTION_START_RE = re.compile(QUESTION_START_REGEX, re.IGNORECASE)

# Matches a block that begins with an explicit question label.
# Examples it accepts:
#   "Question 1:"  "QUESTION 2."  "Q1:"  "Q.3)"  "(1)"  "1."  "a)"
_LABEL_RE = re.compile(
    r"^\s*(?:"
    r"Question\s+\d+\s*[:\.\)]"
    r"|Q\s*\d+\s*[:\.\)]"
    r"|\(\d+\)"
    r"|\d+\s*[\.\)]"
    r"|[a-z]\s*[\)\.]"
    r")",
    re.IGNORECASE,
)

# Rejects blocks that start with legal section numbering, e.g. "11.1 …"
_NON_QUESTION_START_RE = re.compile(NON_QUESTION_START_REGEX)

# Matches a block that starts with a command word ("Define", "State", etc.)
_COMMAND_WORD_RE = re.compile(
    r"^\s*(?:" + "|".join(QUESTION_COMMAND_WORDS) + r")\b",
    re.IGNORECASE,
)


# ============================================================================
# Readers
# ============================================================================
def read_docx(path: Path) -> str:
    """Return all text from a .docx, including paragraphs and tables."""
    doc = Document(str(path))
    parts = [p.text for p in doc.paragraphs]
    # Tables often hold Q&A content in exam-prep documents
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                parts.append(cell.text)
    return "\n".join(parts)


def read_pdf(path: Path) -> str:
    """Return all text from a .pdf, page by page."""
    reader = PdfReader(str(path))
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception as exc:  # noqa: BLE001
            logger.warning("PDF page extract failed in %s: %s", path.name, exc)
    return "\n".join(parts)


def read_any(path: Path) -> str:
    """Dispatch to the correct reader based on file extension."""
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return read_docx(path)
    if suffix == ".pdf":
        return read_pdf(path)
    return ""


# ============================================================================
# Topic hints
# ============================================================================
def infer_topic_hint(path: Path) -> str:
    """Guess a topic from the parent folder name, then the filename.

    The hint is a starting point for manual labelling — not a final label.
    """
    # Walk up the path checking each folder name (innermost first)
    for part in reversed(path.parts):
        key = part.lower().strip()
        if key in FOLDER_TOPIC_HINTS:
            return FOLDER_TOPIC_HINTS[key]

    # Fall back to filename keyword matching
    name = path.name.lower()
    for needle, topic in FILENAME_TOPIC_HINTS.items():
        if needle in name:
            return topic

    return "Other"


# ============================================================================
# Question filtering
# ============================================================================
def looks_like_question(block: str) -> bool:
    """Return True if the block reads like an exam question.

    Balanced filter: prefers recall over precision, then prunes obvious
    answer-bullet fragments that don't look like questions or answers.

    Decision order:
        1. Too short → reject.
        2. Starts with "Question N:" / "Q N:" → accept (strongest signal).
        3. Starts with legal clause numbering ("11.1") → reject.
        4. Starts with a bullet-fragment pattern ("a. Something:" or
           "5. Header:") where the first line ends with ":" and there's no "?"
           anywhere and the block is short → reject.
        5. Ends with "?" → accept.
        6. Starts with a command word → accept.
        7. Starts with a numbered or lettered label → accept.
        8. Heading-like first line → accept.
        9. Otherwise → reject.
    """
    text = block.strip()
    if len(text) < MIN_QUESTION_CHARS:
        return False

    # Rule 2: "Question N:" / "QN:" always wins
    if re.match(r"^\s*(?:Question\s+\d+|Q\s*\d+)\s*[:\.\)]", text, re.IGNORECASE):
        return True

    # Rule 3: legal clause numbering is not a question
    if _NON_QUESTION_START_RE.match(text):
        return False

    # Rule 4: reject obvious bullet-fragment blocks
    first_line = text.split("\n", 1)[0].strip()
    looks_like_fragment = bool(
        re.match(
            r"^\s*(?:[a-z]|\d+)\s*[\.\)]\s+[^?]{0,80}:?\s*$",
            first_line,
            re.IGNORECASE,
        )
    ) and first_line.endswith(":") and "?" not in text and len(text) < 200

    if looks_like_fragment:
        return False

    # Rule 5: ends with ?
    if text.rstrip().endswith("?"):
        return True

    # Rule 6: starts with a command word
    if _COMMAND_WORD_RE.match(text):
        return True

    # Rule 7: starts with a numbered or lettered label
    if _LABEL_RE.match(text):
        return True

    # Rule 8: heading-like first line
    if len(first_line) < 200 and first_line[:1].isupper():
        return True

    return False


# ============================================================================
# Splitting
# ============================================================================
def split_into_questions(text: str) -> list[str]:
    """Split raw document text into filtered question blocks.

    Steps:
        1. Clean whitespace but preserve newlines (needed for line-anchored
           regex matching).
        2. Find every position where a question starts.
        3. Slice the text between consecutive starts.
        4. Truncate excessively long blocks (question + long answer) so that
           the classifier sees mostly the question, with just a few lines of
           context.
        5. Run each block through `looks_like_question()` and keep only those
           that pass.
    """
    text = clean_text(text)
    if not text:
        return []

    matches = list(QUESTION_START_RE.finditer(text))
    if not matches:
        # No question markers found. Accept the whole document only if it
        # looks like a single question and isn't oversized.
        if looks_like_question(text) and len(text) <= MAX_QUESTION_CHARS:
            return [text]
        return []

    blocks: list[str] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end].strip()

        # Cap very long blocks: keep the question line plus up to 10 lines
        # of the answer body, then hard-trim to MAX_QUESTION_CHARS.
        if len(block) > MAX_QUESTION_CHARS:
            lines = block.split("\n")
            block = "\n".join(lines[:10])[:MAX_QUESTION_CHARS]

        if looks_like_question(block):
            blocks.append(block)

    return blocks


# ============================================================================
# Main extraction pipeline
# ============================================================================
def iter_raw_files() -> Iterable[Path]:
    """Yield every .docx and .pdf under data/raw/, skipping excluded paths."""
    for path in sorted(RAW_DIR.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".docx", ".pdf"}:
            continue
        rel = path.relative_to(RAW_DIR).as_posix().lower()
        if any(bad in rel for bad in EXCLUDE_PATH_SUBSTRINGS):
            logger.info("Skipping (excluded): %s", rel)
            continue
        yield path


def extract_all() -> list[dict]:
    """Walk data/raw/, extract question blocks, return a list of row dicts."""
    rows: list[dict] = []
    files = list(iter_raw_files())
    if not files:
        logger.error("No .docx or .pdf files found under %s", RAW_DIR)
        return rows

    for path in files:
        rel = safe_filename(path.relative_to(RAW_DIR))
        topic_hint = infer_topic_hint(path)
        logger.info("Reading %s (topic hint: %s)", rel, topic_hint)

        try:
            raw_text = read_any(path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to read %s: %s", rel, exc)
            continue

        if not raw_text.strip():
            logger.warning("Empty text from %s — skipping", rel)
            continue

        blocks = split_into_questions(raw_text)
        logger.info("  -> %d question blocks", len(blocks))

        for block in blocks:
            rows.append(
                {
                    "question_text": block,
                    "topic": "",               # filled in by hand later
                    "difficulty_label": "",    # filled in by hand later
                    "source_file": rel,
                    "topic_hint": topic_hint,
                    "notes": "",
                }
            )

    return rows


def write_csv(rows: list[dict]) -> None:
    """Write extracted rows to data/interim/questions_raw.csv, deduplicated.

    Deduplication key: question_text, lower-cased and whitespace-normalised.
    This collapses identical questions that appear in both a .docx and .pdf
    version of the same source file (or are simply repeated across files).
    """
    ensure_parent(RAW_QUESTIONS_CSV)
    fieldnames = [
        "question_text",
        "topic",
        "difficulty_label",
        "source_file",
        "topic_hint",
        "notes",
    ]

    seen: set[str] = set()
    unique_rows: list[dict] = []
    for row in rows:
        key = " ".join(str(row["question_text"]).lower().split())
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(row)

    with RAW_QUESTIONS_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(unique_rows)

    logger.info(
        "Deduplicated: %d rows → %d unique rows",
        len(rows), len(unique_rows),
    )


def main() -> None:
    """Entry point: extract, filter, deduplicate, write CSV."""
    ensure_dirs()
    logger.info("Starting extraction from %s", RAW_DIR)
    rows = extract_all()
    if not rows:
        logger.warning("No questions extracted.")
        return
    write_csv(rows)
    logger.info("Wrote rows to %s", RAW_QUESTIONS_CSV)


if __name__ == "__main__":
    main()