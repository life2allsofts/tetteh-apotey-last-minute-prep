"""Small shared helpers: logging, text cleaning, IO."""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path


def setup_logger(name: str = "lmp", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger writing to stderr."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


_WHITESPACE_RE = re.compile(r"[ \t]+")          # collapse spaces & tabs only
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")       # collapse 3+ newlines to 2
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_text(text: str) -> str:
    """Collapse horizontal whitespace, preserve paragraph breaks, strip control chars."""
    if not text:
        return ""
    text = _CONTROL_CHARS_RE.sub(" ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")  # normalise line endings
    text = _WHITESPACE_RE.sub(" ", text)                   # collapse runs of spaces/tabs
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)             # keep paragraph structure
    return text.strip()


def safe_filename(path: Path) -> str:
    """Return a POSIX-style relative string for a path (stable across OS)."""
    return path.as_posix()


def ensure_parent(path: Path) -> None:
    """Create the parent directory of `path` if it doesn't exist."""
    path.parent.mkdir(parents=True, exist_ok=True)