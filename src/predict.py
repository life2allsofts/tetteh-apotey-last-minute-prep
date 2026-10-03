"""Load trained models once and expose a simple predict() API.

This module is the interface between the Streamlit app and the trained
TabPFN classifiers. It caches the loaded models in memory so that repeated
predictions are fast.

Public API:
    predict(question_text: str) -> dict
        Returns {topic, difficulty, topic_confidence, difficulty_confidence}.

Both models were trained in `src/train.py`. If the model files are missing,
`predict()` raises `FileNotFoundError` with a helpful message.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import joblib
import pandas as pd

from src.config import (
    DIFFICULTY_MODEL_PATH,
    FEATURE_COLUMNS,
    TOPIC_MODEL_PATH,
)
from src.feature_engineering import build_features
from src.utils import setup_logger

logger = setup_logger("predict")


# ============================================================================
# Model loading (cached)
# ============================================================================
@lru_cache(maxsize=1)
def _load_models():
    """Load both classifiers from disk once and cache them in memory."""
    if not TOPIC_MODEL_PATH.exists() or not DIFFICULTY_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained models not found.\n"
            f"  Expected: {TOPIC_MODEL_PATH}\n"
            f"  Expected: {DIFFICULTY_MODEL_PATH}\n"
            "Run `bash scripts/run_training.sh` first."
        )
    topic_model = joblib.load(TOPIC_MODEL_PATH)
    difficulty_model = joblib.load(DIFFICULTY_MODEL_PATH)
    logger.info("Loaded topic model from %s", TOPIC_MODEL_PATH)
    logger.info("Loaded difficulty model from %s", DIFFICULTY_MODEL_PATH)
    return topic_model, difficulty_model


# ============================================================================
# Prediction
# ============================================================================
def predict(question_text: str) -> dict[str, Any]:
    """Predict topic + difficulty for a single question string.

    Args:
        question_text: The full question, including any answer preamble
                       (the feature extractor handles the rest).

    Returns:
        A dict with keys:
            topic                 — predicted topic (str)
            difficulty            — predicted difficulty (str)
            topic_confidence      — max class probability in [0, 1] (float | None)
            difficulty_confidence — max class probability in [0, 1] (float | None)
    """
    topic_model, difficulty_model = _load_models()

    # Build the feature matrix the same way training did.
    # We wrap the input in a DataFrame so build_features sees the expected
    # column name ("question_text").
    df = pd.DataFrame({"question_text": [question_text]})
    featured = build_features(df)

    # Convert to plain NumPy array (pandas 3.x + PyArrow backend workaround).
    X = featured[FEATURE_COLUMNS].to_numpy(dtype=float)

    # --- Topic prediction ---
    topic_pred = topic_model.predict(X)[0]
    result: dict[str, Any] = {
        "topic": str(topic_pred),
        "difficulty": None,
        "topic_confidence": None,
        "difficulty_confidence": None,
    }

    try:
        topic_proba = topic_model.predict_proba(X)[0]
        result["topic_confidence"] = float(max(topic_proba))
    except Exception:  # noqa: BLE001
        # Some TabPFN configurations don't expose predict_proba; ignore.
        pass

    # --- Difficulty prediction ---
    difficulty_pred = difficulty_model.predict(X)[0]
    result["difficulty"] = str(difficulty_pred)

    try:
        diff_proba = difficulty_model.predict_proba(X)[0]
        result["difficulty_confidence"] = float(max(diff_proba))
    except Exception:  # noqa: BLE001
        pass

    return result


def predict_batch(question_texts: list[str]) -> list[dict[str, Any]]:
    """Predict for a batch of questions in one pass (faster than a loop)."""
    topic_model, difficulty_model = _load_models()

    df = pd.DataFrame({"question_text": question_texts})
    featured = build_features(df)
    X = featured[FEATURE_COLUMNS].to_numpy(dtype=float)

    topic_preds = topic_model.predict(X)
    difficulty_preds = difficulty_model.predict(X)

    results: list[dict[str, Any]] = []
    for i, (t, d) in enumerate(zip(topic_preds, difficulty_preds)):
        row: dict[str, Any] = {"topic": str(t), "difficulty": str(d)}
        try:
            row["topic_confidence"] = float(
                max(topic_model.predict_proba(X[i : i + 1])[0])
            )
        except Exception:  # noqa: BLE001
            row["topic_confidence"] = None
        try:
            row["difficulty_confidence"] = float(
                max(difficulty_model.predict_proba(X[i : i + 1])[0])
            )
        except Exception:  # noqa: BLE001
            row["difficulty_confidence"] = None
        results.append(row)

    return results


# ============================================================================
# CLI: quick sanity check
# ============================================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "What is an allodial title under Act 1036?"

    result = predict(question)
    print(f"\nQuestion: {question}")
    print(f"  Topic:      {result['topic']}")
    print(f"  Difficulty: {result['difficulty']}")
    if result.get("topic_confidence") is not None:
        print(f"  Topic confidence:      {result['topic_confidence']:.2%}")
    if result.get("difficulty_confidence") is not None:
        print(f"  Difficulty confidence: {result['difficulty_confidence']:.2%}")