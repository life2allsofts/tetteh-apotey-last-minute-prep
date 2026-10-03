"""Evaluate the trained models against the labelled dataset."""

from __future__ import annotations

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.config import (
    DIFFICULTY_MODEL_PATH,
    FEATURES_CSV,
    FEATURE_COLUMNS,
    TOPIC_MODEL_PATH,
)
from src.utils import setup_logger

logger = setup_logger("evaluate")


def report(model_path, df: pd.DataFrame, target: str) -> None:
    model = joblib.load(model_path)
    X = df[FEATURE_COLUMNS].values
    y_true = df[target].astype(str).values
    y_pred = model.predict(X)

    logger.info("── %s ──", target)
    print(classification_report(y_true, y_pred, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))


def main() -> None:
    df = pd.read_csv(FEATURES_CSV)
    df = df[df["topic"].astype(str).str.strip() != ""]
    report(TOPIC_MODEL_PATH, df, "topic")
    df = df[df["difficulty_label"].astype(str).str.strip() != ""]
    report(DIFFICULTY_MODEL_PATH, df, "difficulty_label")


if __name__ == "__main__":
    main()