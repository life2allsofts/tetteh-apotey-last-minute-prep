"""Train two TabPFN classifiers: topic and difficulty.

Each classifier is trained on the same engineered-feature matrix, but with
a different target column. TabPFN is used zero-shot / in-context: it does
not need fine-tuning, just a small labelled training table.

Special handling:
    - For the **topic** model, we drop the `Other` catch-all class. It has
      no consistent pattern and hurts the classifier's macro-F1. Rows
      labelled `Other` are still available for the difficulty model, since
      difficulty is independent of topic.
    - For both models, we convert dataframes to plain NumPy arrays before
      handing them to scikit-learn. This avoids a PyArrow-backed pandas
      incompatibility on pandas 3.x.

Both models are serialised to disk under `models/` for the Streamlit app.
"""

from __future__ import annotations

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier

from src.config import (
    DIFFICULTY_MODEL_PATH,
    FEATURES_CSV,
    FEATURE_COLUMNS,
    TABPFN_DEVICE,
    TABPFN_N_ESTIMATORS,
    TABPFN_RANDOM_STATE,
    TOPIC_MODEL_PATH,
)
from src.utils import ensure_parent, setup_logger

logger = setup_logger("train")


def train_one(
    df: pd.DataFrame,
    target_col: str,
    model_path,
    *,
    drop_labels: tuple[str, ...] = (),
) -> TabPFNClassifier:
    """Train a single TabPFN classifier for the given target column.

    Args:
        df: Feature dataframe (must contain FEATURE_COLUMNS + target_col).
        target_col: Name of the label column (e.g. "topic").
        model_path: Where to save the trained classifier.
        drop_labels: Labels to exclude from training (e.g. ("Other",)).

    Returns:
        The fitted TabPFNClassifier.
    """
    # Drop rows where the target is missing or blank
    df = df.dropna(subset=[target_col]).copy()
    df = df[df[target_col].astype(str).str.strip() != ""]

    # Optionally drop specific classes (used to remove the 'Other' catch-all
    # from the topic model — it has no consistent signal and confuses the
    # classifier).
    if drop_labels:
        before = len(df)
        df = df[~df[target_col].astype(str).isin(drop_labels)]
        dropped = before - len(df)
        if dropped:
            logger.info(
                "Dropped %d rows with labels %s from '%s' training",
                dropped, list(drop_labels), target_col,
            )

    if len(df) < 6:
        raise ValueError(
            f"Not enough labelled rows for '{target_col}' ({len(df)}). "
            f"Fill in more labels in data/processed/questions.csv."
        )

    # Convert to plain NumPy arrays — bypasses the PyArrow-backed pandas
    # backend, which trips up scikit-learn's fancy indexing on pandas 3.x.
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df[target_col].astype(str).to_numpy()

    # Small dataset → use a modest test split only if we have enough rows.
    # Otherwise, evaluate on the training set itself (still useful as a
    # sanity check, just optimistically biased).
    if len(df) >= 20:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=TABPFN_RANDOM_STATE,
            stratify=None,
        )
    else:
        X_train, y_train = X, y
        X_test, y_test = X, y
        logger.warning("Fewer than 20 rows — skipping real train/test split.")

    # TabPFN handles small tabular data out of the box — no hyperparameter
    # tuning needed. N_estimators controls the internal ensemble size.
    clf = TabPFNClassifier(
        device=TABPFN_DEVICE,
        n_estimators=TABPFN_N_ESTIMATORS,
        random_state=TABPFN_RANDOM_STATE,
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    logger.info(
        "[%s] rows=%d  accuracy=%.3f  macro-F1=%.3f",
        target_col, len(df), acc, f1,
    )

    # Save model artifact
    ensure_parent(model_path)
    joblib.dump(clf, model_path)
    logger.info("Saved model → %s", model_path)

    return clf


def main() -> None:
    """Load the engineered features and train both TabPFN classifiers."""
    if not FEATURES_CSV.exists():
        raise FileNotFoundError(
            f"{FEATURES_CSV} not found. "
            f"Run `python -m src.feature_engineering` first."
        )

    df = pd.read_csv(FEATURES_CSV)
    logger.info("Loaded %d rows from %s", len(df), FEATURES_CSV)

    # Topic model: drop the 'Other' catch-all class.
    train_one(df, "topic", TOPIC_MODEL_PATH, drop_labels=("Other",))

    # Difficulty model: keep all rows (difficulty is independent of topic).
    train_one(df, "difficulty_label", DIFFICULTY_MODEL_PATH)

    logger.info("✅ Both models trained successfully.")
    logger.info("   Topic model:      %s", TOPIC_MODEL_PATH)
    logger.info("   Difficulty model: %s", DIFFICULTY_MODEL_PATH)


if __name__ == "__main__":
    main()