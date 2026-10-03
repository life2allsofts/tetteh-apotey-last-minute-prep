import pandas as pd

from src.feature_engineering import build_features


def test_has_statute_reference_detected():
    df = pd.DataFrame({"question_text": ["Under Act 1036, who holds title?"]})
    out = build_features(df)
    assert out["has_statute_reference"].iloc[0] == 1


def test_word_count():
    df = pd.DataFrame({"question_text": ["one two three four five"]})
    out = build_features(df)
    assert out["word_count"].iloc[0] == 5


def test_question_type_computation():
    df = pd.DataFrame({"question_text": ["Calculate the bearing from A to B."]})
    out = build_features(df)
    assert out["question_type_encoded"].iloc[0] == 2  # computation