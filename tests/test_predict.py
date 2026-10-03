import pytest

from src.config import DIFFICULTY_MODEL_PATH, TOPIC_MODEL_PATH
from src.predict import predict


@pytest.mark.skipif(
    not (TOPIC_MODEL_PATH.exists() and DIFFICULTY_MODEL_PATH.exists()),
    reason="Models not trained yet — run scripts/run_training.sh first.",
)
def test_predict_returns_expected_keys():
    result = predict("What is an allodial title under Act 1036?")
    assert "topic" in result
    assert "difficulty" in result
    assert isinstance(result["topic"], str)
    assert isinstance(result["difficulty"], str)