from src.extract_questions import split_into_questions


def test_split_numbered_questions():
    text = "1. What is an allodial title? 2. Define a beacon. 3. State the mix ratio."
    blocks = split_into_questions(text)
    assert len(blocks) >= 2


def test_split_without_markers_returns_single_block():
    text = "This is a long descriptive paragraph about survey practice in Ghana."
    blocks = split_into_questions(text)
    assert len(blocks) == 1