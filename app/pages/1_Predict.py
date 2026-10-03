"""Predict topic + difficulty for a pasted question."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import LABELLED_QUESTIONS_CSV, SAMPLE_QUESTIONS_CSV  # noqa: E402
from src.predict import predict  # noqa: E402

st.set_page_config(page_title="Predict · Last-Minute-Prep", page_icon="🔮")
st.title("🔮 Predict a Question")
st.caption("Paste a question and TabPFN will predict its topic and difficulty.")

question = st.text_area(
    "Question text",
    height=180,
    placeholder="e.g. Under Section 9 of Act 1036, can a stool alienate land without the President's consent?",
)

col1, col2 = st.columns([1, 5])
with col1:
    run = st.button("Predict", type="primary", use_container_width=True)
with col2:
    example = st.button("Load example", use_container_width=False)

if example:
    st.session_state["_example_question"] = (
        "Under Section 9 of Act 1036, can a stool alienate land without "
        "the President's consent?"
    )
    st.rerun()

if "_example_question" in st.session_state and not question:
    question = st.session_state.pop("_example_question")

if run and question.strip():
    with st.spinner("Running TabPFN inference…"):
        try:
            result = predict(question)
        except FileNotFoundError as exc:
            st.error(str(exc))
            st.stop()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Prediction failed: {exc}")
            st.stop()

    c1, c2 = st.columns(2)
    c1.metric("Predicted topic", result["topic"])
    c2.metric("Predicted difficulty", result["difficulty"])

    with st.expander("Confidence & details"):
        st.write(
            {
                "topic_confidence": result.get("topic_confidence"),
                "difficulty_confidence": result.get("difficulty_confidence"),
            }
        )

    # Show nearest labelled examples from the dataset for sanity-checking
    path = LABELLED_QUESTIONS_CSV if LABELLED_QUESTIONS_CSV.exists() else SAMPLE_QUESTIONS_CSV
    if path.exists():
        df = pd.read_csv(path)
        same_topic = df[df["topic"].astype(str) == result["topic"]].head(5)
        if not same_topic.empty:
            st.subheader("Similar labelled questions")
            st.dataframe(
                same_topic[["question_text", "topic", "difficulty_label"]],
                use_container_width=True,
                hide_index=True,
            )
elif run:
    st.warning("Please paste a question first.")