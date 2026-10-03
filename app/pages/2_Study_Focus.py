"""Dashboard: where to focus study time."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import LABELLED_QUESTIONS_CSV, SAMPLE_QUESTIONS_CSV  # noqa: E402

st.set_page_config(page_title="Study Focus · Last-Minute-Prep", page_icon="🎯")
st.title("🎯 Study Focus")
st.caption("Where the hard questions cluster — and what to revise first.")

path = LABELLED_QUESTIONS_CSV if LABELLED_QUESTIONS_CSV.exists() else SAMPLE_QUESTIONS_CSV
if not path.exists():
    st.error("No dataset found. Run the extraction + labelling pipeline first.")
    st.stop()

df = pd.read_csv(path)
df = df[df["topic"].astype(str).str.strip() != ""]

if df.empty:
    st.warning("The dataset is empty or unlabelled.")
    st.stop()

st.subheader("Questions per topic, split by difficulty")
grouped = (
    df.groupby(["topic", "difficulty_label"]).size().reset_index(name="count")
)
fig = px.bar(
    grouped,
    x="topic",
    y="count",
    color="difficulty_label",
    barmode="stack",
    color_discrete_map={"Easy": "#4ade80", "Medium": "#fbbf24", "Hard": "#ef4444"},
)
fig.update_layout(xaxis_title="", yaxis_title="Number of questions")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Hardest questions right now")
hard = df[df["difficulty_label"].astype(str).str.lower() == "hard"]
if hard.empty:
    st.info("No questions labelled 'Hard' yet.")
else:
    st.dataframe(
        hard[["question_text", "topic", "source_file"]].head(15),
        use_container_width=True,
        hide_index=True,
    )

st.divider()
st.caption(f"Loaded {len(df)} labelled questions from {path.name}.")