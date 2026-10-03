"""Browse and filter the labelled dataset."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import LABELLED_QUESTIONS_CSV, SAMPLE_QUESTIONS_CSV  # noqa: E402

st.set_page_config(page_title="Explorer · Last-Minute-Prep", page_icon="🔎")
st.title("🔎 Question Explorer")

path = LABELLED_QUESTIONS_CSV if LABELLED_QUESTIONS_CSV.exists() else SAMPLE_QUESTIONS_CSV
if not path.exists():
    st.error("No dataset found.")
    st.stop()

df = pd.read_csv(path)

topics = ["(all)"] + sorted(df["topic"].dropna().astype(str).unique().tolist())
difficulties = ["(all)"] + sorted(
    df["difficulty_label"].dropna().astype(str).unique().tolist()
)

c1, c2, c3 = st.columns([2, 2, 3])
with c1:
    t = st.selectbox("Topic", topics)
with c2:
    d = st.selectbox("Difficulty", difficulties)
with c3:
    q = st.text_input("Search text")

view = df.copy()
if t != "(all)":
    view = view[view["topic"].astype(str) == t]
if d != "(all)":
    view = view[view["difficulty_label"].astype(str) == d]
if q.strip():
    view = view[view["question_text"].astype(str).str.contains(q, case=False, regex=False)]

st.caption(f"{len(view)} rows")
st.dataframe(view, use_container_width=True, hide_index=True)

st.download_button(
    "Download filtered CSV",
    data=view.to_csv(index=False).encode("utf-8"),
    file_name="filtered_questions.csv",
    mime="text/csv",
)