"""Streamlit entry point for Last-Minute-Prep."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Ensure project root is importable when running `streamlit run app/app.py`
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="Last-Minute-Prep",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar branding
with st.sidebar:
    st.title("📐 Last-Minute-Prep")
    st.caption("TabPFN-powered exam predictor for the Ghana Surveyors Licensing Board")
    st.divider()

st.title("Welcome to Last-Minute-Prep")
st.markdown(
    """
    A local, offline-first study companion that classifies every practice question by
    **topic** and **difficulty**, so a candidate can see at a glance where to focus
    the night before the exam.

    ### 👈 Use the sidebar
    - **1 · Predict** — paste a question, get its topic + difficulty
    - **2 · Study Focus** — see your weakest areas at a glance
    - **3 · Question Explorer** — browse and filter the full dataset

    ### Why open source?
    TabPFN runs **entirely on this laptop**. No exam content ever leaves the machine,
    and the whole thing works without internet — perfect for a dorm room with
    unreliable wifi.
    """
)

st.divider()
st.caption(
    "Built by Isaac Tetteh-Apotey (@life2allsofts) · Apache-2.0 · "
    "TabPFN is © PriorLabs, used under its open licence."
)