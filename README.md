# Last-Minute-Prep

> **An offline-first, TabPFN-powered study companion that turns your own course notes into a personalised exam predictor — built for a friend preparing for the Ghana Surveyors Licensing Board Examination.**

[![Version](https://img.shields.io/badge/version-v0.1.0-amber.svg)](https://github.com/life2allsofts/last-minute-prep)
[![Developer](https://img.shields.io/badge/developer-life2allsofts-sky.svg)](mailto:life2allsofts@gmail.com)
[![AI](https://img.shields.io/badge/AI-TabPFN%20v2.5-blueviolet.svg)](https://github.com/PriorLabs/TabPFN)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b.svg)](https://streamlit.io/)
[![Offline](https://img.shields.io/badge/offline-100%25%20local-success.svg)](https://github.com/life2allsofts)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

Built for the **Hacktoberfest 2026 Weekend Challenge — Build for a Friend**.

📖 **[Full technical documentation →](docs/Design-And-Testing.md)**

---

## 🎯 What Is This?

**Last-Minute-Prep** is a small, focused web app built for one real person: a friend preparing for the **Ghana Surveyors Licensing Examination**. She has hundreds of practice questions and past-paper answers from a refresher course, but no way to know **which topics she is weakest in** or **which questions are actually hard**.

Traditional machine learning needs thousands of rows to be useful. She has about 400.

So this project uses **TabPFN** — an open-weight, transformer-based foundation model for tabular data — which delivers accurate predictions from **tiny datasets** (10–1,000 rows), running **entirely on a laptop with no internet and no API keys**.

### Why open innovation matters here

- **Her study data never leaves her laptop** — no exam content sent to a closed API.
- **It runs offline** in a dorm room with unstable internet.
- **She can retrain the model on her own new questions** by editing a CSV — no ML expertise required.
- **Cost to run: $0. Forever.**

---

## 🧩 The Problem & The Person

**Who it's for:** A fellow candidate preparing for the Licensed Surveyor examination under the GhIS / Lands Commission. She attended a refresher course and now has 30+ files: past-paper answers, preparatory questions, land-law Q&A, land-dispute case notes, lecture slides, and 100+ oral interview questions.

**The problem:**

- 400+ questions, no metadata, no difficulty ratings, no topic labels.
- She doesn't know which questions are "Board-hard" vs. "warm-up easy."
- Existing closed AI APIs cost money, need internet, and she doesn't want her exam prep sent to a server she doesn't control.

**The solution:**

A small local app that reads a CSV of her questions, uses **TabPFN** to predict each question's **difficulty bucket** and **topic category**, and shows her a personalised "study focus" dashboard. She can add new questions by editing the CSV in Excel and re-running the app — no code changes needed.

---

## ✨ Features

- 🔮 **Predict** — paste any question, get its predicted topic and difficulty, with confidence scores
- 🎯 **Study Focus** — a bar chart showing where hard questions cluster, so she knows what to review first
- 🔎 **Question Explorer** — filter the full dataset by topic and difficulty, download filtered CSVs
- 📦 **Offline-first** — runs entirely on a laptop after the first TabPFN model download (~43 MB, one time)
- 🔁 **Reproducible pipeline** — `bash scripts/run_pipeline.sh` re-runs extraction, labelling, and training from scratch

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **AI** | TabPFN v2.5 | Open-weight tabular foundation model — the core predictor |
| **ML Runtime** | PyTorch, scikit-learn | Backend tensor runtime and evaluation |
| **Data** | pandas, NumPy | CSV ingestion, feature engineering |
| **Extraction** | python-docx, pypdf | Parse `.docx` and `.pdf` course files |
| **UI** | Streamlit, Plotly | Local web interface and charts |
| **Persistence** | joblib, local CSVs | Model serialisation and dataset storage |
| **Testing** | pytest | Unit tests for the feature extractor and predictor |

See [docs/Design-And-Testing.md](docs/Design-And-Testing.md) for the full breakdown, architecture diagrams, and challenge log.

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/life2allsofts/last-minute-prep.git
cd last-minute-prep

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate          # macOS/Linux
# venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your own course files to data/raw/ (optional — a synthetic sample is included)
#    Then run the full pipeline:
bash scripts/run_pipeline.sh

# 5. Launch the app
bash scripts/run_app.sh
# → http://localhost:8501
```

**Note:** The first run downloads TabPFN's pretrained weights (~43 MB). After that, the app runs entirely offline.

---

## 📁 Project Structure

```
last-minute-prep/
├── data/
│   ├── raw/                          # 🔒 Your private course files (gitignored)
│   ├── interim/
│   │   └── questions_raw.csv         # Machine-extracted questions
│   └── processed/
│       ├── questions.csv             # 🔒 Labelled dataset (gitignored)
│       ├── questions_sample.csv      # Public synthetic sample (10 rows)
│       └── questions_features.csv    # Engineered feature matrix
├── models/                           # 🔒 Trained TabPFN models (gitignored)
├── src/
│   ├── config.py                     # Central configuration
│   ├── extract_questions.py          # DOCX/PDF → questions_raw.csv
│   ├── feature_engineering.py        # questions.csv → features
│   ├── train.py                      # Two TabPFN classifiers
│   ├── predict.py                    # Prediction API
│   ├── evaluate.py                   # Standalone evaluation
│   └── utils.py                      # Shared helpers
├── app/
│   ├── app.py                        # Streamlit entry point
│   ├── pages/
│   │   ├── 1_Predict.py              # Paste a question, get predictions
│   │   ├── 2_Study_Focus.py          # Topic × difficulty dashboard
│   │   └── 3_Question_Explorer.py    # Filterable dataset view
│   └── assets/style.css
├── scripts/
│   ├── run_pipeline.sh               # End-to-end: extract → train
│   ├── add_questions.sh              # Wrapper for adding new sources
│   ├── bootstrap_labels.py           # Auto-label questions.csv
│   ├── promote_difficulty.py         # Easy → Medium/Hard heuristic
│   ├── run_extraction.sh
│   ├── run_training.sh
│   └── run_app.sh
├── tests/                            # pytest unit tests
├── docs/
│   └── Design-And-Testing.md         # 📖 Deep technical documentation
├── notebooks/                        # EDA
├── requirements.txt
├── pyproject.toml
└── README.md
```

For a file-by-file description, see [docs/Design-And-Testing.md](docs/Design-And-Testing.md).

---

## 📊 Model Performance

Trained on **409 hand-labelled questions** across **7 topic classes** and **3 difficulty classes**.

| Model | Accuracy | Macro-F1 | Notes |
|---|---|---|---|
| Topic classifier | **0.841** | 0.668 | Trained on 344 rows (dropped `Other` catch-all) |
| Difficulty classifier | **0.646** | 0.530 | Trained on all 409 rows |

**Honest caveat:** Difficulty classification is harder than topic classification — the boundary between "Medium" and "Hard" is subjective and even human experts disagree. 65% accuracy on 3 unbalanced classes is far above random (33%) and useful for the intended purpose (surfacing which topics to review).

---

## 🌍 Why Open Source Won Here

| Requirement | Closed API | TabPFN (open) |
|---|---|---|
| Runs offline in a dorm with flaky internet | ❌ | ✅ |
| Exam content never leaves the laptop | ❌ | ✅ |
| Works with only ~400 training rows | ❌ | ✅ (built for this) |
| Cost to run forever | 💸 per-call | $0 |
| Swappable / inspectable model | ❌ | ✅ |
| Friend can retrain by editing a CSV | ❌ | ✅ |

The prompt was *"build for a friend."* The friend is a candidate. The tool is small. But it is hers, it lives on her laptop, and no one else gets to see her study data. That is the whole point.

---

## 🗺️ Roadmap

- [x] v0.1 — extraction, feature engineering, two TabPFN models, Streamlit UI
- [ ] v0.2 — image-aware features (parcel plans, composite plans) via OCR + DINOv2 embeddings
- [ ] v0.3 — per-question "why is this Hard?" using TabPFN feature importances
- [ ] v0.4 — spaced-repetition scheduler that prioritises predicted-Hard × wrong-before
- [ ] v0.5 — export a printable "study focus" PDF for offline revision
- [ ] v0.6 — optional local Gemma 3 for question answering, not just classification

---

## 📄 License & Attribution

- **Application:** Last-Minute-Prep
- **Author:** Isaac Tetteh-Apotey ([@life2allsofts](https://github.com/life2allsofts))
- **Contact:** [life2allsofts@gmail.com](mailto:life2allsofts@gmail.com)
- **Version:** v0.1.0
- **License:** Apache-2.0
- **TabPFN:** © PriorLabs, used under its open licence — [github.com/PriorLabs/TabPFN](https://github.com/PriorLabs/TabPFN)

---

## 👨‍💻 About the Developer

**Isaac Tetteh-Apotey** — Geomatics Engineer → Software Engineer | AI & Flutter Developer

- **GitHub:** [@life2allsofts](https://github.com/life2allsofts)
- **Portfolio:** [tetteh-apotey.vercel.app](https://tetteh-apotey.vercel.app/)
- **LinkedIn:** [Isaac Tetteh-Apotey](https://www.linkedin.com/in/isaac-tetteh-apotey-67408b89)
- **Email:** [life2allsofts@gmail.com](mailto:life2allsofts@gmail.com)

> *"From surveyor to software engineer — building tools for the industry I know best."*
```
