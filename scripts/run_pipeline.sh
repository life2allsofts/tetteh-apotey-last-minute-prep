#!/usr/bin/env bash
#
# Full Last-Minute-Prep pipeline:
#   1. Extract questions from data/raw/
#   2. Bootstrap labels into data/processed/questions.csv
#   3. Promote difficulty (Easy → Medium/Hard)
#   4. Engineer features
#   5. Train TabPFN models
#
# Usage:
#   bash scripts/run_pipeline.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."

echo "════════════════════════════════════════════════════════════"
echo "  Last-Minute-Prep — Full Pipeline"
echo "════════════════════════════════════════════════════════════"

echo ""
echo "▶ Step 1/5 — Extracting questions from data/raw/"
python -m src.extract_questions

echo ""
echo "▶ Step 2/5 — Bootstrapping labels"
python scripts/bootstrap_labels.py

echo ""
echo "▶ Step 3/5 — Promoting difficulty (Easy → Medium/Hard)"
python scripts/promote_difficulty.py

echo ""
echo "▶ Step 4/5 — Engineering features"
python -m src.feature_engineering

echo ""
echo "▶ Step 5/5 — Training TabPFN models"
python -m src.train

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  ✅ Pipeline complete."
echo "     Launch the app with:  bash scripts/run_app.sh"
echo "════════════════════════════════════════════════════════════"
