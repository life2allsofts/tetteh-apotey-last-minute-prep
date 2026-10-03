#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
echo "▶ Extracting questions from data/raw/ ..."
python -m src.extract_questions
echo "✅ Done. Check data/interim/questions_raw.csv"