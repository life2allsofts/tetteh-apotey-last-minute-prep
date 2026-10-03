#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
echo "▶ Engineering features ..."
python -m src.feature_engineering
echo "▶ Training TabPFN models ..."
python -m src.train
echo "✅ Done. Models saved in models/"