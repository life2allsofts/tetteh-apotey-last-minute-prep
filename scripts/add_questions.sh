#!/usr/bin/env bash
#
# Convenience wrapper: drop new .docx/.pdf files into data/raw/ and run this.
# It runs the full pipeline end-to-end.
#
set -euo pipefail
cd "$(dirname "$0")/.."

echo "📥 Adding new questions from data/raw/"
echo ""
bash scripts/run_pipeline.sh
