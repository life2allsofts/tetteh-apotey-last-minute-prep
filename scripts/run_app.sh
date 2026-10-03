#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
echo "▶ Launching Streamlit ..."
streamlit run app/app.py