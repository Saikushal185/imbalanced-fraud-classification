#!/usr/bin/env bash
# Generate imbalanced transactions and compare imbalance treatments.
set -e
cd "$(dirname "$0")"

pip install -r requirements.txt
python3 src/generate_data.py
python3 src/classify.py
echo ""
echo "See reports/pr_curves.png and reports/metrics.json"
