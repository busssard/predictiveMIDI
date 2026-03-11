#!/bin/bash
# Step 5: Plot training metrics for one or all runs
# Usage:
#   bash scripts/05_plot.sh                     # plot most recent run
#   bash scripts/05_plot.sh runs/hebbian_*      # plot specific run
#   bash scripts/05_plot.sh --all               # plot all runs
set -e
cd "$(dirname "$0")/.."

if [ "$1" = "--all" ]; then
    for f in runs/*/metrics.jsonl; do
        if [ -f "$f" ]; then
            dir=$(dirname "$f")
            echo "Plotting: $dir"
            uv run python scripts/plot_metrics.py "$f" "$dir"
            echo ""
        fi
    done
    exit 0
fi

if [ -n "$1" ]; then
    RUN_DIR="$1"
else
    RUN_DIR=$(ls -td runs/*/ 2>/dev/null | head -1)
fi

if [ -z "$RUN_DIR" ]; then echo "No runs found in runs/"; exit 1; fi

METRICS="$RUN_DIR/metrics.jsonl"
if [ ! -f "$METRICS" ]; then echo "No metrics.jsonl in $RUN_DIR"; exit 1; fi

uv run python scripts/plot_metrics.py "$METRICS" "$RUN_DIR"
