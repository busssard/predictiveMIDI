#!/bin/bash
# Step 4: Monitor a training run (or all active runs)
# Usage:
#   bash scripts/04_monitor.sh                  # monitor most recent run
#   bash scripts/04_monitor.sh runs/hebbian_*   # monitor specific run
#   bash scripts/04_monitor.sh --all            # monitor all active runs
set -e
cd "$(dirname "$0")/.."

if [ "$1" = "--all" ]; then
    for d in runs/*/; do
        if [ -f "$d/pid" ]; then
            echo "================================================"
            bash scripts/monitor.sh "$d"
            echo ""
        fi
    done
    exit 0
fi

RUN_DIR=${1:-$(ls -td runs/*/ 2>/dev/null | head -1)}
if [ -z "$RUN_DIR" ]; then echo "No runs found in runs/"; exit 1; fi

bash scripts/monitor.sh "$RUN_DIR"
