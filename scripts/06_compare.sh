#!/bin/bash
# Step 6: Compare Hebbian vs autodiff training runs
# Finds the most recent run of each type and prints side-by-side metrics.
# Run after both 02 and 03 have completed.
set -e
cd "$(dirname "$0")/.."

HEBBIAN_DIR=$(ls -td runs/hebbian_*/ 2>/dev/null | head -1)
AUTODIFF_DIR=$(ls -td runs/autodiff_*/ 2>/dev/null | head -1)

if [ -z "$HEBBIAN_DIR" ]; then echo "No Hebbian run found. Run scripts/02_train_hebbian.sh first."; exit 1; fi
if [ -z "$AUTODIFF_DIR" ]; then echo "No Autodiff run found. Run scripts/03_train_autodiff.sh first."; exit 1; fi

echo "=== Hebbian vs Autodiff Comparison ==="
echo ""
echo "Hebbian:  $HEBBIAN_DIR"
echo "Autodiff: $AUTODIFF_DIR"
echo ""

echo "--- Hebbian: last 3 metrics ---"
tail -3 "$HEBBIAN_DIR/metrics.jsonl" 2>/dev/null || echo "No metrics"
echo ""

echo "--- Autodiff: last 3 metrics ---"
tail -3 "$AUTODIFF_DIR/metrics.jsonl" 2>/dev/null || echo "No metrics"
echo ""

# Plot both
echo "--- Generating plots ---"
uv run python scripts/plot_metrics.py "$HEBBIAN_DIR/metrics.jsonl" "$HEBBIAN_DIR" 2>/dev/null && echo "  Hebbian plots: $HEBBIAN_DIR"
uv run python scripts/plot_metrics.py "$AUTODIFF_DIR/metrics.jsonl" "$AUTODIFF_DIR" 2>/dev/null && echo "  Autodiff plots: $AUTODIFF_DIR"
echo ""
echo "Compare error.png and metrics.png in each directory."
