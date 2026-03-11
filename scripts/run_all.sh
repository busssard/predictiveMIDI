#!/bin/bash
# Full pipeline: rebuild corpus, train both methods, plot and compare.
set -e
cd "$(dirname "$0")/.."

echo "============================================"
echo "  Predictive Coding — Full Training Pipeline"
echo "============================================"
echo ""
echo "Note: Step 1 of each training run is slow (JIT compilation)."
echo "Subsequent steps use lax.scan + fori_loop (1 dispatch/example)."
echo ""

# Step 1: Rebuild corpus
echo "[1/6] Rebuilding corpus index with quality filters..."
bash scripts/01_rebuild_corpus.sh
echo ""

# Step 2: Train Hebbian (foreground, not nohup)
echo "[2/6] Training Hebbian (1000 steps)..."
HEBBIAN_DIR="runs/hebbian_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$HEBBIAN_DIR"
uv run python manage.py train_hierarchical \
    --midi-dir data/midi \
    --num-steps 1000 \
    --batch-size 16 \
    --layer-sizes 128 64 32 64 128 \
    --relaxation-steps 32 \
    --teacher-forcing 0.8 \
    --lr 0.01 \
    --lr-w 0.001 \
    --weight-update hebbian \
    --checkpoint-dir "$HEBBIAN_DIR/checkpoints" \
    --checkpoint-every 100 \
    --metrics-dir "$HEBBIAN_DIR" \
    2>&1 | tee "$HEBBIAN_DIR/stdout.log"
echo ""

# Step 3: Train autodiff (foreground)
echo "[3/6] Training Autodiff (1000 steps)..."
AUTODIFF_DIR="runs/autodiff_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$AUTODIFF_DIR"
uv run python manage.py train_hierarchical \
    --midi-dir data/midi \
    --num-steps 1000 \
    --batch-size 16 \
    --layer-sizes 128 64 32 64 128 \
    --relaxation-steps 32 \
    --teacher-forcing 0.8 \
    --lr 0.01 \
    --lr-w 0.001 \
    --weight-update autodiff \
    --checkpoint-dir "$AUTODIFF_DIR/checkpoints" \
    --checkpoint-every 100 \
    --metrics-dir "$AUTODIFF_DIR" \
    2>&1 | tee "$AUTODIFF_DIR/stdout.log"
echo ""

# Step 4: Plot Hebbian
echo "[4/6] Plotting Hebbian metrics..."
uv run python scripts/plot_metrics.py "$HEBBIAN_DIR/metrics.jsonl" "$HEBBIAN_DIR"
echo ""

# Step 5: Plot autodiff
echo "[5/6] Plotting Autodiff metrics..."
uv run python scripts/plot_metrics.py "$AUTODIFF_DIR/metrics.jsonl" "$AUTODIFF_DIR"
echo ""

# Step 6: Compare
echo "[6/6] Comparison summary"
echo "============================================"
echo "  Hebbian:  $HEBBIAN_DIR"
echo "  Autodiff: $AUTODIFF_DIR"
echo "============================================"
echo ""
echo "--- Hebbian final metrics ---"
tail -1 "$HEBBIAN_DIR/metrics.jsonl"
echo ""
echo "--- Autodiff final metrics ---"
tail -1 "$AUTODIFF_DIR/metrics.jsonl"
echo ""
echo "Plots saved to:"
echo "  $HEBBIAN_DIR/error.png  $HEBBIAN_DIR/metrics.png"
echo "  $AUTODIFF_DIR/error.png  $AUTODIFF_DIR/metrics.png"
echo ""
echo "Done."
