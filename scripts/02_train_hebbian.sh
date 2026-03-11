#!/bin/bash
# Step 2: Train with Hebbian weight updates (fresh start, bounded predictions)
# This is the baseline run with the new 3*tanh bounded prediction architecture.
# Runs in background via nohup. Use scripts/monitor.sh to check progress.
set -e
cd "$(dirname "$0")/.."

RUN_NAME="hebbian_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="runs/$RUN_NAME"
mkdir -p "$RUN_DIR"

nohup uv run python manage.py train_hierarchical \
    --num-steps 1000 \
    --batch-size 16 \
    --layer-sizes 128 64 32 64 128 \
    --relaxation-steps 32 \
    --teacher-forcing 0.8 \
    --lr 0.01 \
    --lr-w 0.001 \
    --weight-update hebbian \
    --checkpoint-dir "$RUN_DIR/checkpoints" \
    --checkpoint-every 100 \
    --metrics-dir "$RUN_DIR" \
    > "$RUN_DIR/stdout.log" 2>&1 &

echo $! > "$RUN_DIR/pid"
echo "Hebbian training started:"
echo "  PID:  $(cat "$RUN_DIR/pid")"
echo "  Dir:  $RUN_DIR"
echo "  Logs: $RUN_DIR/stdout.log"
echo ""
echo "Monitor with: bash scripts/monitor.sh $RUN_DIR"
