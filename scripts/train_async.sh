#!/bin/bash
set -e
RUN_NAME=${1:-$(date +%Y%m%d_%H%M%S)}
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
    --checkpoint-dir "$RUN_DIR/checkpoints" \
    --checkpoint-every 100 \
    --metrics-dir "$RUN_DIR" \
    > "$RUN_DIR/stdout.log" 2>&1 &

echo $! > "$RUN_DIR/pid"
echo "Training started: PID=$(cat "$RUN_DIR/pid"), logs at $RUN_DIR/"
