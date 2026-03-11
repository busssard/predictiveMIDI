#!/bin/bash
set -e

echo "=== Hebbian vs Autodiff Comparison ==="
echo "Training both methods with identical settings..."
echo ""

# Train with Hebbian updates
echo "--- Training with Hebbian weight updates ---"
uv run python manage.py train_hierarchical \
    --num-steps 200 --batch-size 16 \
    --layer-sizes 128 64 32 64 128 \
    --weight-update hebbian \
    --checkpoint-dir runs/compare_hebbian/checkpoints

# Train with autodiff updates
echo ""
echo "--- Training with Autodiff weight updates ---"
uv run python manage.py train_hierarchical \
    --num-steps 200 --batch-size 16 \
    --layer-sizes 128 64 32 64 128 \
    --weight-update autodiff \
    --checkpoint-dir runs/compare_autodiff/checkpoints

echo ""
echo "Done. Compare checkpoints in:"
echo "  runs/compare_hebbian/checkpoints/"
echo "  runs/compare_autodiff/checkpoints/"
