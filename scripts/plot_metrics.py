#!/usr/bin/env python
"""Plot training metrics from a JSONL metrics file."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_metrics(path):
    metrics = []
    for line in Path(path).read_text().strip().split("\n"):
        if line:
            metrics.append(json.loads(line))
    return metrics


def plot_metrics(metrics_path, output_dir=None):
    metrics = load_metrics(metrics_path)
    if not metrics:
        print("No metrics found")
        return

    if output_dir is None:
        output_dir = Path(metrics_path).parent
    output_dir = Path(output_dir)

    steps = [m["step"] for m in metrics]

    # Plot error/loss
    if "error" in metrics[0]:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(steps, [m["error"] for m in metrics])
        ax.set_xlabel("Step")
        ax.set_ylabel("Error")
        ax.set_title("Training Error")
        ax.grid(True)
        fig.savefig(output_dir / "error.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {output_dir / 'error.png'}")

    # Plot F1/precision/recall
    if "f1" in metrics[0]:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(steps, [m.get("f1", 0) for m in metrics], label="F1")
        ax.plot(steps, [m.get("precision", 0) for m in metrics], label="Precision")
        ax.plot(steps, [m.get("recall", 0) for m in metrics], label="Recall")
        ax.set_xlabel("Step")
        ax.set_ylabel("Score")
        ax.set_title("Training Metrics")
        ax.legend()
        ax.grid(True)
        fig.savefig(output_dir / "metrics.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {output_dir / 'metrics.png'}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Find most recent run
        runs = sorted(Path("runs").glob("*/metrics.jsonl"))
        if not runs:
            print("Usage: python scripts/plot_metrics.py <metrics.jsonl> [output_dir]")
            sys.exit(1)
        metrics_path = str(runs[-1])
        print(f"Using most recent: {metrics_path}")
    else:
        metrics_path = sys.argv[1]

    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    plot_metrics(metrics_path, output_dir)
