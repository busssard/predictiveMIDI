"""Diagnostic command: analyze a trained PC grid for common failure modes.

Tests:
1. Signal propagation: clamp a single note, measure activation decay across columns
2. Weight analysis: magnitude, distribution, dead weights
3. Column energy profile: mean |error| per column
4. Temporal coherence: how much output changes across ticks
5. Output diversity: number of unique pitch patterns produced
"""
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from django.core.management.base import BaseCommand

from training.engine.inference import load_checkpoint, run_inference
from training.engine.update_rule import ACTIVATIONS


class Command(BaseCommand):
    help = "Diagnose a trained PC grid for common failure modes"

    def add_arguments(self, parser):
        parser.add_argument("--checkpoint", required=True,
                            help="Path to checkpoint directory")
        parser.add_argument("--relaxation-steps", type=int, default=64)
        parser.add_argument("--fs", type=float, default=8.0)
        parser.add_argument("--output-dir", default=None,
                            help="Directory for diagnostic plots (default: <checkpoint>/diagnostics)")

    def handle(self, *args, **options):
        checkpoint_path = options["checkpoint"]
        relax_steps = options["relaxation_steps"]

        self.stdout.write(f"Loading checkpoint from {checkpoint_path}...")
        grid, metadata = load_checkpoint(checkpoint_path)
        vocabulary = metadata.get("vocabulary", {})

        H, W = grid.state.shape[0], grid.state.shape[1]
        self.stdout.write(f"Grid: {H}x{W}, activation={metadata.get('activation')}, "
                          f"connectivity={metadata.get('connectivity')}")

        output_dir = options.get("output_dir")
        if output_dir is None:
            output_dir = str(Path(checkpoint_path) / "diagnostics")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # ── Test 1: Signal Propagation ──
        self.stdout.write("\n=== Test 1: Signal Propagation ===")
        self._test_signal_propagation(grid, metadata, H, W, relax_steps, output_dir)

        # ── Test 2: Weight Analysis ──
        self.stdout.write("\n=== Test 2: Weight Analysis ===")
        self._test_weights(grid, H, W, output_dir)

        # ── Test 3: Column Energy Profile ──
        self.stdout.write("\n=== Test 3: Column Energy Profile ===")
        self._test_column_energy(grid, H, W)

        # ── Test 4: Temporal Coherence ──
        self.stdout.write("\n=== Test 4: Temporal Coherence ===")
        self._test_temporal_coherence(grid, metadata, vocabulary, relax_steps, output_dir)

        # ── Test 5: Precision Analysis ──
        self.stdout.write("\n=== Test 5: Precision Analysis ===")
        self._test_precision(grid, H, W)

        self.stdout.write(f"\nDiagnostic plots saved to: {output_dir}")

    def _test_signal_propagation(self, grid, metadata, H, W, relax_steps, output_dir):
        """Clamp single notes at different pitches, measure how far signal reaches."""
        test_pitches = [36, 48, 60, 72, 84]  # C2, C3, C4, C5, C6
        vocabulary = metadata.get("vocabulary", {})

        results = {}
        for pitch in test_pitches:
            # Create input with a single sustained note
            T = 8  # 8 ticks = 1 second
            input_seq = np.zeros((T, 128), dtype=np.float32)
            input_seq[:, pitch] = 1.0

            # Conditioning: piano
            cond = np.zeros(len(vocabulary), dtype=np.float32)
            if "piano" in vocabulary:
                cond[vocabulary["piano"]] = 1.0

            output_seq, all_states = run_inference(
                grid, metadata, input_seq, cond,
                relaxation_steps=relax_steps,
            )

            # Mean |activation| per column at the last tick
            last_state = all_states[-1]  # (H, W, 4)
            col_act = np.abs(last_state[:, :, 0]).mean(axis=0)  # (W,)
            col_err = np.abs(last_state[:, :, 1]).mean(axis=0)

            prop_ratio = col_act[-1] / max(col_act[0], 1e-8)
            output_max = output_seq[-1].max()

            results[pitch] = {
                "col_activations": col_act,
                "col_errors": col_err,
                "propagation_ratio": prop_ratio,
                "output_max": float(output_max),
            }

            self.stdout.write(
                f"  Pitch {pitch:3d}: propagation={prop_ratio:.6f}, "
                f"output_max={output_max:.4f}, "
                f"col_act=[{col_act[0]:.4f} → {col_act[W//2]:.4f} → {col_act[-1]:.4f}]")

        # Plot
        self._plot_propagation(results, W, output_dir)

    def _test_weights(self, grid, H, W, output_dir):
        """Analyze weight distributions."""
        w = np.array(grid.weights)  # (H, W, 4)
        labels = ["w_left", "w_right", "w_up", "w_down"]

        for i, label in enumerate(labels):
            wi = w[:, :, i]
            self.stdout.write(
                f"  {label:8s}: mean={wi.mean():.6f}, std={wi.std():.6f}, "
                f"|mean|={np.abs(wi).mean():.6f}, "
                f"min={wi.min():.4f}, max={wi.max():.4f}")

        # Per-column weight magnitude
        w_mag = np.sqrt(np.sum(w ** 2, axis=-1))  # (H, W)
        col_mag = w_mag.mean(axis=0)
        self.stdout.write(f"  Per-column ||w||: {', '.join(f'c{i}={v:.4f}' for i, v in enumerate(col_mag))}")

        # Dead weights (near zero)
        dead = (np.abs(w).max(axis=-1) < 0.01).sum()
        self.stdout.write(f"  Dead neurons (max |w| < 0.01): {dead}/{H*W} ({100*dead/(H*W):.1f}%)")

        # Weight symmetry: are left/right weights similar?
        lr_corr = np.corrcoef(w[:, :, 0].flatten(), w[:, :, 1].flatten())[0, 1]
        ud_corr = np.corrcoef(w[:, :, 2].flatten(), w[:, :, 3].flatten())[0, 1]
        self.stdout.write(f"  Weight correlation: left/right={lr_corr:.3f}, up/down={ud_corr:.3f}")

        self._plot_weights(w, output_dir)

    def _test_column_energy(self, grid, H, W):
        """Report stored error per column from current grid state."""
        state = np.array(grid.state)
        error = np.abs(state[:, :, 1])  # (H, W)
        col_energy = error.mean(axis=0)

        self.stdout.write(f"  Stored |error| per column: "
                          f"{', '.join(f'c{i}={v:.6f}' for i, v in enumerate(col_energy))}")

        boundary_ratio = (col_energy[0] + col_energy[-1]) / (2 * max(col_energy[1:-1].mean(), 1e-8))
        self.stdout.write(f"  Boundary/interior energy ratio: {boundary_ratio:.2f}x")

    def _test_temporal_coherence(self, grid, metadata, vocabulary, relax_steps, output_dir):
        """Feed a simple repeating pattern and check output stability."""
        H = grid.state.shape[0]
        T = 32  # 4 seconds

        # Create simple alternating pattern: C4 and E4
        input_seq = np.zeros((T, 128), dtype=np.float32)
        for t in range(T):
            if t % 4 < 2:
                input_seq[t, 60] = 1.0  # C4
            else:
                input_seq[t, 64] = 1.0  # E4

        cond = np.zeros(len(vocabulary), dtype=np.float32)
        if "piano" in vocabulary:
            cond[vocabulary["piano"]] = 1.0

        output_seq, _ = run_inference(
            grid, metadata, input_seq, cond,
            relaxation_steps=relax_steps,
        )

        # Temporal autocorrelation of output
        output_binary = (output_seq > 0.5).astype(np.float32)
        if T > 1:
            diffs = np.abs(output_binary[1:] - output_binary[:-1]).mean(axis=1)
            autocorr = 1.0 - diffs.mean()
        else:
            autocorr = 1.0

        # Unique patterns
        patterns = set()
        for t in range(T):
            pattern = tuple(np.nonzero(output_binary[t])[0])
            patterns.add(pattern)

        self.stdout.write(f"  Temporal autocorrelation: {autocorr:.4f} (1.0=frozen, 0.0=random)")
        self.stdout.write(f"  Unique output patterns: {len(patterns)}/{T}")
        self.stdout.write(f"  Mean notes per tick: {output_binary.sum(axis=1).mean():.1f}")

        if len(patterns) <= 5:
            for i, pat in enumerate(sorted(patterns)):
                self.stdout.write(f"    Pattern {i}: pitches={list(pat)}")

        self._plot_temporal(input_seq, output_seq, output_binary, output_dir)

    def _test_precision(self, grid, H, W):
        """Analyze learned precision values."""
        lp = np.array(grid.log_precision)  # (H, W)
        precision = np.exp(lp)

        self.stdout.write(f"  Log-precision: mean={lp.mean():.4f}, std={lp.std():.4f}, "
                          f"min={lp.min():.4f}, max={lp.max():.4f}")
        self.stdout.write(f"  Precision (exp): mean={precision.mean():.4f}, "
                          f"min={precision.min():.4f}, max={precision.max():.4f}")

        col_prec = precision.mean(axis=0)
        self.stdout.write(f"  Per-column precision: "
                          f"{', '.join(f'c{i}={v:.4f}' for i, v in enumerate(col_prec))}")

    def _plot_propagation(self, results, W, output_dir):
        """Plot signal propagation across columns for each test pitch."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for pitch, data in results.items():
            axes[0].plot(range(W), data["col_activations"],
                         label=f"pitch {pitch}", marker="o", markersize=3)
            axes[1].plot(range(W), data["col_errors"],
                         label=f"pitch {pitch}", marker="o", markersize=3)

        axes[0].set_xlabel("Column")
        axes[0].set_ylabel("Mean |activation|")
        axes[0].set_title("Signal Propagation: Activation Decay")
        axes[0].legend()
        axes[0].set_yscale("log")

        axes[1].set_xlabel("Column")
        axes[1].set_ylabel("Mean |error|")
        axes[1].set_title("Error Distribution Across Columns")
        axes[1].legend()
        axes[1].set_yscale("log")

        plt.tight_layout()
        plt.savefig(Path(output_dir) / "signal_propagation.png", dpi=100)
        plt.close()

    def _plot_weights(self, w, output_dir):
        """Plot weight magnitude heatmaps."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        w_mag = np.sqrt(np.sum(w ** 2, axis=-1))
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        im0 = axes[0].imshow(w_mag, aspect="auto", origin="lower", cmap="viridis")
        axes[0].set_xlabel("Column")
        axes[0].set_ylabel("Pitch")
        axes[0].set_title("Weight Magnitude ||w||₂")
        plt.colorbar(im0, ax=axes[0])

        # Weight direction: which direction dominates?
        w_abs = np.abs(w)
        dominant = np.argmax(w_abs, axis=-1)  # 0=left, 1=right, 2=up, 3=down
        im1 = axes[1].imshow(dominant, aspect="auto", origin="lower", cmap="tab10", vmin=0, vmax=3)
        axes[1].set_xlabel("Column")
        axes[1].set_ylabel("Pitch")
        axes[1].set_title("Dominant Weight Direction (0=L, 1=R, 2=U, 3=D)")
        plt.colorbar(im1, ax=axes[1])

        plt.tight_layout()
        plt.savefig(Path(output_dir) / "weight_analysis.png", dpi=100)
        plt.close()

    def _plot_temporal(self, input_seq, output_seq, output_binary, output_dir):
        """Plot temporal coherence test results."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

        axes[0].imshow(input_seq.T, aspect="auto", origin="lower",
                       cmap="Blues", interpolation="nearest")
        axes[0].set_ylabel("Pitch")
        axes[0].set_title("Input: Alternating C4/E4")
        axes[0].set_ylim(48, 80)

        im = axes[1].imshow(output_seq.T, aspect="auto", origin="lower",
                            cmap="Reds", interpolation="nearest", vmin=0, vmax=1)
        axes[1].set_ylabel("Pitch")
        axes[1].set_title("Output Probabilities")
        axes[1].set_ylim(48, 80)
        plt.colorbar(im, ax=axes[1], shrink=0.6)

        axes[2].imshow(output_binary.T, aspect="auto", origin="lower",
                       cmap="Greens", interpolation="nearest")
        axes[2].set_ylabel("Pitch")
        axes[2].set_xlabel("Tick")
        axes[2].set_title("Binary Output (threshold=0.5)")
        axes[2].set_ylim(48, 80)

        plt.tight_layout()
        plt.savefig(Path(output_dir) / "temporal_coherence.png", dpi=100)
        plt.close()
