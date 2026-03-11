"""Hierarchical PC trainer.

Wraps the hierarchical grid + update rule into a training loop compatible
with the existing batch generator and management command interface.

Architecture: JIT-compiled relaxation step, Python loops for ticks and batch.
This is simpler than the flat grid's full vmap+scan approach but still fast
enough for experimentation (relaxation is the hot loop, and it's JIT'd).
"""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from training.engine.hierarchical_grid import HierarchicalGridState, create_hierarchical_grid
from training.engine.hierarchical_update import hierarchical_relaxation_step
from corpus.services.batch_generator import BatchGenerator
from corpus.services.curriculum import CurriculumScheduler


def _make_jit_relax_step(layer_sizes, lr, lr_w, lam):
    """Create a JIT-compiled relaxation step for fixed layer structure.

    Hyperparameters (lr, lr_w, lam) are closed over as constants.
    output_supervision is a runtime parameter to support annealing.
    """

    @jax.jit
    def jit_step(reps, pred_w, pred_b, skip_w, skip_b, temp_w, temp_s,
                 clamp_0, clamp_last, do_clamp_0, do_clamp_last,
                 output_target, sup_strength):
        """JIT-friendly relaxation step with optional clamping + supervision."""
        new_reps, errors, new_pred_w, new_pred_b, new_skip_w, new_skip_b, new_temp_w = \
            hierarchical_relaxation_step(
                reps, pred_w, pred_b, skip_w, skip_b, temp_w, temp_s,
                layer_sizes, lr, lr_w, lam,
                output_target=output_target,
                output_supervision=sup_strength,
            )

        # Apply clamping post-hoc (always compute both, conditionally apply)
        new_reps_out = list(new_reps)
        new_reps_out[0] = jnp.where(do_clamp_0, clamp_0, new_reps_out[0])
        new_reps_out[-1] = jnp.where(do_clamp_last, clamp_last, new_reps_out[-1])

        return (new_reps_out, errors, new_pred_w, new_pred_b,
                new_skip_w, new_skip_b, new_temp_w)

    return jit_step


class HierarchicalTrainer:
    def __init__(self, midi_dir=None, layer_sizes=None, relaxation_steps=32,
                 fs=8.0, key=None, scan_results=None, index_path=None,
                 prefetch=False, prefetch_depth=3,
                 lr=0.01, lr_w=0.001, alpha=0.8, lambda_sparse=0.005,
                 curriculum_phases=None, curriculum_patience=10,
                 teacher_forcing_ratio=1.0,
                 tf_min=0.0, tf_anneal_steps=0,
                 output_supervision=0.0,
                 sup_min=0.0, sup_anneal_steps=0):
        """Initialize hierarchical PC trainer.

        Args:
            midi_dir: path to MIDI files directory.
            layer_sizes: list of layer dimensions, e.g. [128, 64, 32, 64, 128].
            relaxation_steps: number of relaxation steps per tick.
            fs: sampling rate (ticks per second).
            key: JAX PRNG key.
            teacher_forcing_ratio: fraction of ticks where output is clamped (1.0=always).
            output_supervision: strength of soft target signal at output layer (0=none).
            sup_min: minimum supervision after annealing.
            sup_anneal_steps: steps to anneal supervision from initial to min (0=no annealing).
        """
        if layer_sizes is None:
            layer_sizes = [128, 64, 32, 64, 128]
        self.layer_sizes = layer_sizes
        self.relaxation_steps = relaxation_steps
        self.fs = fs
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self._tf_initial = teacher_forcing_ratio
        self._tf_min = tf_min
        self._tf_anneal_steps = tf_anneal_steps
        self._step_count = 0

        self.batch_gen = BatchGenerator(
            midi_dir, snippet_ticks=16, fs=fs,
            scan_results=scan_results, index_path=index_path,
        )
        if prefetch:
            from corpus.services.prefetch import PrefetchBatchGenerator
            self._batch_source = PrefetchBatchGenerator(
                self.batch_gen, queue_depth=prefetch_depth
            )
        else:
            self._batch_source = self.batch_gen

        self.curriculum = CurriculumScheduler(
            phases=curriculum_phases, patience=curriculum_patience
        )

        num_instruments = len(self.batch_gen.vocabulary)
        if key is None:
            key = jax.random.PRNGKey(0)

        self.grid = create_hierarchical_grid(
            layer_sizes=layer_sizes,
            num_instruments=num_instruments,
            key=key,
            lr=lr,
            lr_w=lr_w,
            alpha=alpha,
            lambda_sparse=lambda_sparse,
        )

        self.output_supervision = output_supervision
        self._sup_initial = output_supervision
        self._sup_min = sup_min
        self._sup_anneal_steps = sup_anneal_steps
        self._rng = np.random.default_rng(42)
        self._jit_step = _make_jit_relax_step(
            layer_sizes, lr, lr_w, lambda_sparse)

    def _build_conditioning(self, cond_raw):
        """Expand conditioning vector (num_cat,) to input layer size (H_input,)."""
        h_input = self.layer_sizes[0]
        num_cat = len(cond_raw)
        cond_start = (h_input - num_cat) // 2
        cond_full = np.zeros(h_input, dtype=np.float32)
        cond_full[cond_start:cond_start + num_cat] = cond_raw
        return cond_full

    def train_step(self, batch_size=16):
        """Run one training step.

        Returns:
            avg_error: float
            meta: dict with metrics
        """
        # Anneal teacher forcing ratio and output supervision
        self._step_count += 1
        if self._tf_anneal_steps > 0:
            progress = min(self._step_count / self._tf_anneal_steps, 1.0)
            self.teacher_forcing_ratio = (
                self._tf_initial * (1.0 - progress) + self._tf_min * progress
            )
        if self._sup_anneal_steps > 0:
            progress = min(self._step_count / self._sup_anneal_steps, 1.0)
            self.output_supervision = (
                self._sup_initial * (1.0 - progress) + self._sup_min * progress
            )

        self._batch_source.snippet_ticks = self.curriculum.snippet_ticks
        batch = self._batch_source.generate_batch(batch_size)

        h_input = self.layer_sizes[0]
        h_output = self.layer_sizes[-1]

        all_inputs = batch["input"][:, :, :h_input]     # (B, T, H_in)
        all_targets = batch["target"][:, :, :h_output]   # (B, T, H_out)
        all_conds = np.array([self._build_conditioning(c) for c in batch["conditioning"]])

        # Accumulate weight updates across batch
        acc_pred_w = [jnp.zeros_like(w) for w in self.grid.prediction_weights]
        acc_pred_b = [jnp.zeros_like(b) for b in self.grid.prediction_biases]
        acc_skip_w = [jnp.zeros_like(w) for w in self.grid.skip_weights]
        acc_skip_b = [jnp.zeros_like(b) for b in self.grid.skip_biases]
        acc_temp_w = [jnp.zeros_like(w) for w in self.grid.temporal_weights]

        total_error = 0.0
        total_active_error = 0.0
        total_tp, total_fp, total_fn = 0.0, 0.0, 0.0

        for b in range(batch_size):
            inp_seq = all_inputs[b]   # (T, H_in)
            tgt_seq = all_targets[b]  # (T, H_out)
            cond = all_conds[b]       # (H_in,)

            ex_pred_w, ex_pred_b, ex_skip_w, ex_skip_b, ex_temp_w, \
                ex_error, ex_active, tp, fp, fn = \
                self._process_example(inp_seq, tgt_seq, cond)

            for i in range(len(acc_pred_w)):
                acc_pred_w[i] = acc_pred_w[i] + ex_pred_w[i]
            for i in range(len(acc_pred_b)):
                acc_pred_b[i] = acc_pred_b[i] + ex_pred_b[i]
            for i in range(len(acc_skip_w)):
                acc_skip_w[i] = acc_skip_w[i] + ex_skip_w[i]
            for i in range(len(acc_skip_b)):
                acc_skip_b[i] = acc_skip_b[i] + ex_skip_b[i]
            for i in range(len(acc_temp_w)):
                acc_temp_w[i] = acc_temp_w[i] + ex_temp_w[i]

            total_error += ex_error
            total_active_error += ex_active
            total_tp += tp
            total_fp += fp
            total_fn += fn

        # Average across batch
        self.grid.prediction_weights = [w / batch_size for w in acc_pred_w]
        self.grid.prediction_biases = [b / batch_size for b in acc_pred_b]
        self.grid.skip_weights = [w / batch_size for w in acc_skip_w]
        self.grid.skip_biases = [b / batch_size for b in acc_skip_b]
        self.grid.temporal_weights = [w / batch_size for w in acc_temp_w]

        avg_error = total_error / batch_size
        avg_active = total_active_error / batch_size
        self.curriculum.report_error(avg_error)

        precision = total_tp / max(total_tp + total_fp, 1.0)
        recall = total_tp / max(total_tp + total_fn, 1.0)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        return avg_error, {
            "datasets": batch.get("datasets", []),
            "active_error": avg_active,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "teacher_forcing": self.teacher_forcing_ratio,
        }

    def _process_example(self, inp_seq, tgt_seq, cond):
        """Process one example: iterate over ticks with relaxation.

        Returns updated weights and error metrics.
        """
        T = inp_seq.shape[0]
        reps = list(self.grid.representations)
        pred_w = list(self.grid.prediction_weights)
        pred_b = list(self.grid.prediction_biases)
        skip_w = list(self.grid.skip_weights)
        skip_b = list(self.grid.skip_biases)
        temp_w = list(self.grid.temporal_weights)
        temp_s = list(self.grid.temporal_state)

        total_error = 0.0
        total_active = 0.0
        last_output = None

        for t in range(T):
            inp = jnp.array(inp_seq[t])   # (H_in,)
            tgt = jnp.array(tgt_seq[t])   # (H_out,)

            # Add conditioning to input
            input_with_cond = inp + jnp.array(cond)
            input_with_cond = jnp.clip(input_with_cond, 0.0, 1.0)

            # Convert target to logit space for output clamping:
            # sigmoid(3.0) ≈ 0.95, sigmoid(-3.0) ≈ 0.05
            # This gives the representations a meaningful dynamic range
            tgt_logit = jnp.where(tgt > 0.5, 3.0, -3.0)

            # Teacher forcing: clamp output on some ticks
            do_clamp_output = self._rng.random() < self.teacher_forcing_ratio

            # Relaxation loop (JIT-compiled inner step)
            do_clamp_0 = jnp.bool_(True)
            do_clamp_last = jnp.bool_(do_clamp_output)
            sup = jnp.float32(self.output_supervision)
            for step in range(self.relaxation_steps):
                reps, errors, pred_w, pred_b, skip_w, skip_b, temp_w = self._jit_step(
                    reps, pred_w, pred_b, skip_w, skip_b, temp_w, temp_s,
                    input_with_cond, tgt_logit,
                    do_clamp_0, do_clamp_last,
                    tgt_logit, sup,
                )

            # Compute error at output layer
            output = jax.nn.sigmoid(reps[-1])
            error = float(jnp.mean(jnp.abs(output - tgt)))
            total_error += error

            # Active error (only on non-zero target notes)
            active_mask = tgt > 0.0
            active_count = float(jnp.sum(active_mask))
            if active_count > 0:
                active_err = float(jnp.sum(jnp.abs(output - tgt) * active_mask) / active_count)
                total_active += active_err

            # Update temporal state for next tick
            temp_s = [r.copy() for r in reps]
            last_output = output

        # F1 at last tick
        if last_output is not None:
            pred_binary = (last_output > 0.5).astype(jnp.float32)
            target_binary = (jnp.array(tgt_seq[-1]) > 0.0).astype(jnp.float32)
            tp = float(jnp.sum(pred_binary * target_binary))
            fp = float(jnp.sum(pred_binary * (1.0 - target_binary)))
            fn = float(jnp.sum((1.0 - pred_binary) * target_binary))
        else:
            tp, fp, fn = 0.0, 0.0, 0.0

        return pred_w, pred_b, skip_w, skip_b, temp_w, total_error / max(T, 1), total_active / max(T, 1), tp, fp, fn  # noqa: E501

    def save_checkpoint(self, path):
        """Save hierarchical grid state to directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save per-layer arrays
        for i, r in enumerate(self.grid.representations):
            np.save(path / f"layer_{i}_rep.npy", np.array(r))
        for i, e in enumerate(self.grid.errors):
            np.save(path / f"layer_{i}_err.npy", np.array(e))
        for i, ts in enumerate(self.grid.temporal_state):
            np.save(path / f"layer_{i}_temporal.npy", np.array(ts))

        # Save weights
        for i, w in enumerate(self.grid.prediction_weights):
            np.save(path / f"pred_weight_{i}.npy", np.array(w))
        for i, b in enumerate(self.grid.prediction_biases):
            np.save(path / f"pred_bias_{i}.npy", np.array(b))
        for i, w in enumerate(self.grid.skip_weights):
            np.save(path / f"skip_weight_{i}.npy", np.array(w))
        for i, b in enumerate(self.grid.skip_biases):
            np.save(path / f"skip_bias_{i}.npy", np.array(b))
        for i, w in enumerate(self.grid.temporal_weights):
            np.save(path / f"temporal_weight_{i}.npy", np.array(w))

        # Metadata
        metadata = {
            "architecture": "hierarchical",
            "layer_sizes": self.layer_sizes,
            "vocabulary": self.batch_gen.vocabulary,
            "conditioning_size": self.grid.conditioning_size,
            "lr": self.grid.lr,
            "lr_w": self.grid.lr_w,
            "alpha": self.grid.alpha,
            "lambda_sparse": self.grid.lambda_sparse,
            "teacher_forcing_ratio": self.teacher_forcing_ratio,
        }
        (path / "metadata.json").write_text(json.dumps(metadata))

    def load_checkpoint(self, path):
        """Load hierarchical grid state from directory."""
        path = Path(path)
        metadata = json.loads((path / "metadata.json").read_text())

        for i in range(len(self.grid.representations)):
            f = path / f"layer_{i}_rep.npy"
            if f.exists():
                self.grid.representations[i] = jnp.array(np.load(f))

        for i in range(len(self.grid.prediction_weights)):
            f = path / f"pred_weight_{i}.npy"
            if f.exists():
                self.grid.prediction_weights[i] = jnp.array(np.load(f))

        for i in range(len(self.grid.prediction_biases)):
            f = path / f"pred_bias_{i}.npy"
            if f.exists():
                self.grid.prediction_biases[i] = jnp.array(np.load(f))

        for i in range(len(self.grid.skip_weights)):
            f = path / f"skip_weight_{i}.npy"
            if f.exists():
                self.grid.skip_weights[i] = jnp.array(np.load(f))

        for i in range(len(self.grid.skip_biases)):
            f = path / f"skip_bias_{i}.npy"
            if f.exists():
                self.grid.skip_biases[i] = jnp.array(np.load(f))

        for i in range(len(self.grid.temporal_weights)):
            f = path / f"temporal_weight_{i}.npy"
            if f.exists():
                self.grid.temporal_weights[i] = jnp.array(np.load(f))

        for i in range(len(self.grid.temporal_state)):
            f = path / f"layer_{i}_temporal.npy"
            if f.exists():
                self.grid.temporal_state[i] = jnp.array(np.load(f))
