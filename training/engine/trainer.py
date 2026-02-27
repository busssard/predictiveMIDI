from pathlib import Path
import functools
import jax
import jax.numpy as jnp
import jax.lax as lax
import numpy as np
from training.engine.grid import create_grid
from training.engine.update_rule import pc_relaxation_step, apply_clamping, ACTIVATIONS
from corpus.services.batch_generator import BatchGenerator
from corpus.services.curriculum import CurriculumScheduler


def _make_train_fn(relaxation_steps, activation_fn=None):
    """Build a JIT-compiled function that processes a full batch on GPU.

    Uses lax.scan over ticks (sequential — state carries forward) and
    jax.vmap over the batch dimension (parallel — independent examples).
    """
    if activation_fn is None:
        activation_fn = jnp.tanh

    @jax.jit
    def train_fn(state, weights, params,
                 all_inputs, all_targets, all_conds,
                 input_mask, output_mask, conditioning_mask):
        """
        Args:
            state:   (H, W, 4)  — initial grid state (shared across batch)
            weights: (H, W, 4)  — initial weights (shared across batch)
            params:  (H, W, 4)  — grid params (fixed)
            all_inputs:  (B, T, H) — input values per batch item per tick
            all_targets: (B, T, H) — target values per batch item per tick
            all_conds:   (B, H)    — conditioning values per batch item
            input_mask, output_mask, conditioning_mask: (H, W) bool

        Returns:
            new_weights: (H, W, 4) — averaged across batch
            avg_error: scalar
        """

        def process_example(inputs_seq, targets_seq, cond_vals):
            """Process one example: scan over ticks."""

            def process_tick(carry, tick_data):
                st, wt = carry
                inp, tgt = tick_data

                def relax_step(carry, _):
                    s, w = carry
                    s, w = pc_relaxation_step(s, w, params, activation_fn)
                    s = apply_clamping(s, input_mask, inp, channel=0)
                    s = apply_clamping(s, output_mask, tgt, channel=0)
                    s = apply_clamping(s, conditioning_mask, cond_vals, channel=0)
                    return (s, w), jnp.abs(s[:, :, 1]).mean()

                (new_st, new_wt), errors = lax.scan(
                    relax_step, (st, wt), None, length=relaxation_steps
                )
                return (new_st, new_wt), errors[-1]

            (_, final_weights), tick_errors = lax.scan(
                process_tick, (state, weights), (inputs_seq, targets_seq)
            )
            return final_weights, tick_errors.mean()

        # vmap over batch: each example starts from same state/weights
        all_weights, all_errors = jax.vmap(process_example)(
            all_inputs, all_targets, all_conds
        )

        # Average weight updates across batch (mini-batch iPC)
        new_weights = jnp.mean(all_weights, axis=0)
        avg_error = jnp.mean(all_errors)

        return new_weights, avg_error

    return train_fn


class Trainer:
    def __init__(self, midi_dir, grid_size=128, relaxation_steps=128,
                 fs=8.0, key=None, scan_results=None, index_path=None,
                 prefetch=False, prefetch_depth=3, activation="tanh"):
        self.grid_size = grid_size
        self.relaxation_steps = relaxation_steps
        self.fs = fs
        self.activation_name = activation
        self._activation_fn = ACTIVATIONS.get(activation, jnp.tanh)

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
        self.curriculum = CurriculumScheduler()
        num_instruments = len(self.batch_gen.vocabulary)

        if key is None:
            key = jax.random.PRNGKey(0)
        self.grid = create_grid(
            size=grid_size, num_instruments=num_instruments, key=key
        )

        self._train_fn = _make_train_fn(relaxation_steps, self._activation_fn)

    def _build_all_conds(self, conditioning):
        """Vectorized: expand conditioning vectors for the whole batch."""
        batch_size, num_cat = conditioning.shape
        cond_start = (self.grid_size - num_cat) // 2
        all_conds = jnp.zeros((batch_size, self.grid_size))
        all_conds = all_conds.at[:, cond_start:cond_start + num_cat].set(
            jnp.array(conditioning)
        )
        return all_conds

    def train_step(self, batch_size=32):
        """Run one training step: generate batch, process on GPU."""
        self._batch_source.snippet_ticks = self.curriculum.snippet_ticks
        batch = self._batch_source.generate_batch(batch_size)

        all_inputs = jnp.array(batch["input"][:, :, :self.grid_size])
        all_targets = jnp.array(batch["target"][:, :, :self.grid_size])
        all_conds = self._build_all_conds(batch["conditioning"])

        new_weights, avg_error = self._train_fn(
            self.grid.state, self.grid.weights, self.grid.params,
            all_inputs, all_targets, all_conds,
            self.grid.input_mask, self.grid.output_mask,
            self.grid.conditioning_mask,
        )

        self.grid.weights = new_weights
        avg_error = float(avg_error)
        self.curriculum.report_error(avg_error)
        return avg_error

    def evaluate_error(self, batch_size=4):
        """Evaluate current error without updating weights."""
        self._batch_source.snippet_ticks = self.curriculum.snippet_ticks
        batch = self._batch_source.generate_batch(batch_size)

        all_inputs = jnp.array(batch["input"][:, :, :self.grid_size])
        all_targets = jnp.array(batch["target"][:, :, :self.grid_size])
        all_conds = self._build_all_conds(batch["conditioning"])

        _, avg_error = self._train_fn(
            self.grid.state, self.grid.weights, self.grid.params,
            all_inputs, all_targets, all_conds,
            self.grid.input_mask, self.grid.output_mask,
            self.grid.conditioning_mask,
        )
        return float(avg_error)

    def save_checkpoint(self, path):
        """Save grid state, weights, and params as numpy arrays."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "state.npy", np.array(self.grid.state))
        np.save(path / "weights.npy", np.array(self.grid.weights))
        np.save(path / "params.npy", np.array(self.grid.params))

    def load_checkpoint(self, path):
        """Load grid state, weights, and params from numpy arrays."""
        path = Path(path)
        self.grid.state = jnp.array(np.load(path / "state.npy"))
        self.grid.weights = jnp.array(np.load(path / "weights.npy"))
        self.grid.params = jnp.array(np.load(path / "params.npy"))
