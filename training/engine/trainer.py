from pathlib import Path
import jax
import jax.numpy as jnp
import jax.lax as lax
import numpy as np
from training.engine.grid import create_grid
from training.engine.update_rule import pc_relaxation_step, apply_clamping
from corpus.services.batch_generator import BatchGenerator
from corpus.services.curriculum import CurriculumScheduler


class Trainer:
    def __init__(self, midi_dir, grid_size=128, relaxation_steps=128,
                 fs=8.0, key=None):
        self.grid_size = grid_size
        self.relaxation_steps = relaxation_steps
        self.fs = fs

        self.batch_gen = BatchGenerator(midi_dir, snippet_ticks=16, fs=fs)
        self.curriculum = CurriculumScheduler()
        num_instruments = len(self.batch_gen.vocabulary)

        if key is None:
            key = jax.random.PRNGKey(0)
        self.grid = create_grid(
            size=grid_size, num_instruments=num_instruments, key=key
        )

    def _build_cond_vals(self, cond_vec):
        """Expand one-hot conditioning vector to grid-row-sized array."""
        cond_vals = jnp.zeros(self.grid_size)
        num_inst = len(cond_vec)
        cond_start = (self.grid_size - num_inst) // 2
        cond_vals = cond_vals.at[cond_start:cond_start + num_inst].set(
            jnp.array(cond_vec)
        )
        return cond_vals

    def _run_tick(self, state, weights, params, input_vals, target_vals,
                  cond_vals, input_mask, output_mask, conditioning_mask,
                  steps, training=True):
        """Run relaxation steps for one musical tick."""
        def step_fn(carry, _):
            st, wt = carry
            st, wt = pc_relaxation_step(st, wt, params)
            # Re-clamp
            st = apply_clamping(st, input_mask, input_vals, channel=0)
            if training:
                st = apply_clamping(st, output_mask, target_vals, channel=0)
            st = apply_clamping(st, conditioning_mask, cond_vals, channel=0)
            error = jnp.abs(st[:, :, 1]).mean()
            return (st, wt), error

        (final_state, final_weights), errors = lax.scan(
            step_fn, (state, weights), None, length=steps
        )
        return final_state, final_weights, errors

    def train_step(self, batch_size=32):
        """Run one training step: generate batch, process each tick."""
        self.batch_gen.snippet_ticks = self.curriculum.snippet_ticks
        batch = self.batch_gen.generate_batch(batch_size)

        total_error = 0.0
        num_ticks = batch["input"].shape[1]

        for b in range(batch_size):
            state = self.grid.state
            weights = self.grid.weights

            for t in range(num_ticks):
                input_vals = jnp.array(batch["input"][b, t, :self.grid_size])
                target_vals = jnp.array(batch["target"][b, t, :self.grid_size])
                cond_vals = self._build_cond_vals(batch["conditioning"][b])

                state, weights, errors = self._run_tick(
                    state, weights, self.grid.params,
                    input_vals, target_vals, cond_vals,
                    self.grid.input_mask, self.grid.output_mask,
                    self.grid.conditioning_mask,
                    self.relaxation_steps,
                    training=True,
                )
                total_error += float(errors[-1])

            self.grid.weights = weights

        avg_error = total_error / (batch_size * num_ticks)
        self.curriculum.report_error(avg_error)
        return avg_error

    def evaluate_error(self, batch_size=4):
        """Evaluate current error without updating weights."""
        self.batch_gen.snippet_ticks = self.curriculum.snippet_ticks
        batch = self.batch_gen.generate_batch(batch_size)

        total_error = 0.0
        num_ticks = batch["input"].shape[1]

        for b in range(batch_size):
            state = self.grid.state
            for t in range(num_ticks):
                input_vals = jnp.array(batch["input"][b, t, :self.grid_size])
                target_vals = jnp.array(batch["target"][b, t, :self.grid_size])
                cond_vals = self._build_cond_vals(batch["conditioning"][b])

                state, _, errors = self._run_tick(
                    state, self.grid.weights, self.grid.params,
                    input_vals, target_vals, cond_vals,
                    self.grid.input_mask, self.grid.output_mask,
                    self.grid.conditioning_mask,
                    self.relaxation_steps,
                    training=True,
                )
                total_error += float(errors[-1])

        return total_error / (batch_size * num_ticks)

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
