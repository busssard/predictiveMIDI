import json
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


def _make_train_fn(relaxation_steps, activation_fn=None,
                   pos_weight=1.0, lambda_sparse=0.0, lr_precision=0.001,
                   lr_w=None, spike_boost=0.0,
                   asl_gamma_neg=0.0, asl_margin=0.05,
                   use_fc=False, use_fc_skip=False, use_w_temporal=False,
                   use_forward_init=False):
    """Build a JIT-compiled function that processes a full batch on GPU.

    Uses lax.scan over ticks (sequential — state carries forward) and
    jax.vmap over the batch dimension (parallel — independent examples).
    """
    if activation_fn is None:
        activation_fn = jax.nn.leaky_relu

    @jax.jit
    def train_fn(state, weights, params, log_precision,
                 all_inputs, all_targets, all_conds,
                 input_mask, output_mask, conditioning_mask,
                 fc_weights, fc_skip_weights, w_temporal):
        """
        Args:
            state:   (H, W, 4)
            weights: (H, W, 4)
            params:  (H, W, 4)
            log_precision: (H, W)
            all_inputs:  (B, T, H)
            all_targets: (B, T, H)
            all_conds:   (B, H)
            input_mask, output_mask, conditioning_mask: (H, W) bool
            fc_weights: (W, H, H) or dummy
            fc_skip_weights: (W, H, H) or dummy
            w_temporal: (H, W) or dummy

        Returns:
            new_weights: (H, W, 4)
            new_log_precision: (H, W)
            avg_error: scalar
            avg_active_error: scalar
            col_energy: (W,)
            avg_final_state: (H, W, 4)
            new_fc_weights, new_fc_skip_weights, new_w_temporal
        """
        # Conditionals resolved at trace time via use_fc etc.
        _fc_w = fc_weights if use_fc else None
        _fc_skip_w = fc_skip_weights if use_fc_skip else None
        _w_temp = w_temporal if use_w_temporal else None

        def process_example(inputs_seq, targets_seq, cond_vals):
            """Process one example: scan over ticks."""

            def process_tick(carry, tick_data):
                st, wt, lp, fcw, fcsw, wt_temp = carry
                inp, tgt = tick_data
                H = st.shape[0]
                W = st.shape[1]

                # Forward initialization: linear interpolation
                if use_forward_init:
                    col_fracs = jnp.linspace(0.0, 1.0, W)[None, :]  # (1, W)
                    interp = inp[:, None] * (1 - col_fracs) + tgt[:, None] * col_fracs
                    interior = ~(input_mask | output_mask)
                    st = st.at[:, :, 0].set(
                        jnp.where(interior, interp, st[:, :, 0]))

                # Build boundary precision mask (pos_weight for active notes)
                precision = jnp.ones((H, W))
                precision = precision.at[:, 0].set(
                    jnp.where(inp != 0.0, pos_weight, 1.0))

                # ASL at output boundary
                if asl_gamma_neg > 0:
                    pred_sig = jax.nn.sigmoid(st[:, -1, 0])
                    shifted = jnp.maximum(pred_sig - asl_margin, 0.0)
                    neg_w = shifted ** asl_gamma_neg
                    asl_col = jnp.where(tgt != 0.0, pos_weight, neg_w)
                    precision = precision.at[:, -1].set(asl_col)
                else:
                    precision = precision.at[:, -1].set(
                        jnp.where(tgt != 0.0, pos_weight, 1.0))

                # Relaxation scan with FISTA and step index
                step_indices = jnp.arange(relaxation_steps)

                def relax_step(carry, step_idx):
                    s, w, log_p, r_prev, fc_w, fc_sw, w_t = carry
                    (s, w, new_log_p,
                     new_fc_w, new_fc_sw, new_w_t) = pc_relaxation_step(
                        s, w, params, activation_fn,
                        precision=precision, lambda_sparse=lambda_sparse,
                        log_precision=log_p, lr_precision=lr_precision,
                        fc_weights=fc_w, fc_skip_weights=fc_sw,
                        lr_w=lr_w, step_index=step_idx, r_prev=r_prev,
                        spike_boost=spike_boost, w_temporal=w_t,
                    )
                    s = apply_clamping(s, input_mask, inp, channel=0)
                    s = apply_clamping(s, output_mask, tgt, channel=0)
                    s = apply_clamping(s, conditioning_mask, cond_vals,
                                       channel=0)

                    full_err = jnp.abs(s[:, :, 1]).mean()
                    out_e = s[:, -1, 1]
                    active = (tgt != 0.0)
                    active_err = (jnp.where(active, jnp.abs(out_e), 0.0).sum()
                                  / jnp.maximum(active.sum(), 1.0))

                    # Track r for FISTA
                    new_r_prev = s[:, :, 0]

                    # Pass through fc/skip/temporal weights
                    out_fc_w = new_fc_w if use_fc else fc_w
                    out_fc_sw = new_fc_sw if use_fc_skip else fc_sw
                    out_w_t = new_w_t if use_w_temporal else w_t

                    return ((s, w, new_log_p, new_r_prev,
                             out_fc_w, out_fc_sw, out_w_t),
                            (full_err, active_err))

                r_prev_init = st[:, :, 0]
                ((new_st, new_wt, new_lp, _, new_fcw, new_fcsw, new_wt_temp),
                 (full_errors, active_errors)) = lax.scan(
                    relax_step,
                    (st, wt, lp, r_prev_init, fcw, fcsw, wt_temp),
                    step_indices,
                )
                # Hebbian update for w_temporal: once per tick, not per relaxation step
                if use_w_temporal:
                    final_r = new_st[:, :, 0]
                    final_h = new_st[:, :, 3]
                    lr_grid = params[:, :, 2]
                    new_wt_temp = new_wt_temp + lr_grid * 0.01 * activation_fn(final_h) * final_r

                return ((new_st, new_wt, new_lp, new_fcw, new_fcsw, new_wt_temp),
                        (full_errors[-1], active_errors[-1]))

            init_carry = (state, weights, log_precision,
                          _fc_w, _fc_skip_w, _w_temp)
            ((final_state, final_weights, final_lp, final_fcw, final_fcsw, final_wt),
             (tick_errors, tick_active)) = lax.scan(
                process_tick, init_carry, (inputs_seq, targets_seq))

            # F1 components at output column (last tick)
            pred_out = jax.nn.sigmoid(final_state[:, -1, 0])  # (H,)
            last_target = targets_seq[-1]  # (H,)
            pred_binary = (pred_out > 0.5).astype(jnp.float32)
            target_binary = (last_target > 0.0).astype(jnp.float32)
            tp = jnp.sum(pred_binary * target_binary)
            fp = jnp.sum(pred_binary * (1.0 - target_binary))
            fn = jnp.sum((1.0 - pred_binary) * target_binary)

            return (final_weights, final_lp,
                    tick_errors.mean(), tick_active.mean(),
                    final_fcw, final_fcsw, final_wt,
                    tp, fp, fn)

        # vmap over batch: each example starts from same state/weights
        (all_weights, all_lp, all_errors, all_active,
         all_fcw, all_fcsw, all_wt,
         all_tp, all_fp, all_fn) = jax.vmap(
            process_example)(all_inputs, all_targets, all_conds)

        # Average weight updates across batch
        new_weights = jnp.mean(all_weights, axis=0)
        new_lp = jnp.mean(all_lp, axis=0)
        avg_error = jnp.mean(all_errors)
        avg_active = jnp.mean(all_active)

        # Sum F1 components across batch
        total_tp = jnp.sum(all_tp)
        total_fp = jnp.sum(all_fp)
        total_fn = jnp.sum(all_fn)

        # Average FC weights across batch if present
        new_fc_w = jnp.mean(all_fcw, axis=0) if use_fc else fc_weights
        new_fc_sw = jnp.mean(all_fcsw, axis=0) if use_fc_skip else fc_skip_weights
        new_w_t = jnp.mean(all_wt, axis=0) if use_w_temporal else w_temporal

        return (new_weights, new_lp, avg_error, avg_active,
                new_fc_w, new_fc_sw, new_w_t,
                total_tp, total_fp, total_fn)

    return train_fn


class Trainer:
    def __init__(self, midi_dir, grid_size=16, grid_width=None, grid_height=None,
                 relaxation_steps=64,
                 fs=8.0, key=None, scan_results=None, index_path=None,
                 prefetch=False, prefetch_depth=3, activation="leaky_relu",
                 lr=0.005, alpha=0.9, beta=0.1,
                 curriculum_phases=None, curriculum_patience=10,
                 pos_weight=20.0, lambda_sparse=0.01,
                 connectivity="neighbor", lr_w=0.001,
                 state_momentum=0.9,
                 spike_boost=5.0, asl_gamma_neg=4.0, asl_margin=0.05,
                 lr_amplification=0.0, forward_init=True):
        # Support both old grid_size and new grid_width/grid_height
        self.grid_width = grid_width if grid_width is not None else grid_size
        self.grid_height = grid_height if grid_height is not None else grid_size
        self.relaxation_steps = relaxation_steps
        self.fs = fs
        self.activation_name = activation
        self._activation_fn = ACTIVATIONS.get(activation, jax.nn.leaky_relu)
        self.state_momentum = state_momentum
        self.connectivity = connectivity

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
        self.grid = create_grid(
            width=self.grid_width, height=self.grid_height,
            num_instruments=num_instruments, key=key,
            lr=lr, alpha=alpha, beta=beta,
            pos_weight=pos_weight, lambda_sparse=lambda_sparse,
            connectivity=connectivity, lr_w=lr_w,
            spike_boost=spike_boost,
            asl_gamma_neg=asl_gamma_neg, asl_margin=asl_margin,
            lr_amplification=lr_amplification,
        )

        self._train_fn = _make_train_fn(
            relaxation_steps, self._activation_fn,
            pos_weight=pos_weight, lambda_sparse=lambda_sparse,
            lr_w=lr_w, spike_boost=spike_boost,
            asl_gamma_neg=asl_gamma_neg, asl_margin=asl_margin,
            use_fc=(connectivity in ("fc", "fc_double")),
            use_fc_skip=(connectivity == "fc_double"),
            use_w_temporal=True,
            use_forward_init=forward_init,
        )

    def _get_fc_weights(self):
        """Return FC weights or a dummy array for JIT compatibility."""
        if self.grid.fc_weights is not None:
            return self.grid.fc_weights
        return jnp.zeros((self.grid_width, self.grid_height, self.grid_height))

    def _get_fc_skip_weights(self):
        if self.grid.fc_skip_weights is not None:
            return self.grid.fc_skip_weights
        return jnp.zeros((self.grid_width, self.grid_height, self.grid_height))

    def _get_w_temporal(self):
        if self.grid.w_temporal is not None:
            return self.grid.w_temporal
        return jnp.full((self.grid_height, self.grid_width), 0.1)

    def _build_all_conds(self, conditioning):
        """Vectorized: expand conditioning vectors for the whole batch."""
        batch_size, num_cat = conditioning.shape
        cond_start = (self.grid_height - num_cat) // 2
        all_conds = jnp.zeros((batch_size, self.grid_height))
        all_conds = all_conds.at[:, cond_start:cond_start + num_cat].set(
            jnp.array(conditioning)
        )
        return all_conds

    def train_step(self, batch_size=32):
        """Run one training step: generate batch, process on GPU.

        Returns:
            avg_error: float
            batch_meta: dict with 'datasets', 'active_error', 'col_energy'
        """
        self._batch_source.snippet_ticks = self.curriculum.snippet_ticks
        batch = self._batch_source.generate_batch(batch_size)

        all_inputs = jnp.array(batch["input"][:, :, :self.grid_height])
        all_targets = jnp.array(batch["target"][:, :, :self.grid_height])
        all_conds = self._build_all_conds(batch["conditioning"])

        (new_weights, new_lp, avg_error, avg_active,
         new_fc_w, new_fc_sw, new_w_t,
         total_tp, total_fp, total_fn) = self._train_fn(
            self.grid.state, self.grid.weights, self.grid.params,
            self.grid.log_precision,
            all_inputs, all_targets, all_conds,
            self.grid.input_mask, self.grid.output_mask,
            self.grid.conditioning_mask,
            self._get_fc_weights(), self._get_fc_skip_weights(),
            self._get_w_temporal(),
        )

        self.grid.weights = new_weights
        self.grid.log_precision = new_lp

        # Update FC weights if applicable
        if self.grid.fc_weights is not None:
            self.grid.fc_weights = new_fc_w
        if self.grid.fc_skip_weights is not None:
            self.grid.fc_skip_weights = new_fc_sw
        if self.grid.w_temporal is not None:
            self.grid.w_temporal = new_w_t

        avg_error = float(avg_error)
        avg_active = float(avg_active)
        self.curriculum.report_error(avg_error)

        # Note detection F1 at output boundary
        tp = float(total_tp)
        fp = float(total_fp)
        fn = float(total_fn)
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        # Per-column energy (sampled)
        col_energy_sample = {}
        W = self.grid_width
        if W > 0:
            e_ch = self.grid.state[:, :, 1]
            ce = jnp.mean(jnp.abs(e_ch), axis=0)
            col_energy_sample = {
                0: float(ce[0]),
                W // 2: float(ce[W // 2]),
                W - 1: float(ce[W - 1]),
            }

        return avg_error, {
            "datasets": batch.get("datasets", []),
            "active_error": avg_active,
            "col_energy": col_energy_sample,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def evaluate_error(self, batch_size=4):
        """Evaluate current error without updating weights."""
        self._batch_source.snippet_ticks = self.curriculum.snippet_ticks
        batch = self._batch_source.generate_batch(batch_size)

        all_inputs = jnp.array(batch["input"][:, :, :self.grid_height])
        all_targets = jnp.array(batch["target"][:, :, :self.grid_height])
        all_conds = self._build_all_conds(batch["conditioning"])

        result = self._train_fn(
            self.grid.state, self.grid.weights, self.grid.params,
            self.grid.log_precision,
            all_inputs, all_targets, all_conds,
            self.grid.input_mask, self.grid.output_mask,
            self.grid.conditioning_mask,
            self._get_fc_weights(), self._get_fc_skip_weights(),
            self._get_w_temporal(),
        )
        return float(result[2])  # avg_error

    def save_checkpoint(self, path):
        """Save grid state, weights, params, log_precision, and metadata."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "state.npy", np.array(self.grid.state))
        np.save(path / "weights.npy", np.array(self.grid.weights))
        np.save(path / "params.npy", np.array(self.grid.params))
        np.save(path / "log_precision.npy", np.array(self.grid.log_precision))
        if self.grid.w_temporal is not None:
            np.save(path / "w_temporal.npy", np.array(self.grid.w_temporal))
        if self.grid.fc_weights is not None:
            np.save(path / "fc_weights.npy", np.array(self.grid.fc_weights))
        if self.grid.fc_skip_weights is not None:
            np.save(path / "fc_skip_weights.npy", np.array(self.grid.fc_skip_weights))
        # Save vocabulary + config metadata for export
        metadata = {
            "vocabulary": self.batch_gen.vocabulary,
            "grid_width": self.grid_width,
            "grid_height": self.grid_height,
            "activation": self.activation_name,
            "connectivity": self.connectivity,
        }
        (path / "metadata.json").write_text(json.dumps(metadata))

    def load_checkpoint(self, path):
        """Load grid state, weights, params, and optionally log_precision."""
        path = Path(path)
        self.grid.state = jnp.array(np.load(path / "state.npy"))
        self.grid.weights = jnp.array(np.load(path / "weights.npy"))
        self.grid.params = jnp.array(np.load(path / "params.npy"))
        lp_path = path / "log_precision.npy"
        if lp_path.exists():
            self.grid.log_precision = jnp.array(np.load(lp_path))
        wt_path = path / "w_temporal.npy"
        if wt_path.exists():
            self.grid.w_temporal = jnp.array(np.load(wt_path))
        fc_path = path / "fc_weights.npy"
        if fc_path.exists():
            self.grid.fc_weights = jnp.array(np.load(fc_path))
        fc_skip_path = path / "fc_skip_weights.npy"
        if fc_skip_path.exists():
            self.grid.fc_skip_weights = jnp.array(np.load(fc_skip_path))
