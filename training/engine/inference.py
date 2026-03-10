"""Inference engine: run a trained PC grid without output clamping.

Given a checkpoint and input MIDI, clamps only the input column and
lets the output column evolve freely through relaxation dynamics.
"""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.lax as lax
import numpy as np

from training.engine.grid import GridState, create_grid
from training.engine.update_rule import pc_relaxation_step, apply_clamping, ACTIVATIONS


def load_checkpoint(checkpoint_path):
    """Load a checkpoint into a GridState + metadata."""
    path = Path(checkpoint_path)
    metadata = json.loads((path / "metadata.json").read_text())

    width = metadata["grid_width"]
    height = metadata["grid_height"]
    activation = metadata.get("activation", "leaky_relu")
    connectivity = metadata.get("connectivity", "neighbor")
    vocabulary = metadata.get("vocabulary", {})

    state = jnp.array(np.load(path / "state.npy"))
    weights = jnp.array(np.load(path / "weights.npy"))
    params = jnp.array(np.load(path / "params.npy"))

    log_precision = jnp.zeros((height, width))
    lp_path = path / "log_precision.npy"
    if lp_path.exists():
        log_precision = jnp.array(np.load(lp_path))

    w_temporal = jnp.full((height, width), 0.1)
    wt_path = path / "w_temporal.npy"
    if wt_path.exists():
        w_temporal = jnp.array(np.load(wt_path))

    fc_weights = None
    fc_path = path / "fc_weights.npy"
    if fc_path.exists():
        fc_weights = jnp.array(np.load(fc_path))

    fc_skip_weights = None
    fcs_path = path / "fc_skip_weights.npy"
    if fcs_path.exists():
        fc_skip_weights = jnp.array(np.load(fcs_path))

    # Build masks
    input_mask = jnp.zeros((height, width), dtype=bool).at[:, 0].set(True)
    output_mask = jnp.zeros((height, width), dtype=bool).at[:, -1].set(True)

    num_instruments = len(vocabulary)
    cond_start = (height - num_instruments) // 2
    conditioning_mask = jnp.zeros((height, width), dtype=bool)
    conditioning_mask = conditioning_mask.at[
        cond_start:cond_start + num_instruments, 0
    ].set(True)

    grid = GridState(
        state=state,
        weights=weights,
        params=params,
        log_precision=log_precision,
        input_mask=input_mask,
        output_mask=output_mask,
        conditioning_mask=conditioning_mask,
        connectivity=connectivity,
        fc_weights=fc_weights,
        fc_skip_weights=fc_skip_weights,
        w_temporal=w_temporal,
    )

    return grid, metadata


def _make_inference_fn(relaxation_steps, activation_fn, use_fc=False,
                       use_fc_skip=False, use_w_temporal=False,
                       lambda_sparse=0.01, spike_boost=0.0):
    """Build a JIT-compiled inference function (no output clamping).

    Key differences from training:
    - Output column is NOT clamped — it evolves freely
    - Learned precision is RESET at output column to prevent divergence
      (during training, output clamping kept things stable; without it,
       high precision amplifies unbounded errors → NaN)
    - No weight updates (lr_w=0), no precision learning (lr_precision=0)
    - No FISTA momentum (r_prev=None equivalent via step_index=0)
    - Representation values are clipped to [-5, 5] for stability
    """

    @jax.jit
    def infer_fn(state, weights, params, log_precision,
                 input_seq, cond_vals,
                 input_mask, conditioning_mask,
                 fc_weights, fc_skip_weights, w_temporal):
        """
        Args:
            state: (H, W, 4)
            weights: (H, W, 4)
            params: (H, W, 4)
            log_precision: (H, W)
            input_seq: (T, H) -- input piano roll sequence
            cond_vals: (H,) -- conditioning vector (zero-padded, full height)
            input_mask: (H, W) bool
            conditioning_mask: (H, W) bool
            fc_weights, fc_skip_weights, w_temporal: as in training

        Returns:
            output_seq: (T, H) -- predicted output at final column
            final_state: (H, W, 4) -- grid state after last tick
            per_tick_states: (T, H, W, 4) -- full grid state at each tick
        """
        _fc_w = fc_weights if use_fc else None
        _fc_sw = fc_skip_weights if use_fc_skip else None
        _w_t = w_temporal if use_w_temporal else None

        H, W = state.shape[0], state.shape[1]

        # Reset learned precision: use flat precision=1.0 everywhere
        # The trained log_precision was learned with output clamping;
        # using it during free inference causes divergence at boundaries.
        inference_log_precision = jnp.zeros((H, W))

        # Reduce representation learning rate for stability
        # (keep alpha/beta/bias from params, but lower lr)
        inference_params = params.at[:, :, 2].set(params[:, :, 2] * 0.5)

        def process_tick(carry, inp):
            st, wt, lp, fcw, fcsw, wt_temp = carry

            # External precision: only boost input boundary
            precision = jnp.ones((H, W))
            precision = precision.at[:, 0].set(
                jnp.where(inp != 0.0, 10.0, 1.0))

            step_indices = jnp.arange(relaxation_steps)

            def relax_step(carry, step_idx):
                s, w, log_p, fc_w, fc_sw, w_t = carry
                (s, w, new_log_p,
                 new_fc_w, new_fc_sw, new_w_t) = pc_relaxation_step(
                    s, w, inference_params, activation_fn,
                    precision=precision, lambda_sparse=lambda_sparse,
                    log_precision=log_p, lr_precision=0.0,
                    fc_weights=fc_w, fc_skip_weights=fc_sw,
                    lr_w=0.0,
                    step_index=0,  # constant 0 disables FISTA momentum
                    r_prev=None,   # no FISTA
                    spike_boost=0.0,  # no spike boost during inference
                    w_temporal=w_t,
                )
                # Only clamp input, NOT output
                s = apply_clamping(s, input_mask, inp, channel=0)
                s = apply_clamping(s, conditioning_mask, cond_vals, channel=0)

                # Clip representations to prevent divergence
                r_clipped = jnp.clip(s[:, :, 0], -5.0, 5.0)
                s = s.at[:, :, 0].set(r_clipped)

                out_fc_w = new_fc_w if use_fc else fc_w
                out_fc_sw = new_fc_sw if use_fc_skip else fc_sw
                out_w_t = new_w_t if use_w_temporal else w_t

                return (s, w, new_log_p,
                        out_fc_w, out_fc_sw, out_w_t), None

            (new_st, new_wt, new_lp, new_fcw, new_fcsw, new_wt_temp), _ = lax.scan(
                relax_step,
                (st, wt, lp, fcw, fcsw, wt_temp),
                step_indices,
            )

            # Read output from final column
            output = jax.nn.sigmoid(new_st[:, -1, 0])  # (H,)

            return (new_st, new_wt, new_lp, new_fcw, new_fcsw, new_wt_temp), (output, new_st)

        init_carry = (state, weights, inference_log_precision,
                      _fc_w, _fc_sw, _w_t)
        (final_state, final_wt, final_lp, _, _, _), (output_seq, all_states) = lax.scan(
            process_tick, init_carry, input_seq)

        return output_seq, final_state, all_states

    return infer_fn


def run_inference(grid, metadata, input_piano_roll, conditioning_vec,
                  relaxation_steps=64):
    """High-level inference: run grid on input, return output piano roll.

    Args:
        grid: GridState (loaded from checkpoint)
        metadata: dict with vocabulary, activation, connectivity, etc.
        input_piano_roll: (T, 128) float32 array — input instrument(s)
        conditioning_vec: (num_categories,) float32 — one-hot target instrument
        relaxation_steps: number of relaxation steps per tick

    Returns:
        output_roll: (T, 128) float32 — predicted output (sigmoid probabilities)
        all_states: (T, H, W, 4) — full grid state at each tick
    """
    activation_name = metadata.get("activation", "leaky_relu")
    activation_fn = ACTIVATIONS.get(activation_name, jax.nn.leaky_relu)
    connectivity = metadata.get("connectivity", "neighbor")

    H, W = grid.state.shape[0], grid.state.shape[1]

    # Pad input to grid height if needed
    T = input_piano_roll.shape[0]
    inp = np.zeros((T, H), dtype=np.float32)
    inp[:, :min(128, H)] = input_piano_roll[:, :min(128, H)]
    input_seq = jnp.array(inp)

    # Build full conditioning vector (zero-padded to H)
    num_cat = len(conditioning_vec)
    cond_start = (H - num_cat) // 2
    cond_full = np.zeros(H, dtype=np.float32)
    cond_full[cond_start:cond_start + num_cat] = conditioning_vec
    cond_vals = jnp.array(cond_full)

    # FC weight dummies
    fc_w = grid.fc_weights if grid.fc_weights is not None else jnp.zeros((W, H, H))
    fc_sw = grid.fc_skip_weights if grid.fc_skip_weights is not None else jnp.zeros((W, H, H))
    w_t = grid.w_temporal if grid.w_temporal is not None else jnp.full((H, W), 0.1)

    infer_fn = _make_inference_fn(
        relaxation_steps, activation_fn,
        use_fc=(connectivity in ("fc", "fc_double")),
        use_fc_skip=(connectivity == "fc_double"),
        use_w_temporal=(grid.w_temporal is not None),
        lambda_sparse=grid.lambda_sparse,
        spike_boost=grid.spike_boost,
    )

    output_seq, final_state, all_states = infer_fn(
        grid.state, grid.weights, grid.params, grid.log_precision,
        input_seq, cond_vals,
        grid.input_mask, grid.conditioning_mask,
        fc_w, fc_sw, w_t,
    )

    return np.array(output_seq), np.array(all_states)
