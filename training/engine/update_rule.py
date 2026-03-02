import jax
import jax.numpy as jnp


ACTIVATIONS = {
    "tanh": jnp.tanh,
    "sigmoid": jax.nn.sigmoid,
    "relu": jax.nn.relu,
    "leaky_relu": jax.nn.leaky_relu,
    "linear": lambda x: x,
}


def pc_relaxation_step(state, weights, params, activation_fn=None,
                       precision=None, lambda_sparse=0.0,
                       log_precision=None, lr_precision=0.001,
                       fc_weights=None, fc_skip_weights=None,
                       lr_w=None, step_index=0, r_prev=None,
                       spike_boost=0.0, w_temporal=None):
    """One relaxation step of the PC grid. All neurons update simultaneously.

    Args:
        state: (H, W, 4) -- channels: r, e, s, h
        weights: (H, W, 4) -- channels: w_left, w_right, w_up, w_down
        params: (H, W, 4) -- channels: alpha, beta, lr, bias
        activation_fn: callable, defaults to jnp.tanh
        precision: (H, W) optional external precision mask (e.g. pos_weight)
        lambda_sparse: L1 sparsity penalty coefficient
        log_precision: (H, W) optional learned log-precision per neuron
        lr_precision: learning rate for log-precision updates
        fc_weights: (W, H, H) optional FC inter-column weights
        fc_skip_weights: (W, H, H) optional FC skip-2 weights
        lr_w: explicit weight learning rate (if None, uses lr * 0.1)
        step_index: relaxation step index for FISTA momentum
        r_prev: (H, W) previous representation for FISTA momentum
        spike_boost: column wavefront spiking precision boost
        w_temporal: (H, W) learnable temporal gain per neuron

    Returns:
        new_state: (H, W, 4)
        new_weights: (H, W, 4) -- updated via iPC
        new_log_precision: (H, W) or None -- updated log-precision
        new_fc_weights: same shape as fc_weights or None
        new_fc_skip_weights: same shape as fc_skip_weights or None
        new_w_temporal: (H, W) or None
    """
    if activation_fn is None:
        activation_fn = jnp.tanh
    r = state[:, :, 0]
    e = state[:, :, 1]
    s = state[:, :, 2]
    h = state[:, :, 3]
    H, W = r.shape

    w_left = weights[:, :, 0]
    w_right = weights[:, :, 1]
    w_up = weights[:, :, 2]
    w_down = weights[:, :, 3]

    alpha = params[:, :, 0]
    beta = params[:, :, 1]
    lr = params[:, :, 2]
    bias = params[:, :, 3]

    # Activation function
    r_act = activation_fn(r)

    # Gather neighbor representations using zero-padding at boundaries
    r_left = jnp.zeros_like(r_act).at[:, 1:].set(r_act[:, :-1])
    r_right = jnp.zeros_like(r_act).at[:, :-1].set(r_act[:, 1:])
    r_up = jnp.zeros_like(r_act).at[1:, :].set(r_act[:-1, :])
    r_down = jnp.zeros_like(r_act).at[:-1, :].set(r_act[1:, :])

    # Predictions arriving at this neuron from neighbors
    pred_from_left = w_left * r_left
    pred_from_right = w_right * r_right
    pred_from_up = w_up * r_up
    pred_from_down = w_down * r_down
    total_prediction = pred_from_left + pred_from_right + pred_from_up + pred_from_down

    # FC inter-column predictions
    new_fc_weights_out = None
    new_fc_skip_weights_out = None
    if fc_weights is not None:
        # r_act transposed to (W, H) for batched matmul
        r_cols = r_act.T  # (W, H)
        # Shift to get left/right neighbor columns
        r_left_cols = jnp.zeros_like(r_cols).at[1:].set(r_cols[:-1])
        r_right_cols = jnp.zeros_like(r_cols).at[:-1].set(r_cols[1:])
        # FC predictions: fc_weights[j] @ r_left_cols[j] for each column j
        fc_pred_left = jnp.einsum('wij,wj->wi', fc_weights, r_left_cols).T  # (H, W)
        fc_pred_right = jnp.einsum('wji,wj->wi', fc_weights, r_right_cols).T
        total_prediction = total_prediction + fc_pred_left + fc_pred_right

    if fc_skip_weights is not None:
        r_cols = r_act.T  # (W, H)
        r_left2_cols = jnp.zeros_like(r_cols).at[2:].set(r_cols[:-2])
        r_right2_cols = jnp.zeros_like(r_cols).at[:-2].set(r_cols[2:])
        fc_skip_pred_left = jnp.einsum('wij,wj->wi', fc_skip_weights, r_left2_cols).T
        fc_skip_pred_right = jnp.einsum('wji,wj->wi', fc_skip_weights, r_right2_cols).T
        total_prediction = total_prediction + fc_skip_pred_left + fc_skip_pred_right

    # Error: difference between representation and incoming predictions
    new_e = r - total_prediction - bias

    # Learned precision update (Ofner 2021 eq. 8)
    if log_precision is not None:
        pi = jnp.exp(log_precision)
        new_log_precision = log_precision + lr_precision * (1.0 - pi * new_e ** 2)
        new_log_precision = jnp.clip(new_log_precision, -4.0, 4.0)
        effective_precision = pi
    else:
        new_log_precision = None
        effective_precision = jnp.ones_like(r)

    # Apply external precision mask (pos_weight at boundaries)
    if precision is not None:
        effective_precision = effective_precision * precision

    # Spiking precision (column wavefront)
    if spike_boost > 0.0:
        col_idx = jnp.arange(W)[None, :]  # (1, W)
        left_spike = (col_idx == step_index).astype(jnp.float32)
        right_spike = (col_idx == (W - 1 - step_index)).astype(jnp.float32)
        effective_precision = effective_precision * (1.0 + spike_boost * (left_spike + right_spike))

    # Precision-weighted error
    weighted_e = effective_precision * new_e

    # Gather neighbor errors (use weighted errors)
    e_left = jnp.zeros_like(weighted_e).at[:, 1:].set(weighted_e[:, :-1])
    e_right = jnp.zeros_like(weighted_e).at[:, :-1].set(weighted_e[:, 1:])
    e_up = jnp.zeros_like(weighted_e).at[1:, :].set(weighted_e[:-1, :])
    e_down = jnp.zeros_like(weighted_e).at[:-1, :].set(weighted_e[1:, :])

    neighbor_error_signal = (
        e_left * w_left + e_right * w_right +
        e_up * w_up + e_down * w_down
    )

    # Update temporal state
    new_s = alpha * s + (1.0 - alpha) * r

    # Temporal state with learnable gain (w_temporal updated per tick, not per relaxation step)
    new_w_temporal_out = w_temporal
    if w_temporal is not None:
        new_h = w_temporal * activation_fn(h)
    else:
        new_h = beta * activation_fn(h)

    # Representation update WITH sparsity penalty
    new_r = r + lr * (-weighted_e + neighbor_error_signal + new_h + new_s - r
                      - lambda_sparse * jnp.sign(r))

    # FISTA momentum
    if r_prev is not None:
        mom = jnp.where(step_index > 0, (step_index - 1.0) / (step_index + 2.0), 0.0)
        new_r = new_r + mom * (new_r - r_prev)

    # Weight learning rate
    if lr_w is None:
        w_lr = lr * 0.1
    else:
        w_lr = lr_w

    # iPC weight updates (use weighted error)
    new_w_left = w_left + w_lr * weighted_e * r_left
    new_w_right = w_right + w_lr * weighted_e * r_right
    new_w_up = w_up + w_lr * weighted_e * r_up
    new_w_down = w_down + w_lr * weighted_e * r_down

    new_weights = jnp.stack([new_w_left, new_w_right, new_w_up, new_w_down], axis=-1)

    # L2 weight normalization: normalize per neuron so ||w||_2 <= 1
    w_norm = jnp.sqrt(jnp.sum(new_weights ** 2, axis=-1, keepdims=True))
    new_weights = new_weights / jnp.maximum(w_norm, 1e-8)

    # FC weight updates
    if fc_weights is not None:
        weighted_e_t = weighted_e.T  # (W, H)
        r_cols = r_act.T
        r_left_cols = jnp.zeros_like(r_cols).at[1:].set(r_cols[:-1])
        new_fc = fc_weights + w_lr * jnp.einsum('wh,wk->whk', weighted_e_t, r_left_cols)
        # L2 normalize FC weights per-row
        fc_norm = jnp.sqrt(jnp.sum(new_fc ** 2, axis=-1, keepdims=True))
        new_fc = new_fc / jnp.maximum(fc_norm, 1e-8)
        new_fc_weights_out = new_fc

    if fc_skip_weights is not None:
        weighted_e_t = weighted_e.T
        r_cols = r_act.T
        r_left2_cols = jnp.zeros_like(r_cols).at[2:].set(r_cols[:-2])
        new_fc_skip = fc_skip_weights + w_lr * jnp.einsum('wh,wk->whk', weighted_e_t, r_left2_cols)
        fc_skip_norm = jnp.sqrt(jnp.sum(new_fc_skip ** 2, axis=-1, keepdims=True))
        new_fc_skip = new_fc_skip / jnp.maximum(fc_skip_norm, 1e-8)
        new_fc_skip_weights_out = new_fc_skip

    # new_e stored in state is RAW unweighted error (for monitoring)
    new_state = jnp.stack([new_r, new_e, new_s, new_h], axis=-1)

    return (new_state, new_weights, new_log_precision,
            new_fc_weights_out, new_fc_skip_weights_out, new_w_temporal_out)


def apply_clamping(state, mask, values, channel=0):
    """Overwrite the representation of clamped neurons.

    Args:
        state: (H, W, 4)
        mask: (H, W) bool -- which neurons to clamp
        values: (H,) -- values to clamp (one per row)
        channel: which state channel to clamp (0 = representation)

    Returns:
        new_state with clamped values applied.
    """
    # Broadcast values to match mask shape: values[i] applied to row i
    value_grid = jnp.broadcast_to(values[:, None], mask.shape)
    current = state[:, :, channel]
    clamped = jnp.where(mask, value_grid, current)
    return state.at[:, :, channel].set(clamped)
