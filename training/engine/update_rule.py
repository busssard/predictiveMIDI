import jax
import jax.numpy as jnp


def pc_relaxation_step(state, weights, params):
    """One relaxation step of the PC grid. All neurons update simultaneously.

    Args:
        state: (H, W, 4) -- channels: r, e, s, h
        weights: (H, W, 4) -- channels: w_left, w_right, w_up, w_down
        params: (H, W, 4) -- channels: alpha, beta, lr, bias

    Returns:
        new_state: (H, W, 4)
        new_weights: (H, W, 4) -- updated via iPC
    """
    r = state[:, :, 0]
    e = state[:, :, 1]
    s = state[:, :, 2]
    h = state[:, :, 3]

    w_left = weights[:, :, 0]
    w_right = weights[:, :, 1]
    w_up = weights[:, :, 2]
    w_down = weights[:, :, 3]

    alpha = params[:, :, 0]
    beta = params[:, :, 1]
    lr = params[:, :, 2]
    bias = params[:, :, 3]

    # Activation function
    r_act = jnp.tanh(r)

    # Gather neighbor representations using zero-padding at boundaries
    # r_left: the neighbor to the left of each neuron (i.e., value at col-1)
    r_left = jnp.zeros_like(r_act).at[:, 1:].set(r_act[:, :-1])
    # r_right: the neighbor to the right (value at col+1)
    r_right = jnp.zeros_like(r_act).at[:, :-1].set(r_act[:, 1:])
    # r_up: the neighbor above (value at row-1)
    r_up = jnp.zeros_like(r_act).at[1:, :].set(r_act[:-1, :])
    # r_down: the neighbor below (value at row+1)
    r_down = jnp.zeros_like(r_act).at[:-1, :].set(r_act[1:, :])

    # Predictions arriving at this neuron from neighbors
    pred_from_left = w_left * r_left
    pred_from_right = w_right * r_right
    pred_from_up = w_up * r_up
    pred_from_down = w_down * r_down
    total_prediction = pred_from_left + pred_from_right + pred_from_up + pred_from_down

    # Error: difference between representation and incoming predictions
    new_e = r - total_prediction - bias

    # Gather neighbor errors for the representation update
    e_left = jnp.zeros_like(new_e).at[:, 1:].set(new_e[:, :-1])
    e_right = jnp.zeros_like(new_e).at[:, :-1].set(new_e[:, 1:])
    e_up = jnp.zeros_like(new_e).at[1:, :].set(new_e[:-1, :])
    e_down = jnp.zeros_like(new_e).at[:-1, :].set(new_e[1:, :])

    neighbor_error_signal = (
        e_left * w_left + e_right * w_right +
        e_up * w_up + e_down * w_down
    )

    # Update temporal state
    new_s = alpha * s + (1.0 - alpha) * r
    new_h = beta * jnp.tanh(h)

    # Representation update
    new_r = r + lr * (-new_e + neighbor_error_signal + new_h + new_s - r)

    # iPC weight updates
    lr_w = lr * 0.1  # weight learning rate is fraction of main lr
    new_w_left = w_left + lr_w * new_e * r_left
    new_w_right = w_right + lr_w * new_e * r_right
    new_w_up = w_up + lr_w * new_e * r_up
    new_w_down = w_down + lr_w * new_e * r_down

    new_state = jnp.stack([new_r, new_e, new_s, new_h], axis=-1)
    new_weights = jnp.stack([new_w_left, new_w_right, new_w_up, new_w_down], axis=-1)

    return new_state, new_weights


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
