"""Hierarchical PC update rule.

Predictions flow from higher layers to lower layers (top-down).
Errors flow from lower layers to higher layers (bottom-up).

Per relaxation step, for each adjacent layer pair (l, l+1):
    pred_l = tanh(W_pred[l] @ x[l+1])           # top-down prediction
    e_l = x[l] - pred_l - skip_pred_l            # prediction error
    x[l+1] += lr * W_pred[l].T @ e_l             # error drives higher layer
    W_pred[l] += lr_w * outer(e_l, x[l+1])       # Hebbian weight update
"""
import jax
import jax.numpy as jnp
from typing import Dict, List, Optional, Tuple


def hierarchical_relaxation_step(
    representations: List[jnp.ndarray],
    prediction_weights: List[jnp.ndarray],
    skip_weights: List[jnp.ndarray],
    temporal_weights: List[jnp.ndarray],
    temporal_state: List[jnp.ndarray],
    layer_sizes: List[int],
    lr: float = 0.01,
    lr_w: float = 0.001,
    lambda_sparse: float = 0.005,
    clamp_mask: Optional[Dict[int, jnp.ndarray]] = None,
) -> Tuple[List[jnp.ndarray], List[jnp.ndarray],
           List[jnp.ndarray], List[jnp.ndarray]]:
    """One relaxation step of the hierarchical PC network.

    Args:
        representations: list of (H_l,) per layer
        prediction_weights: list of (H_low, H_high) — W_pred[i] predicts layer i from i+1
        skip_weights: list of (H_dec, H_enc) — skip from encoder i to decoder N-1-i
        temporal_weights: list of (H_l, H_l) — temporal prediction per layer
        temporal_state: list of (H_l,) — previous tick representations
        layer_sizes: list of ints
        lr: representation learning rate
        lr_w: weight learning rate
        lambda_sparse: L1 sparsity penalty
        clamp_mask: dict {layer_idx: clamp_values} — layers to hold fixed

    Returns:
        new_representations, new_errors, new_prediction_weights, new_skip_weights
    """
    if clamp_mask is None:
        clamp_mask = {}

    num_layers = len(layer_sizes)
    activation_fn = jnp.tanh

    # Compute predictions for each layer
    predictions = [jnp.zeros(s) for s in layer_sizes]

    # Top-down predictions: layer i+1 predicts layer i
    for i in range(num_layers - 1):
        x_high = activation_fn(representations[i + 1])
        pred = prediction_weights[i] @ x_high
        predictions[i] = predictions[i] + pred

    # Skip connection predictions: encoder i predicts decoder (N-1-i)
    n_skip = len(skip_weights)
    for si in range(n_skip):
        enc_idx = si
        dec_idx = num_layers - 1 - si
        if enc_idx != dec_idx:
            x_enc = activation_fn(representations[enc_idx])
            skip_pred = skip_weights[si] @ x_enc
            predictions[dec_idx] = predictions[dec_idx] + skip_pred

    # Temporal predictions (mild influence)
    for i in range(num_layers):
        temp_pred = temporal_weights[i] @ activation_fn(temporal_state[i])
        predictions[i] = predictions[i] + temp_pred * 0.1

    # Compute errors
    errors = []
    for i in range(num_layers):
        errors.append(representations[i] - predictions[i])

    # Update representations
    new_reps = list(representations)
    for i in range(num_layers):
        if i in clamp_mask:
            new_reps[i] = clamp_mask[i]
            continue

        # Self error minimization
        update = -errors[i]

        # Bottom-up error propagation: this layer predicted layer i-1,
        # so gradients from e[i-1] flow back through W_pred[i-1].T
        # Include tanh derivative for correct gradient
        if i > 0:
            tanh_deriv = 1.0 - jnp.tanh(representations[i]) ** 2
            update = update + tanh_deriv * (prediction_weights[i - 1].T @ errors[i - 1])

        # Sparsity penalty
        update = update - lambda_sparse * jnp.sign(representations[i])

        # Mild L2 damping for stability (clipping handles explosion)
        update = update - 0.01 * representations[i]

        new_reps[i] = representations[i] + lr * update
        new_reps[i] = jnp.clip(new_reps[i], -5.0, 5.0)

    # Update prediction weights (Hebbian: dW = e_low @ x_high.T)
    new_pred_w = list(prediction_weights)
    for i in range(num_layers - 1):
        x_high = activation_fn(representations[i + 1])
        dw = jnp.outer(errors[i], x_high)
        new_pred_w[i] = prediction_weights[i] + lr_w * dw

        # Spectral normalization with relaxed threshold
        frob = jnp.sqrt(jnp.sum(new_pred_w[i] ** 2))
        min_dim = float(min(new_pred_w[i].shape))
        max_sv_approx = frob / jnp.sqrt(min_dim)
        new_pred_w[i] = jnp.where(
            max_sv_approx > 2.0,
            new_pred_w[i] * (2.0 / max_sv_approx),
            new_pred_w[i],
        )

    # Update skip weights (same lr as prediction — skip connections are critical)
    new_skip_w = list(skip_weights)
    for si in range(n_skip):
        enc_idx = si
        dec_idx = num_layers - 1 - si
        if enc_idx != dec_idx:
            x_enc = activation_fn(representations[enc_idx])
            e_dec = errors[dec_idx]
            dw = jnp.outer(e_dec, x_enc)
            new_skip_w[si] = skip_weights[si] + lr_w * dw

            frob = jnp.sqrt(jnp.sum(new_skip_w[si] ** 2))
            min_dim = float(min(new_skip_w[si].shape))
            max_sv_approx = frob / jnp.sqrt(min_dim)
            new_skip_w[si] = jnp.where(
                max_sv_approx > 2.0,
                new_skip_w[si] * (2.0 / max_sv_approx),
                new_skip_w[si],
            )

    # Update temporal weights (Hebbian at relaxation level)
    new_temp_w = list(temporal_weights)
    for i in range(num_layers):
        temp_pred = temporal_weights[i] @ activation_fn(temporal_state[i])
        temp_error = representations[i] - temp_pred
        dw = jnp.outer(temp_error, activation_fn(temporal_state[i]))
        new_temp_w[i] = temporal_weights[i] + lr_w * 0.1 * dw

        # Spectral norm for temporal weights
        frob = jnp.sqrt(jnp.sum(new_temp_w[i] ** 2))
        dim = float(new_temp_w[i].shape[0])
        max_sv_approx = frob / jnp.sqrt(dim)
        new_temp_w[i] = jnp.where(
            max_sv_approx > 2.0,
            new_temp_w[i] * (2.0 / max_sv_approx),
            new_temp_w[i],
        )

    return new_reps, errors, new_pred_w, new_skip_w, new_temp_w
