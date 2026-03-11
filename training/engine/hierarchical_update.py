"""Hierarchical PC update rule with bounded predictions.

Predictions flow from higher layers to lower layers (top-down).
Errors flow from lower layers to higher layers (bottom-up).

All predictions are bounded via PRED_SCALE * tanh(...) to ensure stable
attractor dynamics during inference. Without bounding, predictions are
linear functions of weights and can grow without limit, preventing
convergence to meaningful fixed points.

Per relaxation step, for each adjacent layer pair (l, l+1):
    pred_l = PRED_SCALE * tanh(W_pred[l] @ tanh(x[l+1]) + b[l])  # bounded top-down prediction
    e_l = x[l] - pred_l - skip_pred_l                              # prediction error
    x[l+1] += lr * tanh'(x[l+1]) * W_pred[l].T @ (pred_deriv * e_l)  # error with outer deriv
    W_pred[l] += lr_w * outer(pred_deriv * e_l, tanh(x[l+1]))     # Hebbian weight update
"""
import jax
import jax.numpy as jnp
from typing import Dict, List, Optional, Tuple

# Prediction outputs are bounded to [-PRED_SCALE, PRED_SCALE].
# This ensures predictions cannot grow without limit, which is required
# for stable attractor dynamics during free-running inference.
PRED_SCALE = 3.0


def hierarchical_relaxation_step(
    representations: List[jnp.ndarray],
    prediction_weights: List[jnp.ndarray],
    prediction_biases: List[jnp.ndarray],
    skip_weights: List[jnp.ndarray],
    skip_biases: List[jnp.ndarray],
    temporal_weights: List[jnp.ndarray],
    temporal_state: List[jnp.ndarray],
    layer_sizes: List[int],
    lr: float = 0.01,
    lr_w: float = 0.001,
    lambda_sparse: float = 0.005,
    clamp_mask: Optional[Dict[int, jnp.ndarray]] = None,
    output_target: Optional[jnp.ndarray] = None,
    output_supervision: float = 0.0,
) -> Tuple[List[jnp.ndarray], List[jnp.ndarray],
           List[jnp.ndarray], List[jnp.ndarray],
           List[jnp.ndarray], List[jnp.ndarray],
           List[jnp.ndarray]]:
    """One relaxation step of the hierarchical PC network.

    Args:
        representations: list of (H_l,) per layer
        prediction_weights: list of (H_low, H_high) — W_pred[i] predicts layer i from i+1
        prediction_biases: list of (H_low,) — per-layer prediction bias
        skip_weights: list of (H_dec, H_enc) — skip from encoder i to decoder N-1-i
        skip_biases: list of (H_dec,) — per-skip-connection bias
        temporal_weights: list of (H_l, H_l) — temporal prediction per layer
        temporal_state: list of (H_l,) — previous tick representations
        layer_sizes: list of ints
        lr: representation learning rate
        lr_w: weight learning rate
        lambda_sparse: L1 sparsity penalty
        clamp_mask: dict {layer_idx: clamp_values} — layers to hold fixed
        output_target: optional (H_output,) logit-space target for output layer
        output_supervision: strength of output supervision (0=none, 1=strong)

    Returns:
        new_representations, new_errors, new_prediction_weights,
        new_prediction_biases, new_skip_weights, new_skip_biases,
        new_temporal_weights
    """
    if clamp_mask is None:
        clamp_mask = {}

    num_layers = len(layer_sizes)
    activation_fn = jnp.tanh

    # Compute predictions for each layer
    predictions = [jnp.zeros(s) for s in layer_sizes]

    # Top-down predictions: layer i+1 predicts layer i (bounded)
    for i in range(num_layers - 1):
        x_high = activation_fn(representations[i + 1])
        pred_pre = prediction_weights[i] @ x_high + prediction_biases[i]
        pred = PRED_SCALE * activation_fn(pred_pre)
        predictions[i] = predictions[i] + pred

    # Skip connection predictions: encoder i predicts decoder (N-1-i) (bounded)
    n_skip = len(skip_weights)
    for si in range(n_skip):
        enc_idx = si
        dec_idx = num_layers - 1 - si
        if enc_idx != dec_idx:
            x_enc = activation_fn(representations[enc_idx])
            skip_pre = skip_weights[si] @ x_enc + skip_biases[si]
            skip_pred = PRED_SCALE * activation_fn(skip_pre)
            predictions[dec_idx] = predictions[dec_idx] + skip_pred

    # Temporal predictions (mild influence, bounded)
    for i in range(num_layers):
        temp_pre = temporal_weights[i] @ activation_fn(temporal_state[i])
        temp_pred = PRED_SCALE * activation_fn(temp_pre) * 0.1
        predictions[i] = predictions[i] + temp_pred

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
        # pred_{i-1} = PRED_SCALE * tanh(W_{i-1} @ tanh(x[i]) + b)
        # d/dx[i] = PRED_SCALE * tanh'(pre) * W^T * tanh'(x[i])
        if i > 0:
            x_high_i = activation_fn(representations[i])
            pred_pre = prediction_weights[i - 1] @ x_high_i + prediction_biases[i - 1]
            pred_deriv = PRED_SCALE * (1.0 - jnp.tanh(pred_pre) ** 2)
            tanh_deriv = 1.0 - jnp.tanh(representations[i]) ** 2
            update = update + tanh_deriv * (prediction_weights[i - 1].T @ (pred_deriv * errors[i - 1]))

        # Output supervision: soft target signal at the output layer
        # Drives the output toward the target logits without hard clamping
        if i == num_layers - 1 and output_target is not None:
            supervision_error = representations[i] - output_target
            update = update - output_supervision * supervision_error

        # Sparsity penalty
        update = update - lambda_sparse * jnp.sign(representations[i])

        # Mild L2 damping for stability (clipping handles explosion)
        update = update - 0.01 * representations[i]

        new_reps[i] = representations[i] + lr * update
        new_reps[i] = jnp.clip(new_reps[i], -5.0, 5.0)

    # Update prediction weights (Hebbian with outer activation derivative)
    # pred = PRED_SCALE * tanh(W @ tanh(x) + b)
    # dW = (PRED_SCALE * tanh'(pre) * e) @ tanh(x).T
    new_pred_w = list(prediction_weights)
    new_pred_b = list(prediction_biases)
    for i in range(num_layers - 1):
        x_high = activation_fn(representations[i + 1])
        pred_pre = prediction_weights[i] @ x_high + prediction_biases[i]
        pred_deriv = PRED_SCALE * (1.0 - jnp.tanh(pred_pre) ** 2)
        dw = jnp.outer(pred_deriv * errors[i], x_high)
        new_pred_w[i] = prediction_weights[i] + lr_w * dw

        # Bias update: includes outer activation derivative
        new_pred_b[i] = prediction_biases[i] + lr_w * (pred_deriv * errors[i])

        # Spectral normalization with relaxed threshold
        frob = jnp.sqrt(jnp.sum(new_pred_w[i] ** 2))
        min_dim = float(min(new_pred_w[i].shape))
        max_sv_approx = frob / jnp.sqrt(min_dim)
        new_pred_w[i] = jnp.where(
            max_sv_approx > 2.0,
            new_pred_w[i] * (2.0 / max_sv_approx),
            new_pred_w[i],
        )

    # Update skip weights and biases (with outer activation derivative)
    # skip_pred = PRED_SCALE * tanh(W @ tanh(x_enc) + b)
    # dW = (PRED_SCALE * tanh'(pre) * e_dec) @ tanh(x_enc).T
    new_skip_w = list(skip_weights)
    new_skip_b = list(skip_biases)
    for si in range(n_skip):
        enc_idx = si
        dec_idx = num_layers - 1 - si
        if enc_idx != dec_idx:
            x_enc = activation_fn(representations[enc_idx])
            e_dec = errors[dec_idx]
            skip_pre = skip_weights[si] @ x_enc + skip_biases[si]
            skip_deriv = PRED_SCALE * (1.0 - jnp.tanh(skip_pre) ** 2)
            dw = jnp.outer(skip_deriv * e_dec, x_enc)
            new_skip_w[si] = skip_weights[si] + lr_w * dw

            # Skip bias update: includes outer activation derivative
            new_skip_b[si] = skip_biases[si] + lr_w * (skip_deriv * e_dec)

            frob = jnp.sqrt(jnp.sum(new_skip_w[si] ** 2))
            min_dim = float(min(new_skip_w[si].shape))
            max_sv_approx = frob / jnp.sqrt(min_dim)
            new_skip_w[si] = jnp.where(
                max_sv_approx > 2.0,
                new_skip_w[si] * (2.0 / max_sv_approx),
                new_skip_w[si],
            )

    # Update temporal weights (Hebbian with outer activation derivative)
    # temp_pred = PRED_SCALE * tanh(W @ tanh(temp_state)) * 0.1
    # dW = (PRED_SCALE * tanh'(pre) * 0.1 * temp_error) @ tanh(temp_state).T
    new_temp_w = list(temporal_weights)
    for i in range(num_layers):
        temp_act = activation_fn(temporal_state[i])
        temp_pre = temporal_weights[i] @ temp_act
        temp_deriv = PRED_SCALE * (1.0 - jnp.tanh(temp_pre) ** 2)
        temp_pred = PRED_SCALE * activation_fn(temp_pre) * 0.1
        temp_error = representations[i] - temp_pred
        dw = jnp.outer(temp_deriv * 0.1 * temp_error, temp_act)
        new_temp_w[i] = temporal_weights[i] + lr_w * dw

        # Spectral norm for temporal weights
        frob = jnp.sqrt(jnp.sum(new_temp_w[i] ** 2))
        dim = float(new_temp_w[i].shape[0])
        max_sv_approx = frob / jnp.sqrt(dim)
        new_temp_w[i] = jnp.where(
            max_sv_approx > 2.0,
            new_temp_w[i] * (2.0 / max_sv_approx),
            new_temp_w[i],
        )

    return new_reps, errors, new_pred_w, new_pred_b, new_skip_w, new_skip_b, new_temp_w
