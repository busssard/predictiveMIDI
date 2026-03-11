"""Autodiff weight updates for hierarchical PC network.

Alternative to the Hebbian outer-product updates in hierarchical_update.py.
Uses jax.grad to compute exact gradients of the free energy F = sum(||e_i||^2)
w.r.t. all weights, which Millidge et al. 2021 showed makes PC approximate
backprop.

Same function signature as hierarchical_relaxation_step() — takes the same
args, returns the same 7 values (reps, errors, pred_w, pred_b, skip_w,
skip_b, temp_w).

Representation updates are identical to the Hebbian version (gradient descent
on F w.r.t. representations). Only weight updates differ: instead of Hebbian
outer products, we use jax.value_and_grad on the free energy.
"""
import jax
import jax.numpy as jnp
from typing import Dict, List, Optional, Tuple

# Must match the Hebbian version exactly.
PRED_SCALE = 3.0


def _compute_free_energy(
    pred_w_flat: jnp.ndarray,
    pred_b_flat: jnp.ndarray,
    skip_w_flat: jnp.ndarray,
    skip_b_flat: jnp.ndarray,
    temp_w_flat: jnp.ndarray,
    representations: List[jnp.ndarray],
    temporal_state: List[jnp.ndarray],
    layer_sizes: List[int],
    pred_w_shapes: List[Tuple[int, int]],
    pred_b_shapes: List[Tuple[int, ...]],
    skip_w_shapes: List[Tuple[int, int]],
    skip_b_shapes: List[Tuple[int, ...]],
    temp_w_shapes: List[Tuple[int, int]],
) -> jnp.ndarray:
    """Compute free energy F = sum(||e_i||^2) as a function of weights.

    All weight arrays are passed as flat 1D arrays for differentiation,
    then reshaped internally.
    """
    num_layers = len(layer_sizes)
    activation_fn = jnp.tanh

    # Unflatten weights
    pred_w = []
    offset = 0
    for shape in pred_w_shapes:
        size = shape[0] * shape[1]
        pred_w.append(pred_w_flat[offset:offset + size].reshape(shape))
        offset += size

    pred_b = []
    offset = 0
    for shape in pred_b_shapes:
        size = shape[0]
        pred_b.append(pred_b_flat[offset:offset + size])
        offset += size

    skip_w = []
    offset = 0
    for shape in skip_w_shapes:
        size = shape[0] * shape[1]
        skip_w.append(skip_w_flat[offset:offset + size].reshape(shape))
        offset += size

    skip_b = []
    offset = 0
    for shape in skip_b_shapes:
        size = shape[0]
        skip_b.append(skip_b_flat[offset:offset + size])
        offset += size

    temp_w = []
    offset = 0
    for shape in temp_w_shapes:
        size = shape[0] * shape[1]
        temp_w.append(temp_w_flat[offset:offset + size].reshape(shape))
        offset += size

    # Compute predictions (same logic as Hebbian version)
    predictions = [jnp.zeros(s) for s in layer_sizes]

    # Top-down predictions
    for i in range(num_layers - 1):
        x_high = activation_fn(representations[i + 1])
        pred_pre = pred_w[i] @ x_high + pred_b[i]
        pred = PRED_SCALE * activation_fn(pred_pre)
        predictions[i] = predictions[i] + pred

    # Skip connection predictions
    n_skip = len(skip_w)
    for si in range(n_skip):
        enc_idx = si
        dec_idx = num_layers - 1 - si
        if enc_idx != dec_idx:
            x_enc = activation_fn(representations[enc_idx])
            skip_pre = skip_w[si] @ x_enc + skip_b[si]
            skip_pred = PRED_SCALE * activation_fn(skip_pre)
            predictions[dec_idx] = predictions[dec_idx] + skip_pred

    # Temporal predictions
    for i in range(num_layers):
        temp_pre = temp_w[i] @ activation_fn(temporal_state[i])
        temp_pred = PRED_SCALE * activation_fn(temp_pre) * 0.1
        predictions[i] = predictions[i] + temp_pred

    # Compute errors and free energy
    F = jnp.float32(0.0)
    for i in range(num_layers):
        error = representations[i] - predictions[i]
        F = F + jnp.sum(error ** 2)

    return F


def _flatten_weights(weights: List[jnp.ndarray]) -> jnp.ndarray:
    """Flatten a list of weight arrays into a single 1D array."""
    if len(weights) == 0:
        return jnp.zeros(0)
    return jnp.concatenate([w.flatten() for w in weights])


def _unflatten_weights(
    flat: jnp.ndarray, shapes: List[Tuple[int, ...]]
) -> List[jnp.ndarray]:
    """Unflatten a 1D array back into a list of arrays with given shapes."""
    result = []
    offset = 0
    for shape in shapes:
        size = 1
        for s in shape:
            size *= s
        result.append(flat[offset:offset + size].reshape(shape))
        offset += size
    return result


def _spectral_norm_clip(w: jnp.ndarray, threshold: float = 2.0) -> jnp.ndarray:
    """Clip weight matrix if approximate max singular value exceeds threshold."""
    frob = jnp.sqrt(jnp.sum(w ** 2))
    min_dim = float(min(w.shape))
    max_sv_approx = frob / jnp.sqrt(min_dim)
    return jnp.where(
        max_sv_approx > threshold,
        w * (threshold / max_sv_approx),
        w,
    )


def hierarchical_relaxation_step_autodiff(
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
    """One relaxation step with autodiff weight updates.

    Representation updates are identical to the Hebbian version.
    Weight updates use jax.grad on free energy F = sum(||e_i||^2).

    Args and returns match hierarchical_relaxation_step() exactly.
    """
    if clamp_mask is None:
        clamp_mask = {}

    num_layers = len(layer_sizes)
    activation_fn = jnp.tanh

    # ---- Step 1: Compute predictions (same as Hebbian) ----
    predictions = [jnp.zeros(s) for s in layer_sizes]

    for i in range(num_layers - 1):
        x_high = activation_fn(representations[i + 1])
        pred_pre = prediction_weights[i] @ x_high + prediction_biases[i]
        pred = PRED_SCALE * activation_fn(pred_pre)
        predictions[i] = predictions[i] + pred

    n_skip = len(skip_weights)
    for si in range(n_skip):
        enc_idx = si
        dec_idx = num_layers - 1 - si
        if enc_idx != dec_idx:
            x_enc = activation_fn(representations[enc_idx])
            skip_pre = skip_weights[si] @ x_enc + skip_biases[si]
            skip_pred = PRED_SCALE * activation_fn(skip_pre)
            predictions[dec_idx] = predictions[dec_idx] + skip_pred

    for i in range(num_layers):
        temp_pre = temporal_weights[i] @ activation_fn(temporal_state[i])
        temp_pred = PRED_SCALE * activation_fn(temp_pre) * 0.1
        predictions[i] = predictions[i] + temp_pred

    # ---- Step 2: Compute errors ----
    errors = []
    for i in range(num_layers):
        errors.append(representations[i] - predictions[i])

    # ---- Step 3: Update representations (identical to Hebbian version) ----
    new_reps = list(representations)
    for i in range(num_layers):
        if i in clamp_mask:
            new_reps[i] = clamp_mask[i]
            continue

        update = -errors[i]

        if i > 0:
            x_high_i = activation_fn(representations[i])
            pred_pre = prediction_weights[i - 1] @ x_high_i + prediction_biases[i - 1]
            pred_deriv = PRED_SCALE * (1.0 - jnp.tanh(pred_pre) ** 2)
            tanh_deriv = 1.0 - jnp.tanh(representations[i]) ** 2
            update = update + tanh_deriv * (prediction_weights[i - 1].T @ (pred_deriv * errors[i - 1]))

        if i == num_layers - 1 and output_target is not None:
            supervision_error = representations[i] - output_target
            update = update - output_supervision * supervision_error

        update = update - lambda_sparse * jnp.sign(representations[i])
        update = update - 0.01 * representations[i]

        new_reps[i] = representations[i] + lr * update
        new_reps[i] = jnp.clip(new_reps[i], -5.0, 5.0)

    # ---- Step 4: Autodiff weight updates ----
    # Collect shapes for flattening/unflattening
    pred_w_shapes = [w.shape for w in prediction_weights]
    pred_b_shapes = [b.shape for b in prediction_biases]
    skip_w_shapes = [w.shape for w in skip_weights]
    skip_b_shapes = [b.shape for b in skip_biases]
    temp_w_shapes = [w.shape for w in temporal_weights]

    # Flatten all weights
    pred_w_flat = _flatten_weights(prediction_weights)
    pred_b_flat = _flatten_weights(prediction_biases)
    skip_w_flat = _flatten_weights(skip_weights)
    skip_b_flat = _flatten_weights(skip_biases)
    temp_w_flat = _flatten_weights(temporal_weights)

    # Compute gradients of F w.r.t. all weight groups via jax.grad
    # We differentiate w.r.t. args 0-4 (pred_w, pred_b, skip_w, skip_b, temp_w)
    grad_fn = jax.grad(_compute_free_energy, argnums=(0, 1, 2, 3, 4))
    grad_pred_w_flat, grad_pred_b_flat, grad_skip_w_flat, grad_skip_b_flat, \
        grad_temp_w_flat = grad_fn(
            pred_w_flat, pred_b_flat, skip_w_flat, skip_b_flat, temp_w_flat,
            representations, temporal_state, layer_sizes,
            pred_w_shapes, pred_b_shapes, skip_w_shapes, skip_b_shapes,
            temp_w_shapes,
        )

    # Unflatten gradients
    grad_pred_w = _unflatten_weights(grad_pred_w_flat, pred_w_shapes)
    grad_pred_b = _unflatten_weights(grad_pred_b_flat, pred_b_shapes)
    grad_skip_w = _unflatten_weights(grad_skip_w_flat, skip_w_shapes)
    grad_skip_b = _unflatten_weights(grad_skip_b_flat, skip_b_shapes)
    grad_temp_w = _unflatten_weights(grad_temp_w_flat, temp_w_shapes)

    # Gradient descent on weights: w_new = w_old - lr_w * grad_F
    # (negative because we minimize F)
    new_pred_w = []
    for i in range(len(prediction_weights)):
        # Clip gradients
        grad_clipped = jnp.clip(grad_pred_w[i], -1.0, 1.0)
        w = prediction_weights[i] - lr_w * grad_clipped
        w = _spectral_norm_clip(w)
        new_pred_w.append(w)

    new_pred_b = []
    for i in range(len(prediction_biases)):
        grad_clipped = jnp.clip(grad_pred_b[i], -1.0, 1.0)
        new_pred_b.append(prediction_biases[i] - lr_w * grad_clipped)

    new_skip_w = []
    for i in range(len(skip_weights)):
        grad_clipped = jnp.clip(grad_skip_w[i], -1.0, 1.0)
        w = skip_weights[i] - lr_w * grad_clipped
        w = _spectral_norm_clip(w)
        new_skip_w.append(w)

    new_skip_b = []
    for i in range(len(skip_biases)):
        grad_clipped = jnp.clip(grad_skip_b[i], -1.0, 1.0)
        new_skip_b.append(skip_biases[i] - lr_w * grad_clipped)

    new_temp_w = []
    for i in range(len(temporal_weights)):
        grad_clipped = jnp.clip(grad_temp_w[i], -1.0, 1.0)
        w = temporal_weights[i] - lr_w * grad_clipped
        w = _spectral_norm_clip(w)
        new_temp_w.append(w)

    return new_reps, errors, new_pred_w, new_pred_b, new_skip_w, new_skip_b, new_temp_w
