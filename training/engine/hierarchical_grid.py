"""Hierarchical PC network state.

A layered predictive coding network where higher layers predict lower layers.
Layers form an hourglass: 128→64→32→64→128 (encoder-decoder with bottleneck).

Predictions flow outward from bottleneck, errors flow inward toward bottleneck.
Skip connections link encoder layers to their decoder mirrors (U-Net style).
"""
from dataclasses import dataclass
from typing import List, Optional
import jax
import jax.numpy as jnp


@dataclass
class HierarchicalGridState:
    """Hierarchical PC network state.

    representations: list of (H_l,) arrays — one per layer
    errors: list of (H_l,) arrays — prediction errors per layer
    prediction_weights: list of (H_low, H_high) — W_pred[i] predicts layer i from i+1
    prediction_biases: list of (H_low,) — per-layer prediction bias
    skip_weights: list of (H_dec, H_enc) — skip from encoder i to decoder N-1-i
    temporal_weights: list of (H_l, H_l) — temporal prediction per layer
    temporal_state: list of (H_l,) — previous tick's representation
    layer_sizes: list of ints
    conditioning_size: int
    lr: float
    lr_w: float
    alpha: float — temporal decay
    lambda_sparse: float — L1 sparsity
    """
    representations: List[jnp.ndarray]
    errors: List[jnp.ndarray]
    prediction_weights: List[jnp.ndarray]
    prediction_biases: List[jnp.ndarray]
    skip_weights: List[jnp.ndarray]
    temporal_weights: List[jnp.ndarray]
    temporal_state: List[jnp.ndarray]
    layer_sizes: List[int]
    conditioning_size: int = 16
    lr: float = 0.01
    lr_w: float = 0.001
    alpha: float = 0.8
    lambda_sparse: float = 0.005


def create_hierarchical_grid(
    layer_sizes=None,
    num_instruments=16,
    key=None,
    lr=0.01,
    lr_w=0.001,
    alpha=0.8,
    lambda_sparse=0.005,
):
    """Create a hierarchical PC grid with small random weights.

    Args:
        layer_sizes: list of layer dimensions, e.g. [128, 64, 32, 64, 128].
            First = input layer, last = output layer.
        num_instruments: size of one-hot conditioning vector.
        key: JAX PRNG key.
        lr: representation learning rate.
        lr_w: weight learning rate.
        alpha: temporal state decay (0=no memory, 1=full memory).
        lambda_sparse: L1 sparsity penalty coefficient.
    """
    if layer_sizes is None:
        layer_sizes = [128, 64, 32, 64, 128]
    if key is None:
        key = jax.random.PRNGKey(42)

    num_layers = len(layer_sizes)

    # Small noise init for representations (breaks symmetry)
    representations = []
    for s in layer_sizes:
        key, subkey = jax.random.split(key)
        representations.append(jax.random.normal(subkey, (s,)) * 0.01)
    errors = [jnp.zeros(s) for s in layer_sizes]
    temporal_state = [jnp.zeros(s) for s in layer_sizes]

    # Prediction weights: W_pred[i] has shape (layer_sizes[i], layer_sizes[i+1])
    # Layer i+1 predicts layer i via: pred_i = tanh(W_pred[i] @ x[i+1])
    # Xavier initialization: sqrt(2 / (fan_in + fan_out))
    prediction_weights = []
    for i in range(num_layers - 1):
        key, subkey = jax.random.split(key)
        h_low, h_high = layer_sizes[i], layer_sizes[i + 1]
        scale = jnp.sqrt(2.0 / (h_low + h_high))
        w = jax.random.normal(subkey, (h_low, h_high)) * scale
        prediction_weights.append(w)

    # Prediction biases: (H_low,) per layer — captures average prediction
    # Init to zero; the bias will learn the mean target during training
    prediction_biases = [jnp.zeros(layer_sizes[i]) for i in range(num_layers - 1)]

    # Skip connections: encoder layer i → decoder layer (N-1-i)
    # Number of skip pairs = floor(num_layers / 2)
    # Same Xavier scale as prediction weights — skip connections are critical
    skip_weights = []
    n_skip = num_layers // 2
    for i in range(n_skip):
        j = num_layers - 1 - i
        key, subkey = jax.random.split(key)
        h_enc, h_dec = layer_sizes[i], layer_sizes[j]
        scale = jnp.sqrt(2.0 / (h_enc + h_dec))
        w = jax.random.normal(subkey, (h_dec, h_enc)) * scale
        skip_weights.append(w)

    # Temporal weights: predict current tick from previous tick, per layer
    # Smaller than spatial weights (temporal is auxiliary)
    temporal_weights = []
    for i in range(num_layers):
        key, subkey = jax.random.split(key)
        s = layer_sizes[i]
        scale = jnp.sqrt(1.0 / s)  # smaller than Xavier but non-trivial
        w = jax.random.normal(subkey, (s, s)) * scale
        temporal_weights.append(w)

    return HierarchicalGridState(
        representations=representations,
        errors=errors,
        prediction_weights=prediction_weights,
        prediction_biases=prediction_biases,
        skip_weights=skip_weights,
        temporal_weights=temporal_weights,
        temporal_state=temporal_state,
        layer_sizes=layer_sizes,
        conditioning_size=num_instruments,
        lr=lr,
        lr_w=lr_w,
        alpha=alpha,
        lambda_sparse=lambda_sparse,
    )
