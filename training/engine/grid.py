from dataclasses import dataclass, field
import jax
import jax.numpy as jnp


@dataclass
class GridState:
    """Holds all arrays that define the PC grid.

    state: (H, W, 4) -- r, e, s, h per neuron
    weights: (H, W, 4) -- w_left, w_right, w_up, w_down
    params: (H, W, 4) -- alpha, beta, learning_rate, bias
    log_precision: (H, W) -- learned log-precision per neuron
    input_mask: (H, W) bool -- True for input-clamped neurons
    output_mask: (H, W) bool -- True for output-clamped neurons
    conditioning_mask: (H, W) bool -- True for conditioning neurons
    pos_weight: float -- positive-class weight for boundary precision
    lambda_sparse: float -- L1 sparsity penalty coefficient
    connectivity: str -- "neighbor", "fc", or "fc_double"
    fc_weights: jnp.ndarray -- (W, H, H) FC inter-column weights (or None)
    fc_skip_weights: jnp.ndarray -- (W, H, H) FC skip-2 weights (or None)
    w_temporal: jnp.ndarray -- (H, W) learnable temporal gain per neuron
    spike_boost: float -- column wavefront spiking boost
    asl_gamma_neg: float -- ASL negative-class focusing parameter
    asl_margin: float -- ASL probability shift margin
    lr_w: float -- independent weight learning rate
    """
    state: jnp.ndarray
    weights: jnp.ndarray
    params: jnp.ndarray
    log_precision: jnp.ndarray
    input_mask: jnp.ndarray
    output_mask: jnp.ndarray
    conditioning_mask: jnp.ndarray
    pos_weight: float = 20.0
    lambda_sparse: float = 0.01
    connectivity: str = "neighbor"
    fc_weights: jnp.ndarray = None
    fc_skip_weights: jnp.ndarray = None
    w_temporal: jnp.ndarray = None
    spike_boost: float = 5.0
    asl_gamma_neg: float = 4.0
    asl_margin: float = 0.05
    lr_w: float = 0.001


def create_grid(width=16, height=128, num_instruments=8, key=None,
                lr=0.005, alpha=0.9, beta=0.1,
                pos_weight=20.0, lambda_sparse=0.01,
                connectivity="neighbor", lr_w=0.001,
                spike_boost=5.0, asl_gamma_neg=4.0, asl_margin=0.05,
                lr_amplification=0.0):
    """Initialize a PC grid with small random values.

    Args:
        width: grid columns (processing depth)
        height: grid rows (MIDI pitches)
        num_instruments: number of instrument categories (one-hot block size)
        key: JAX PRNG key. Uses jax.random.PRNGKey(0) if None.
        lr: representation learning rate
        alpha: leaky integrator decay rate (0=fast, 1=slow)
        beta: recurrent state gain
        pos_weight: positive-class weight for boundary precision.
        lambda_sparse: L1 sparsity penalty coefficient.
        connectivity: "neighbor", "fc", or "fc_double"
        lr_w: independent weight learning rate
        spike_boost: column wavefront spiking precision boost
        asl_gamma_neg: ASL negative-class focusing parameter
        asl_margin: ASL probability shift margin
        lr_amplification: per-column lr scaling (0.0 = disabled)
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)

    # State: small random init for representation, zeros for error/memory
    r = jax.random.normal(k1, (height, width)) * 0.01
    e = jnp.zeros((height, width))
    s = jnp.zeros((height, width))
    h = jnp.zeros((height, width))
    state = jnp.stack([r, e, s, h], axis=-1)

    # Weights: small random init
    weights = jax.random.normal(k2, (height, width, 4)) * 0.1

    # Per-column lr scaling
    lr_base = jnp.full((height, width), lr)
    if lr_amplification > 0.0 and width > 1:
        cols = jnp.arange(width)[None, :]  # (1, W)
        lr_scale = 1.0 + lr_amplification * (1.0 - jnp.abs(cols - (width - 1) / 2) / ((width - 1) / 2))
        lr_base = lr_base * lr_scale

    alpha_arr = jnp.full((height, width), alpha)
    beta_arr = jnp.full((height, width), beta)
    bias = jnp.zeros((height, width))
    params = jnp.stack([alpha_arr, beta_arr, lr_base, bias], axis=-1)

    # Learned log-precision: initialized to zero (precision = 1.0)
    log_precision = jnp.zeros((height, width))

    # Clamp masks
    input_mask = jnp.zeros((height, width), dtype=bool).at[:, 0].set(True)
    output_mask = jnp.zeros((height, width), dtype=bool).at[:, -1].set(True)

    # Conditioning: centered vertically in column 0
    cond_start = (height - num_instruments) // 2
    conditioning_mask = jnp.zeros((height, width), dtype=bool)
    conditioning_mask = conditioning_mask.at[
        cond_start:cond_start + num_instruments, 0
    ].set(True)

    # FC weights for fully-connected inter-column connectivity
    fc_weights = None
    fc_skip_weights = None
    if connectivity in ("fc", "fc_double"):
        # Xavier init: normal * 0.1 / sqrt(H)
        scale = 0.1 / jnp.sqrt(float(height))
        fc_weights = jax.random.normal(k4, (width, height, height)) * scale
        if connectivity == "fc_double":
            fc_skip_weights = jax.random.normal(k5, (width, height, height)) * scale

    # Learnable temporal gain, initialized to beta
    w_temporal = jnp.full((height, width), beta)

    return GridState(
        state=state,
        weights=weights,
        params=params,
        log_precision=log_precision,
        input_mask=input_mask,
        output_mask=output_mask,
        conditioning_mask=conditioning_mask,
        pos_weight=pos_weight,
        lambda_sparse=lambda_sparse,
        connectivity=connectivity,
        fc_weights=fc_weights,
        fc_skip_weights=fc_skip_weights,
        w_temporal=w_temporal,
        spike_boost=spike_boost,
        asl_gamma_neg=asl_gamma_neg,
        asl_margin=asl_margin,
        lr_w=lr_w,
    )
