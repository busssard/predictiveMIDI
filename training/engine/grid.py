from dataclasses import dataclass
import jax
import jax.numpy as jnp


@dataclass
class GridState:
    """Holds all arrays that define the PC grid.

    state: (H, W, 4) -- r, e, s, h per neuron
    weights: (H, W, 4) -- w_left, w_right, w_up, w_down
    params: (H, W, 4) -- alpha, beta, learning_rate, bias
    input_mask: (H, W) bool -- True for input-clamped neurons
    output_mask: (H, W) bool -- True for output-clamped neurons
    conditioning_mask: (H, W) bool -- True for conditioning neurons
    """
    state: jnp.ndarray
    weights: jnp.ndarray
    params: jnp.ndarray
    input_mask: jnp.ndarray
    output_mask: jnp.ndarray
    conditioning_mask: jnp.ndarray


def create_grid(size=128, num_instruments=8, key=None):
    """Initialize a PC grid with small random values.

    Args:
        size: grid dimension (size x size)
        num_instruments: number of instrument categories (one-hot block size)
        key: JAX PRNG key. Uses jax.random.PRNGKey(0) if None.
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)

    # State: small random init for representation, zeros for error/memory
    r = jax.random.normal(k1, (size, size)) * 0.01
    e = jnp.zeros((size, size))
    s = jnp.zeros((size, size))
    h = jnp.zeros((size, size))
    state = jnp.stack([r, e, s, h], axis=-1)

    # Weights: small random init
    weights = jax.random.normal(k2, (size, size, 4)) * 0.1

    # Params: alpha ~ 0.9 (slow decay), beta ~ 0.1 (weak recurrence),
    # lr = 0.01, bias = 0
    alpha = jnp.full((size, size), 0.9)
    beta = jnp.full((size, size), 0.1)
    lr = jnp.full((size, size), 0.01)
    bias = jnp.zeros((size, size))
    params = jnp.stack([alpha, beta, lr, bias], axis=-1)

    # Clamp masks
    input_mask = jnp.zeros((size, size), dtype=bool).at[:, 0].set(True)
    output_mask = jnp.zeros((size, size), dtype=bool).at[:, -1].set(True)

    # Conditioning: centered vertically in column 0
    cond_start = (size - num_instruments) // 2
    conditioning_mask = jnp.zeros((size, size), dtype=bool)
    conditioning_mask = conditioning_mask.at[
        cond_start:cond_start + num_instruments, 0
    ].set(True)

    return GridState(
        state=state,
        weights=weights,
        params=params,
        input_mask=input_mask,
        output_mask=output_mask,
        conditioning_mask=conditioning_mask,
    )
