"""Hierarchical PC inference: input-only clamping, free output generation."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from training.engine.hierarchical_grid import HierarchicalGridState, create_hierarchical_grid
from training.engine.hierarchical_update import hierarchical_relaxation_step


def load_hierarchical_checkpoint(checkpoint_path):
    """Load a hierarchical checkpoint into grid + metadata."""
    path = Path(checkpoint_path)
    metadata = json.loads((path / "metadata.json").read_text())

    layer_sizes = metadata["layer_sizes"]
    vocabulary = metadata.get("vocabulary", {})
    num_instruments = metadata.get("conditioning_size", len(vocabulary))

    grid = create_hierarchical_grid(
        layer_sizes=layer_sizes,
        num_instruments=num_instruments,
        lr=metadata.get("lr", 0.01),
        lr_w=metadata.get("lr_w", 0.001),
        alpha=metadata.get("alpha", 0.8),
        lambda_sparse=metadata.get("lambda_sparse", 0.005),
    )

    # Load arrays
    for i in range(len(layer_sizes)):
        f = path / f"layer_{i}_rep.npy"
        if f.exists():
            grid.representations[i] = jnp.array(np.load(f))
        f = path / f"layer_{i}_temporal.npy"
        if f.exists():
            grid.temporal_state[i] = jnp.array(np.load(f))

    for i in range(len(grid.prediction_weights)):
        f = path / f"pred_weight_{i}.npy"
        if f.exists():
            grid.prediction_weights[i] = jnp.array(np.load(f))

    for i in range(len(grid.skip_weights)):
        f = path / f"skip_weight_{i}.npy"
        if f.exists():
            grid.skip_weights[i] = jnp.array(np.load(f))

    for i in range(len(grid.temporal_weights)):
        f = path / f"temporal_weight_{i}.npy"
        if f.exists():
            grid.temporal_weights[i] = jnp.array(np.load(f))

    return grid, metadata


def run_hierarchical_inference(grid, input_piano_roll, conditioning_vec,
                               relaxation_steps=32):
    """Run inference: clamp input layer, let output evolve freely.

    Args:
        grid: HierarchicalGridState
        input_piano_roll: (T, 128) float32
        conditioning_vec: (num_cat,) float32 one-hot
        relaxation_steps: steps per tick

    Returns:
        output_roll: (T, 128) float32 — sigmoid probabilities
        all_layer_states: list of (T, H_l) per layer
    """
    T = input_piano_roll.shape[0]
    h_input = grid.layer_sizes[0]
    h_output = grid.layer_sizes[-1]

    # Build conditioning (add to input)
    num_cat = len(conditioning_vec)
    cond_start = (h_input - num_cat) // 2
    cond_full = np.zeros(h_input, dtype=np.float32)
    cond_full[cond_start:cond_start + num_cat] = conditioning_vec

    # JIT the relaxation step
    @jax.jit
    def relax_step(reps, pred_w, skip_w, temp_w, temp_s, clamp_input):
        new_reps, errors, new_pred_w, new_skip_w, new_temp_w = hierarchical_relaxation_step(
            reps, pred_w, skip_w, temp_w, temp_s,
            grid.layer_sizes,
            lr=grid.lr * 0.5,  # reduced lr for inference stability
            lr_w=0.0,  # no weight learning
            lambda_sparse=grid.lambda_sparse,
        )
        # Clamp only input layer
        new_reps_out = list(new_reps)
        new_reps_out[0] = clamp_input
        return new_reps_out, errors, new_pred_w, new_skip_w

    reps = list(grid.representations)
    pred_w = list(grid.prediction_weights)
    skip_w = list(grid.skip_weights)
    temp_w = list(grid.temporal_weights)
    temp_s = list(grid.temporal_state)

    outputs = []
    all_layer_states = [[] for _ in grid.layer_sizes]

    for t in range(T):
        inp = np.zeros(h_input, dtype=np.float32)
        inp[:min(128, h_input)] = input_piano_roll[t, :min(128, h_input)]
        inp_with_cond = jnp.clip(jnp.array(inp) + jnp.array(cond_full), 0.0, 1.0)

        for step in range(relaxation_steps):
            reps, errors, pred_w, skip_w = relax_step(
                reps, pred_w, skip_w, temp_w, temp_s, inp_with_cond)

        # Read output
        output = jax.nn.sigmoid(reps[-1])
        out_128 = np.zeros(128, dtype=np.float32)
        out_128[:min(128, h_output)] = np.array(output[:min(128, h_output)])
        outputs.append(out_128)

        # Store layer states
        for i, r in enumerate(reps):
            all_layer_states[i].append(np.array(r))

        # Update temporal state
        temp_s = [r for r in reps]

    output_roll = np.array(outputs)
    all_layer_states = [np.array(s) for s in all_layer_states]

    return output_roll, all_layer_states
