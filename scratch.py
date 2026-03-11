"""Verify bounded prediction dynamics: free relaxation convergence test.

Creates a fresh hierarchical grid, runs 100 free relaxation steps with no
clamping (lr_w=0 to isolate representation dynamics), and prints the error
curve. Verifies representations converge to a fixed point.
"""
import jax
import jax.numpy as jnp
from training.engine.hierarchical_grid import create_hierarchical_grid
from training.engine.hierarchical_update import hierarchical_relaxation_step, PRED_SCALE

print(f"PRED_SCALE = {PRED_SCALE}")
print()

# Create a fresh grid with the default hourglass architecture
grid = create_hierarchical_grid(layer_sizes=[128, 64, 32, 64, 128])

# Initialize representations with moderate random values
reps = [jax.random.normal(jax.random.PRNGKey(i), (s,)) * 0.5
        for i, s in enumerate(grid.layer_sizes)]

pred_w = grid.prediction_weights
pred_b = grid.prediction_biases
skip_w = grid.skip_weights
skip_b = grid.skip_biases
temp_w = grid.temporal_weights

print("=== Free Relaxation (no clamping, no weight learning) ===")
print(f"{'Step':>5}  {'Total MSE':>12}  {'Max |rep|':>10}  {'Max |err|':>10}")
print("-" * 50)

errors_over_time = []
for step in range(100):
    reps, errors, pred_w, pred_b, skip_w, skip_b, temp_w = \
        hierarchical_relaxation_step(
            reps, pred_w, pred_b, skip_w, skip_b,
            temp_w, grid.temporal_state, grid.layer_sizes,
            lr=0.01, lr_w=0.0, lambda_sparse=0.0,
        )
    total_mse = sum(float(jnp.mean(e**2)) for e in errors)
    max_rep = max(float(jnp.max(jnp.abs(r))) for r in reps)
    max_err = max(float(jnp.max(jnp.abs(e))) for e in errors)
    errors_over_time.append(total_mse)

    if step < 10 or step % 10 == 9:
        print(f"{step:5d}  {total_mse:12.6f}  {max_rep:10.4f}  {max_err:10.4f}")

print()
print(f"Error reduction: {errors_over_time[0]:.6f} -> {errors_over_time[-1]:.6f}")
print(f"  Ratio: {errors_over_time[-1] / errors_over_time[0]:.4f}x")
print()

# Check fixed point convergence: compare last two steps
print("=== Fixed Point Check ===")
reps_before = [r.copy() for r in reps]
reps_after, _, _, _, _, _, _ = hierarchical_relaxation_step(
    reps, pred_w, pred_b, skip_w, skip_b,
    temp_w, grid.temporal_state, grid.layer_sizes,
    lr=0.01, lr_w=0.0, lambda_sparse=0.0,
)
for i, (rb, ra) in enumerate(zip(reps_before, reps_after)):
    delta = float(jnp.max(jnp.abs(ra - rb)))
    print(f"  Layer {i} (size {grid.layer_sizes[i]:3d}): max |delta| = {delta:.8f}")

print()
print("=== Per-Layer Error at Final Step ===")
for i, e in enumerate(errors):
    print(f"  Layer {i}: mean |e| = {float(jnp.mean(jnp.abs(e))):.6f}, "
          f"max |e| = {float(jnp.max(jnp.abs(e))):.6f}")

print()
print("=== Per-Layer Representation Stats at Final Step ===")
for i, r in enumerate(reps):
    print(f"  Layer {i}: mean = {float(jnp.mean(r)):+.4f}, "
          f"std = {float(jnp.std(r)):.4f}, "
          f"min = {float(jnp.min(r)):+.4f}, "
          f"max = {float(jnp.max(r)):+.4f}")
