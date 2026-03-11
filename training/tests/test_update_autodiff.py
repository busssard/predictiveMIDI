"""Tests for autodiff weight updates.

The autodiff update computes exact gradients of free energy F = sum(||e_i||^2)
w.r.t. weights using jax.grad, as an alternative to the Hebbian outer-product
updates. Both should produce the same return signature and converge.
"""
import jax
import jax.numpy as jnp
from training.engine.hierarchical_grid import create_hierarchical_grid
from training.engine.hierarchical_update_autodiff import (
    hierarchical_relaxation_step_autodiff,
    PRED_SCALE,
)
from training.engine.hierarchical_update import (
    hierarchical_relaxation_step,
    PRED_SCALE as HEBBIAN_PRED_SCALE,
)


class TestAutodiffUpdate:
    def test_output_shapes_match_hebbian(self):
        """Same 7-value return as Hebbian version."""
        grid = create_hierarchical_grid(layer_sizes=[16, 8, 16])
        result = hierarchical_relaxation_step_autodiff(
            grid.representations,
            grid.prediction_weights,
            grid.prediction_biases,
            grid.skip_weights,
            grid.skip_biases,
            grid.temporal_weights,
            grid.temporal_state,
            grid.layer_sizes,
            lr=0.01, lr_w=0.001, lambda_sparse=0.005,
        )
        assert len(result) == 7, f"Expected 7 return values, got {len(result)}"
        new_reps, errors, new_pred_w, new_pred_b, new_skip_w, new_skip_b, new_temp_w = result

        # Check shapes match
        assert len(new_reps) == 3
        assert new_reps[0].shape == (16,)
        assert new_reps[1].shape == (8,)
        assert new_reps[2].shape == (16,)
        assert len(errors) == 3
        assert len(new_pred_w) == 2
        assert new_pred_w[0].shape == grid.prediction_weights[0].shape
        assert len(new_pred_b) == 2
        assert new_pred_b[0].shape == grid.prediction_biases[0].shape
        assert len(new_skip_w) == 1
        assert new_skip_w[0].shape == grid.skip_weights[0].shape
        assert len(new_skip_b) == 1
        assert new_skip_b[0].shape == grid.skip_biases[0].shape
        assert len(new_temp_w) == 3
        assert new_temp_w[0].shape == grid.temporal_weights[0].shape

    def test_all_outputs_finite(self):
        """No NaN or Inf."""
        grid = create_hierarchical_grid(layer_sizes=[16, 8, 16])
        reps = [jnp.ones(s) * 0.5 for s in grid.layer_sizes]
        new_reps, errors, new_pred_w, new_pred_b, new_skip_w, new_skip_b, \
            new_temp_w = hierarchical_relaxation_step_autodiff(
                reps,
                grid.prediction_weights,
                grid.prediction_biases,
                grid.skip_weights,
                grid.skip_biases,
                grid.temporal_weights,
                grid.temporal_state,
                grid.layer_sizes,
                lr=0.01, lr_w=0.001, lambda_sparse=0.005,
            )
        for r in new_reps:
            assert jnp.all(jnp.isfinite(r)), f"Non-finite rep: {r}"
        for e in errors:
            assert jnp.all(jnp.isfinite(e)), f"Non-finite error: {e}"
        for w in new_pred_w:
            assert jnp.all(jnp.isfinite(w)), f"Non-finite pred weight"
        for b in new_pred_b:
            assert jnp.all(jnp.isfinite(b)), f"Non-finite pred bias"
        for w in new_skip_w:
            assert jnp.all(jnp.isfinite(w)), f"Non-finite skip weight"
        for b in new_skip_b:
            assert jnp.all(jnp.isfinite(b)), f"Non-finite skip bias"
        for w in new_temp_w:
            assert jnp.all(jnp.isfinite(w)), f"Non-finite temp weight"

    def test_error_decreases_over_steps(self):
        """Error should decrease with autodiff updates."""
        grid = create_hierarchical_grid(layer_sizes=[16, 8, 16])
        reps = [jax.random.normal(jax.random.PRNGKey(i), (s,)) * 0.5
                for i, s in enumerate(grid.layer_sizes)]

        errors_over_time = []
        pred_w = grid.prediction_weights
        pred_b = grid.prediction_biases
        skip_w = grid.skip_weights
        skip_b = grid.skip_biases
        temp_w = grid.temporal_weights
        for step in range(20):
            reps, errors, pred_w, pred_b, skip_w, skip_b, temp_w = \
                hierarchical_relaxation_step_autodiff(
                    reps, pred_w, pred_b, skip_w, skip_b,
                    temp_w, grid.temporal_state,
                    grid.layer_sizes,
                    lr=0.01, lr_w=0.001, lambda_sparse=0.0,
                )
            total_error = sum(float(jnp.mean(e ** 2)) for e in errors)
            errors_over_time.append(total_error)

        assert errors_over_time[-1] < errors_over_time[0], \
            f"Error didn't decrease: {errors_over_time[0]:.4f} -> {errors_over_time[-1]:.4f}"

    def test_gradient_direction_similar_to_hebbian(self):
        """With enough relaxation, Hebbian and autodiff should agree on direction.

        Run both methods on same inputs, check cosine similarity > 0
        for at least one weight matrix.
        """
        grid = create_hierarchical_grid(layer_sizes=[16, 8, 16])
        reps = [jax.random.normal(jax.random.PRNGKey(i), (s,)) * 0.5
                for i, s in enumerate(grid.layer_sizes)]

        # Run Hebbian
        _, _, hebb_pred_w, _, _, _, _ = hierarchical_relaxation_step(
            reps,
            grid.prediction_weights,
            grid.prediction_biases,
            grid.skip_weights,
            grid.skip_biases,
            grid.temporal_weights,
            grid.temporal_state,
            grid.layer_sizes,
            lr=0.01, lr_w=0.001, lambda_sparse=0.0,
        )

        # Run autodiff
        _, _, auto_pred_w, _, _, _, _ = hierarchical_relaxation_step_autodiff(
            reps,
            grid.prediction_weights,
            grid.prediction_biases,
            grid.skip_weights,
            grid.skip_biases,
            grid.temporal_weights,
            grid.temporal_state,
            grid.layer_sizes,
            lr=0.01, lr_w=0.001, lambda_sparse=0.0,
        )

        # Compute weight deltas and check cosine similarity
        at_least_one_similar = False
        for i in range(len(hebb_pred_w)):
            delta_hebb = (hebb_pred_w[i] - grid.prediction_weights[i]).flatten()
            delta_auto = (auto_pred_w[i] - grid.prediction_weights[i]).flatten()
            norm_h = jnp.linalg.norm(delta_hebb)
            norm_a = jnp.linalg.norm(delta_auto)
            if norm_h > 1e-8 and norm_a > 1e-8:
                cosine = float(jnp.dot(delta_hebb, delta_auto) / (norm_h * norm_a))
                if cosine > 0:
                    at_least_one_similar = True

        assert at_least_one_similar, \
            "No prediction weight had positive cosine similarity between Hebbian and autodiff"

    def test_pred_scale_matches_hebbian(self):
        """PRED_SCALE should be the same in both modules."""
        assert PRED_SCALE == HEBBIAN_PRED_SCALE

    def test_five_layer_hourglass(self):
        """Test with the full 5-layer hourglass architecture."""
        grid = create_hierarchical_grid(layer_sizes=[32, 16, 8, 16, 32])
        reps = [jax.random.normal(jax.random.PRNGKey(i), (s,)) * 0.1
                for i, s in enumerate(grid.layer_sizes)]

        new_reps, errors, new_pred_w, new_pred_b, new_skip_w, new_skip_b, \
            new_temp_w = hierarchical_relaxation_step_autodiff(
                reps,
                grid.prediction_weights,
                grid.prediction_biases,
                grid.skip_weights,
                grid.skip_biases,
                grid.temporal_weights,
                grid.temporal_state,
                grid.layer_sizes,
            )
        assert len(new_reps) == 5
        assert all(jnp.all(jnp.isfinite(r)) for r in new_reps)
        assert all(jnp.all(jnp.isfinite(e)) for e in errors)
        assert len(new_pred_w) == 4
        assert len(new_skip_w) == 2

    def test_prediction_weights_change(self):
        """Weights should update when there are errors."""
        grid = create_hierarchical_grid(layer_sizes=[16, 8, 16])
        reps = [jax.random.normal(jax.random.PRNGKey(i), (s,))
                for i, s in enumerate(grid.layer_sizes)]
        old_w = [w.copy() for w in grid.prediction_weights]

        _, _, new_pred_w, _, _, _, _ = hierarchical_relaxation_step_autodiff(
            reps,
            grid.prediction_weights,
            grid.prediction_biases,
            grid.skip_weights,
            grid.skip_biases,
            grid.temporal_weights,
            grid.temporal_state,
            grid.layer_sizes,
            lr=0.01, lr_w=0.01, lambda_sparse=0.0,
        )
        for old, new in zip(old_w, new_pred_w):
            assert not jnp.allclose(old, new)

    def test_skip_weights_change(self):
        """Skip connection weights should also update."""
        grid = create_hierarchical_grid(layer_sizes=[16, 8, 16])
        reps = [jax.random.normal(jax.random.PRNGKey(i), (s,))
                for i, s in enumerate(grid.layer_sizes)]
        old_skip = [w.copy() for w in grid.skip_weights]

        _, _, _, _, new_skip_w, _, _ = hierarchical_relaxation_step_autodiff(
            reps,
            grid.prediction_weights,
            grid.prediction_biases,
            grid.skip_weights,
            grid.skip_biases,
            grid.temporal_weights,
            grid.temporal_state,
            grid.layer_sizes,
            lr=0.01, lr_w=0.01, lambda_sparse=0.0,
        )
        for old, new in zip(old_skip, new_skip_w):
            assert not jnp.allclose(old, new)

    def test_temporal_weights_change(self):
        """Temporal weights should update when there's temporal prediction error."""
        grid = create_hierarchical_grid(layer_sizes=[16, 8, 16])
        reps = [jax.random.normal(jax.random.PRNGKey(i), (s,))
                for i, s in enumerate(grid.layer_sizes)]
        temp_s = [jax.random.normal(jax.random.PRNGKey(i + 10), (s,)) * 0.5
                  for i, s in enumerate(grid.layer_sizes)]
        old_temp = [w.copy() for w in grid.temporal_weights]

        _, _, _, _, _, _, new_temp_w = hierarchical_relaxation_step_autodiff(
            reps,
            grid.prediction_weights,
            grid.prediction_biases,
            grid.skip_weights,
            grid.skip_biases,
            grid.temporal_weights,
            temp_s,
            grid.layer_sizes,
            lr=0.01, lr_w=0.01, lambda_sparse=0.0,
        )
        for old, new in zip(old_temp, new_temp_w):
            assert not jnp.allclose(old, new)

    def test_clamped_layer_unchanged(self):
        """If we clamp layer 0, its representation shouldn't change."""
        grid = create_hierarchical_grid(layer_sizes=[16, 8, 16])
        clamp_val = jnp.ones(16) * 0.7
        reps = list(grid.representations)
        reps[0] = clamp_val

        new_reps, _, _, _, _, _, _ = hierarchical_relaxation_step_autodiff(
            reps,
            grid.prediction_weights,
            grid.prediction_biases,
            grid.skip_weights,
            grid.skip_biases,
            grid.temporal_weights,
            grid.temporal_state,
            grid.layer_sizes,
            lr=0.01, lr_w=0.001, lambda_sparse=0.005,
            clamp_mask={0: clamp_val},
        )
        assert jnp.allclose(new_reps[0], clamp_val)

    def test_spectral_normalization_bounds_weights(self):
        """Large weight updates should be normalized."""
        grid = create_hierarchical_grid(layer_sizes=[16, 8, 16])
        reps = [jnp.ones(s) * 10.0 for s in grid.layer_sizes]

        _, _, new_pred_w, _, _, _, _ = hierarchical_relaxation_step_autodiff(
            reps,
            grid.prediction_weights,
            grid.prediction_biases,
            grid.skip_weights,
            grid.skip_biases,
            grid.temporal_weights,
            grid.temporal_state,
            grid.layer_sizes,
            lr=0.01, lr_w=1.0,
        )
        for w in new_pred_w:
            frob = float(jnp.sqrt(jnp.sum(w ** 2)))
            min_dim = min(w.shape)
            assert frob / (min_dim ** 0.5) <= 2.0 + 1e-5

    def test_output_supervision_pulls_toward_target(self):
        """Output supervision should move output layer toward the target."""
        grid = create_hierarchical_grid(layer_sizes=[16, 8, 16])
        reps = [jnp.zeros(s) for s in grid.layer_sizes]
        target = jnp.ones(16) * 3.0

        new_reps_no_sup, _, _, _, _, _, _ = hierarchical_relaxation_step_autodiff(
            reps, grid.prediction_weights, grid.prediction_biases,
            grid.skip_weights, grid.skip_biases,
            grid.temporal_weights, grid.temporal_state,
            grid.layer_sizes,
            output_target=target, output_supervision=0.0,
        )
        new_reps_sup, _, _, _, _, _, _ = hierarchical_relaxation_step_autodiff(
            reps, grid.prediction_weights, grid.prediction_biases,
            grid.skip_weights, grid.skip_biases,
            grid.temporal_weights, grid.temporal_state,
            grid.layer_sizes,
            output_target=target, output_supervision=1.0,
        )
        dist_no = float(jnp.mean(jnp.abs(new_reps_no_sup[-1] - target)))
        dist_sup = float(jnp.mean(jnp.abs(new_reps_sup[-1] - target)))
        assert dist_sup < dist_no

    def test_prediction_biases_change(self):
        """Prediction biases should update from errors."""
        grid = create_hierarchical_grid(layer_sizes=[16, 8, 16])
        reps = [jax.random.normal(jax.random.PRNGKey(i), (s,))
                for i, s in enumerate(grid.layer_sizes)]
        old_b = [b.copy() for b in grid.prediction_biases]

        _, _, _, new_pred_b, _, _, _ = hierarchical_relaxation_step_autodiff(
            reps,
            grid.prediction_weights,
            grid.prediction_biases,
            grid.skip_weights,
            grid.skip_biases,
            grid.temporal_weights,
            grid.temporal_state,
            grid.layer_sizes,
            lr=0.01, lr_w=0.01, lambda_sparse=0.0,
        )
        for old, new in zip(old_b, new_pred_b):
            assert not jnp.allclose(old, new)

    def test_skip_biases_change(self):
        """Skip biases should update from decoder errors."""
        grid = create_hierarchical_grid(layer_sizes=[16, 8, 16])
        reps = [jax.random.normal(jax.random.PRNGKey(i), (s,))
                for i, s in enumerate(grid.layer_sizes)]
        old_b = [b.copy() for b in grid.skip_biases]

        _, _, _, _, _, new_skip_b, _ = hierarchical_relaxation_step_autodiff(
            reps,
            grid.prediction_weights,
            grid.prediction_biases,
            grid.skip_weights,
            grid.skip_biases,
            grid.temporal_weights,
            grid.temporal_state,
            grid.layer_sizes,
            lr=0.01, lr_w=0.01, lambda_sparse=0.0,
        )
        for old, new in zip(old_b, new_skip_b):
            assert not jnp.allclose(old, new)
