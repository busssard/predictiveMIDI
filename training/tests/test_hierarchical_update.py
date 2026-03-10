import jax
import jax.numpy as jnp
from training.engine.hierarchical_grid import create_hierarchical_grid
from training.engine.hierarchical_update import hierarchical_relaxation_step


class TestHierarchicalRelaxationStep:
    def test_output_shapes_match_input(self):
        grid = create_hierarchical_grid(layer_sizes=[16, 8, 16])
        new_reps, new_errors, new_pred_w, new_skip_w, new_temp_w = hierarchical_relaxation_step(
            grid.representations,
            grid.prediction_weights,
            grid.skip_weights,
            grid.temporal_weights,
            grid.temporal_state,
            grid.layer_sizes,
            lr=0.01, lr_w=0.001, lambda_sparse=0.005,
        )
        assert len(new_reps) == 3
        assert new_reps[0].shape == (16,)
        assert new_reps[1].shape == (8,)
        assert new_reps[2].shape == (16,)

    def test_error_shapes(self):
        grid = create_hierarchical_grid(layer_sizes=[16, 8, 16])
        _, errors, _, _, _ = hierarchical_relaxation_step(
            grid.representations,
            grid.prediction_weights,
            grid.skip_weights,
            grid.temporal_weights,
            grid.temporal_state,
            grid.layer_sizes,
        )
        assert len(errors) == 3
        assert errors[0].shape == (16,)
        assert errors[1].shape == (8,)

    def test_all_outputs_finite(self):
        grid = create_hierarchical_grid(layer_sizes=[16, 8, 16])
        reps = [jnp.ones(s) * 0.5 for s in grid.layer_sizes]
        new_reps, new_errors, new_pred_w, new_skip_w, _ = hierarchical_relaxation_step(
            reps,
            grid.prediction_weights,
            grid.skip_weights,
            grid.temporal_weights,
            grid.temporal_state,
            grid.layer_sizes,
            lr=0.01, lr_w=0.001, lambda_sparse=0.005,
        )
        for r in new_reps:
            assert jnp.all(jnp.isfinite(r))
        for e in new_errors:
            assert jnp.all(jnp.isfinite(e))

    def test_error_decreases_over_steps(self):
        """Running multiple relaxation steps should reduce total error."""
        grid = create_hierarchical_grid(layer_sizes=[16, 8, 16])
        reps = [jax.random.normal(jax.random.PRNGKey(i), (s,)) * 0.5
                for i, s in enumerate(grid.layer_sizes)]

        errors_over_time = []
        pred_w = grid.prediction_weights
        skip_w = grid.skip_weights
        temp_w = grid.temporal_weights
        for step in range(20):
            reps, errors, pred_w, skip_w, temp_w = hierarchical_relaxation_step(
                reps, pred_w, skip_w,
                temp_w, grid.temporal_state,
                grid.layer_sizes,
                lr=0.01, lr_w=0.001, lambda_sparse=0.0,
            )
            total_error = sum(float(jnp.mean(e ** 2)) for e in errors)
            errors_over_time.append(total_error)

        assert errors_over_time[-1] < errors_over_time[0]

    def test_clamped_layer_unchanged(self):
        """If we clamp layer 0, its representation shouldn't change."""
        grid = create_hierarchical_grid(layer_sizes=[16, 8, 16])
        clamp_val = jnp.ones(16) * 0.7
        reps = list(grid.representations)
        reps[0] = clamp_val

        new_reps, _, _, _, _ = hierarchical_relaxation_step(
            reps,
            grid.prediction_weights,
            grid.skip_weights,
            grid.temporal_weights,
            grid.temporal_state,
            grid.layer_sizes,
            lr=0.01, lr_w=0.001, lambda_sparse=0.005,
            clamp_mask={0: clamp_val},
        )
        assert jnp.allclose(new_reps[0], clamp_val)

    def test_prediction_weights_change(self):
        """Weights should update when there are errors."""
        grid = create_hierarchical_grid(layer_sizes=[16, 8, 16])
        reps = [jax.random.normal(jax.random.PRNGKey(i), (s,))
                for i, s in enumerate(grid.layer_sizes)]
        old_w = [w.copy() for w in grid.prediction_weights]

        _, _, new_pred_w, _, _ = hierarchical_relaxation_step(
            reps,
            grid.prediction_weights,
            grid.skip_weights,
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

        _, _, _, new_skip_w, _ = hierarchical_relaxation_step(
            reps,
            grid.prediction_weights,
            grid.skip_weights,
            grid.temporal_weights,
            grid.temporal_state,
            grid.layer_sizes,
            lr=0.01, lr_w=0.01, lambda_sparse=0.0,
        )
        for old, new in zip(old_skip, new_skip_w):
            assert not jnp.allclose(old, new)

    def test_five_layer_hourglass(self):
        """Test with the full 5-layer hourglass architecture."""
        grid = create_hierarchical_grid(layer_sizes=[128, 64, 32, 64, 128])
        reps = [jax.random.normal(jax.random.PRNGKey(i), (s,)) * 0.1
                for i, s in enumerate(grid.layer_sizes)]

        new_reps, errors, _, _, _ = hierarchical_relaxation_step(
            reps,
            grid.prediction_weights,
            grid.skip_weights,
            grid.temporal_weights,
            grid.temporal_state,
            grid.layer_sizes,
        )
        assert len(new_reps) == 5
        assert all(jnp.all(jnp.isfinite(r)) for r in new_reps)
        assert all(jnp.all(jnp.isfinite(e)) for e in errors)

    def test_spectral_normalization_bounds_weights(self):
        """Large weight updates should be normalized."""
        grid = create_hierarchical_grid(layer_sizes=[16, 8, 16])
        # Create large representations to generate large gradients
        reps = [jnp.ones(s) * 10.0 for s in grid.layer_sizes]

        _, _, new_pred_w, _, _ = hierarchical_relaxation_step(
            reps,
            grid.prediction_weights,
            grid.skip_weights,
            grid.temporal_weights,
            grid.temporal_state,
            grid.layer_sizes,
            lr=0.01, lr_w=1.0,  # large lr_w to trigger normalization
        )
        for w in new_pred_w:
            frob = float(jnp.sqrt(jnp.sum(w ** 2)))
            min_dim = min(w.shape)
            # After normalization, Frobenius/sqrt(min_dim) should be <= 2.0
            assert frob / (min_dim ** 0.5) <= 2.0 + 1e-5

    def test_temporal_weights_change(self):
        """Temporal weights should update when there's temporal prediction error."""
        grid = create_hierarchical_grid(layer_sizes=[16, 8, 16])
        reps = [jax.random.normal(jax.random.PRNGKey(i), (s,))
                for i, s in enumerate(grid.layer_sizes)]
        # Set non-zero temporal state to generate temporal errors
        temp_s = [jax.random.normal(jax.random.PRNGKey(i + 10), (s,)) * 0.5
                  for i, s in enumerate(grid.layer_sizes)]
        old_temp = [w.copy() for w in grid.temporal_weights]

        _, _, _, _, new_temp_w = hierarchical_relaxation_step(
            reps,
            grid.prediction_weights,
            grid.skip_weights,
            grid.temporal_weights,
            temp_s,
            grid.layer_sizes,
            lr=0.01, lr_w=0.01, lambda_sparse=0.0,
        )
        for old, new in zip(old_temp, new_temp_w):
            assert not jnp.allclose(old, new)

    def test_output_supervision_pulls_toward_target(self):
        """Output supervision should move output layer toward the target."""
        grid = create_hierarchical_grid(layer_sizes=[16, 8, 16])
        reps = [jnp.zeros(s) for s in grid.layer_sizes]
        target = jnp.ones(16) * 3.0  # strong positive target

        # Without supervision
        new_reps_no_sup, _, _, _, _ = hierarchical_relaxation_step(
            reps, grid.prediction_weights, grid.skip_weights,
            grid.temporal_weights, grid.temporal_state, grid.layer_sizes,
            output_target=target, output_supervision=0.0,
        )
        # With supervision
        new_reps_sup, _, _, _, _ = hierarchical_relaxation_step(
            reps, grid.prediction_weights, grid.skip_weights,
            grid.temporal_weights, grid.temporal_state, grid.layer_sizes,
            output_target=target, output_supervision=1.0,
        )
        # Supervised output should be closer to target
        dist_no = float(jnp.mean(jnp.abs(new_reps_no_sup[-1] - target)))
        dist_sup = float(jnp.mean(jnp.abs(new_reps_sup[-1] - target)))
        assert dist_sup < dist_no
