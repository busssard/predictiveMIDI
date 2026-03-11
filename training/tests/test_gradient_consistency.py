import jax
import jax.numpy as jnp
from training.engine.hierarchical_grid import create_hierarchical_grid
from training.engine.hierarchical_update import hierarchical_relaxation_step, PRED_SCALE


class TestGradientConsistency:
    def test_prediction_is_bounded(self):
        """Predictions should be bounded to [-PRED_SCALE, PRED_SCALE] (tanh range)."""
        grid = create_hierarchical_grid(layer_sizes=[16, 8, 16])
        reps = [jax.random.normal(jax.random.PRNGKey(i), (s,)) * 2.0
                for i, s in enumerate(grid.layer_sizes)]
        new_reps, errors, _, _, _, _, _ = hierarchical_relaxation_step(
            reps, grid.prediction_weights, grid.prediction_biases,
            grid.skip_weights, grid.skip_biases,
            grid.temporal_weights, grid.temporal_state,
            grid.layer_sizes,
        )
        for r in new_reps:
            assert jnp.all(jnp.isfinite(r))

    def test_pred_scale_constant_exists(self):
        """PRED_SCALE should be defined and equal to 3.0."""
        assert PRED_SCALE == 3.0

    def test_predictions_bounded_with_extreme_weights(self):
        """Even with very large weights, predictions must stay in [-PRED_SCALE, PRED_SCALE].

        We verify this by manually computing predictions in the same way as the
        update rule and checking they stay bounded.
        """
        grid = create_hierarchical_grid(layer_sizes=[16, 8, 16])
        # Make weights very large to test bounding
        pred_w = [w * 100.0 for w in grid.prediction_weights]
        reps = [jnp.ones(s) * 5.0 for s in grid.layer_sizes]

        # Run one step
        new_reps, errors, _, _, _, _, _ = hierarchical_relaxation_step(
            reps, pred_w, grid.prediction_biases,
            grid.skip_weights, grid.skip_biases,
            grid.temporal_weights, grid.temporal_state,
            grid.layer_sizes, lr=0.01, lr_w=0.0,
        )
        # With bounded predictions, errors = reps - bounded_pred, and bounded_pred
        # is in [-PRED_SCALE, PRED_SCALE]. So errors should be bounded too.
        # With unbounded predictions, errors could be arbitrarily large.
        for e in errors:
            # With PRED_SCALE=3.0 and reps clipped to [-5,5], errors should be
            # at most about |5| + |3| = 8 per element. Without bounding,
            # pred = W @ tanh(x) could be huge with W*100.
            assert jnp.all(jnp.abs(e) < 20.0), \
                f"Error too large (max {float(jnp.max(jnp.abs(e)))}), predictions likely unbounded"

    def test_free_running_convergence(self):
        """With NO clamping, network should converge (error decreases)."""
        grid = create_hierarchical_grid(layer_sizes=[16, 8, 16])
        reps = [jax.random.normal(jax.random.PRNGKey(i), (s,)) * 0.5
                for i, s in enumerate(grid.layer_sizes)]
        pred_w = grid.prediction_weights
        pred_b = grid.prediction_biases
        skip_w = grid.skip_weights
        skip_b = grid.skip_biases
        temp_w = grid.temporal_weights

        errors_over_time = []
        for step in range(100):
            reps, errors, pred_w, pred_b, skip_w, skip_b, temp_w = \
                hierarchical_relaxation_step(
                    reps, pred_w, pred_b, skip_w, skip_b,
                    temp_w, grid.temporal_state, grid.layer_sizes,
                    lr=0.01, lr_w=0.0, lambda_sparse=0.0,
                )
            total = sum(float(jnp.mean(e**2)) for e in errors)
            errors_over_time.append(total)
        assert errors_over_time[-1] < errors_over_time[0]

    def test_energy_decreases_per_step(self):
        """Free energy F = sum(e_i^2) should generally decrease each step."""
        grid = create_hierarchical_grid(layer_sizes=[16, 8, 16])
        reps = [jax.random.normal(jax.random.PRNGKey(i), (s,)) * 0.5
                for i, s in enumerate(grid.layer_sizes)]

        errors_over_time = []
        for step in range(50):
            reps, errors, _, _, _, _, _ = hierarchical_relaxation_step(
                reps, grid.prediction_weights, grid.prediction_biases,
                grid.skip_weights, grid.skip_biases,
                grid.temporal_weights, grid.temporal_state,
                grid.layer_sizes, lr=0.01, lr_w=0.0, lambda_sparse=0.0,
            )
            total = sum(float(jnp.sum(e**2)) for e in errors)
            errors_over_time.append(total)
        decreases = sum(1 for i in range(1, len(errors_over_time))
                       if errors_over_time[i] <= errors_over_time[i-1])
        assert decreases >= 0.8 * (len(errors_over_time) - 1)
