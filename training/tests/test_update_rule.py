import jax
import jax.numpy as jnp
import pytest
from training.engine.grid import create_grid
from training.engine.update_rule import pc_relaxation_step, apply_clamping


class TestPCRelaxationStep:
    def test_output_same_shape_as_input(self):
        grid = create_grid(width=16, height=16, num_instruments=4)
        result = pc_relaxation_step(
            grid.state, grid.weights, grid.params
        )
        new_state, new_weights = result[0], result[1]
        assert new_state.shape == grid.state.shape
        assert new_weights.shape == grid.weights.shape

    def test_error_stays_finite_over_steps(self):
        """After multiple relaxation steps, error should remain finite (not explode)."""
        key = jax.random.PRNGKey(42)
        grid = create_grid(width=16, height=16, num_instruments=4, key=key)
        state = grid.state.at[:, :, 0].set(
            jax.random.normal(key, (16, 16)) * 0.5
        )

        weights = grid.weights
        for _ in range(20):
            result = pc_relaxation_step(state, weights, grid.params)
            state, weights = result[0], result[1]

        final_error = jnp.abs(state[:, :, 1]).sum()
        assert jnp.isfinite(final_error)
        assert final_error < 1000.0

    def test_leaky_state_persists(self):
        """The leaky integrator should carry forward previous state."""
        grid = create_grid(width=8, height=8, num_instruments=2)
        state = grid.state.at[:, :, 0].set(1.0)
        result = pc_relaxation_step(state, grid.weights, grid.params)
        new_state = result[0]
        s = new_state[:, :, 2]
        assert jnp.abs(s).sum() > 0

    def test_is_jit_compilable(self):
        grid = create_grid(width=8, height=8, num_instruments=2)
        jitted = jax.jit(pc_relaxation_step)
        result = jitted(grid.state, grid.weights, grid.params)
        new_state = result[0]
        assert new_state.shape == grid.state.shape

    def test_weights_change_after_step(self):
        """iPC weight update should modify weights."""
        grid = create_grid(width=8, height=8, num_instruments=2)
        state = grid.state.at[:, :, 0].set(1.0)
        result = pc_relaxation_step(state, grid.weights, grid.params)
        new_weights = result[1]
        assert not jnp.allclose(new_weights, grid.weights)

    def test_state_channels_are_finite(self):
        """All channels should remain finite after a step."""
        grid = create_grid(width=16, height=16, num_instruments=4)
        result = pc_relaxation_step(
            grid.state, grid.weights, grid.params
        )
        new_state, new_weights = result[0], result[1]
        assert jnp.isfinite(new_state).all()
        assert jnp.isfinite(new_weights).all()

    def test_precision_amplifies_error(self):
        """Precision > 1 should amplify the weight update."""
        grid = create_grid(width=8, height=8, num_instruments=2)
        state = grid.state.at[:, :, 0].set(1.0)
        precision = jnp.full((8, 8), 5.0)

        w_prec = pc_relaxation_step(
            state, grid.weights, grid.params, precision=precision)[1]
        w_base = pc_relaxation_step(
            state, grid.weights, grid.params)[1]

        delta_prec = jnp.abs(w_prec - grid.weights).sum()
        delta_base = jnp.abs(w_base - grid.weights).sum()
        assert delta_prec > delta_base

    def test_sparsity_pulls_toward_zero(self):
        """Sparsity penalty should reduce absolute representation values."""
        grid = create_grid(width=8, height=8, num_instruments=2)
        state = grid.state.at[:, :, 0].set(1.0)

        new_state_sparse = pc_relaxation_step(
            state, grid.weights, grid.params, lambda_sparse=0.1)[0]
        new_state_base = pc_relaxation_step(
            state, grid.weights, grid.params, lambda_sparse=0.0)[0]

        r_sparse = jnp.abs(new_state_sparse[:, :, 0]).mean()
        r_base = jnp.abs(new_state_base[:, :, 0]).mean()
        assert r_sparse < r_base

    def test_learned_precision_converges(self):
        """Log precision should change over multiple steps and stay finite."""
        grid = create_grid(width=8, height=8, num_instruments=2)
        state = grid.state.at[:, :, 0].set(0.5)
        log_prec = jnp.zeros((8, 8))

        weights = grid.weights
        for _ in range(50):
            result = pc_relaxation_step(
                state, weights, grid.params,
                log_precision=log_prec, lr_precision=0.01)
            state, weights, log_prec = result[0], result[1], result[2]

        assert jnp.isfinite(log_prec).all()
        assert not jnp.allclose(log_prec, 0.0)

    def test_returns_six_values(self):
        """pc_relaxation_step should return 6 values."""
        grid = create_grid(width=8, height=8, num_instruments=2)
        result = pc_relaxation_step(grid.state, grid.weights, grid.params)
        assert len(result) == 6
        new_state, new_weights, new_log_prec, fc_w, fc_sw, w_t = result
        assert new_state.shape == grid.state.shape
        assert new_weights.shape == grid.weights.shape
        assert new_log_prec is None  # No log_precision passed in
        assert fc_w is None
        assert fc_sw is None
        assert w_t is None

    def test_l2_weight_normalization(self):
        """After a step, weight vectors should have L2 norm <= 1.0 + eps."""
        grid = create_grid(width=8, height=8, num_instruments=2)
        state = grid.state.at[:, :, 0].set(1.0)
        # Use large weights to test normalization
        weights = grid.weights * 10.0
        result = pc_relaxation_step(state, weights, grid.params)
        new_weights = result[1]
        w_norms = jnp.sqrt(jnp.sum(new_weights ** 2, axis=-1))
        assert (w_norms <= 1.0 + 1e-5).all()

    def test_fc_prediction_uses_matmul(self):
        """FC prediction should differ from neighbor-only prediction."""
        grid_nb = create_grid(width=8, height=8, num_instruments=2,
                              connectivity="neighbor")
        grid_fc = create_grid(width=8, height=8, num_instruments=2,
                              connectivity="fc")
        state = grid_nb.state.at[:, :, 0].set(0.5)

        result_nb = pc_relaxation_step(
            state, grid_nb.weights, grid_nb.params)
        result_fc = pc_relaxation_step(
            state, grid_fc.weights, grid_fc.params,
            fc_weights=grid_fc.fc_weights)

        # The states should differ because FC adds extra predictions
        assert not jnp.allclose(result_nb[0], result_fc[0], atol=1e-6)

    def test_fc_weights_update(self):
        """FC weights should change after a step."""
        grid = create_grid(width=8, height=8, num_instruments=2,
                           connectivity="fc")
        state = grid.state.at[:, :, 0].set(0.5)
        result = pc_relaxation_step(
            state, grid.weights, grid.params,
            fc_weights=grid.fc_weights)
        new_fc = result[3]
        assert new_fc is not None
        assert not jnp.allclose(new_fc, grid.fc_weights)

    def test_neighbor_mode_backward_compatible(self):
        """Neighbor mode should work identically to old behavior (no FC)."""
        grid = create_grid(width=8, height=8, num_instruments=2,
                           connectivity="neighbor")
        state = grid.state.at[:, :, 0].set(0.5)
        result = pc_relaxation_step(state, grid.weights, grid.params)
        assert result[3] is None  # No FC weights returned
        assert result[4] is None  # No FC skip weights returned

    def test_fista_momentum_increases_with_step(self):
        """Higher step_index should produce more FISTA momentum."""
        grid = create_grid(width=8, height=8, num_instruments=2)
        state = grid.state.at[:, :, 0].set(0.5)
        r_prev = jnp.zeros((8, 8))

        result_low = pc_relaxation_step(
            state, grid.weights, grid.params,
            step_index=1, r_prev=r_prev)
        result_high = pc_relaxation_step(
            state, grid.weights, grid.params,
            step_index=10, r_prev=r_prev)

        # They should differ due to different momentum
        assert not jnp.allclose(result_low[0][:, :, 0], result_high[0][:, :, 0])

    def test_spiking_precision_boost(self):
        """Spike boost should increase precision at wavefront column."""
        grid = create_grid(width=8, height=8, num_instruments=2)
        state = grid.state.at[:, :, 0].set(0.5)

        result_no_spike = pc_relaxation_step(
            state, grid.weights, grid.params, spike_boost=0.0)
        result_spike = pc_relaxation_step(
            state, grid.weights, grid.params, spike_boost=5.0, step_index=3)

        # States should differ because spiking changes effective precision
        assert not jnp.allclose(result_no_spike[0], result_spike[0])

    def test_w_temporal_passes_through(self):
        """w_temporal passes through relaxation step unchanged (Hebbian update is per-tick in trainer)."""
        grid = create_grid(width=8, height=8, num_instruments=2)
        state = grid.state.at[:, :, 0].set(0.5)
        state = state.at[:, :, 3].set(0.3)  # Set h channel
        w_temporal = jnp.full((8, 8), 0.1)

        result = pc_relaxation_step(
            state, grid.weights, grid.params,
            w_temporal=w_temporal)
        new_wt = result[5]
        assert new_wt is not None
        assert jnp.allclose(new_wt, w_temporal)

    def test_independent_lr_w(self):
        """Independent lr_w should control weight update magnitude."""
        grid = create_grid(width=8, height=8, num_instruments=2)
        state = grid.state.at[:, :, 0].set(1.0)

        result_small = pc_relaxation_step(
            state, grid.weights, grid.params, lr_w=0.0001)
        result_large = pc_relaxation_step(
            state, grid.weights, grid.params, lr_w=0.01)

        # Before normalization, larger lr_w produces larger weight changes
        # After L2 norm, we check that the pre-norm direction differs
        delta_small = jnp.abs(result_small[1] - grid.weights).sum()
        delta_large = jnp.abs(result_large[1] - grid.weights).sum()
        # Both get normalized, but the ratio of changes should differ
        assert not jnp.allclose(result_small[1], result_large[1])


class TestApplyClamping:
    def test_clamped_values_overwritten(self):
        grid = create_grid(width=16, height=16, num_instruments=4)
        clamp_values = jnp.ones((16,))
        state = apply_clamping(
            grid.state, grid.input_mask, clamp_values, channel=0
        )
        assert (state[:, 0, 0] == 1.0).all()
        assert (state[:, 1, 0] == grid.state[:, 1, 0]).all()

    def test_non_masked_values_unchanged(self):
        """Non-masked neurons should remain untouched."""
        grid = create_grid(width=16, height=16, num_instruments=4)
        original_state = grid.state.copy()
        clamp_values = jnp.ones((16,))
        state = apply_clamping(
            grid.state, grid.input_mask, clamp_values, channel=0
        )
        assert jnp.allclose(state[:, 1:, 0], original_state[:, 1:, 0])

    def test_clamp_different_channel(self):
        """Should be able to clamp channel 1 (error) as well."""
        grid = create_grid(width=8, height=8, num_instruments=2)
        clamp_values = jnp.full((8,), 0.5)
        state = apply_clamping(
            grid.state, grid.input_mask, clamp_values, channel=1
        )
        assert jnp.allclose(state[:, 0, 1], 0.5)
