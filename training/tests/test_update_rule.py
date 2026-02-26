import jax
import jax.numpy as jnp
import pytest
from training.engine.grid import create_grid
from training.engine.update_rule import pc_relaxation_step, apply_clamping


class TestPCRelaxationStep:
    def test_output_same_shape_as_input(self):
        grid = create_grid(size=16, num_instruments=4)
        new_state, new_weights = pc_relaxation_step(
            grid.state, grid.weights, grid.params
        )
        assert new_state.shape == grid.state.shape
        assert new_weights.shape == grid.weights.shape

    def test_error_stays_finite_over_steps(self):
        """After multiple relaxation steps, error should remain finite (not explode)."""
        key = jax.random.PRNGKey(42)
        grid = create_grid(size=16, num_instruments=4, key=key)
        # Set some non-trivial initial state
        state = grid.state.at[:, :, 0].set(
            jax.random.normal(key, (16, 16)) * 0.5
        )

        # Run 20 relaxation steps
        weights = grid.weights
        for _ in range(20):
            state, weights = pc_relaxation_step(state, weights, grid.params)

        final_error = jnp.abs(state[:, :, 1]).sum()
        assert jnp.isfinite(final_error)
        # Error should not explode — stay under a reasonable bound
        assert final_error < 1000.0

    def test_leaky_state_persists(self):
        """The leaky integrator should carry forward previous state."""
        grid = create_grid(size=8, num_instruments=2)
        # Set representation to 1.0 everywhere
        state = grid.state.at[:, :, 0].set(1.0)
        new_state, _ = pc_relaxation_step(state, grid.weights, grid.params)
        # Leaky state (channel 2) should be nonzero after one step
        s = new_state[:, :, 2]
        assert jnp.abs(s).sum() > 0

    def test_is_jit_compilable(self):
        grid = create_grid(size=8, num_instruments=2)
        jitted = jax.jit(pc_relaxation_step)
        new_state, new_weights = jitted(grid.state, grid.weights, grid.params)
        assert new_state.shape == grid.state.shape

    def test_weights_change_after_step(self):
        """iPC weight update should modify weights."""
        grid = create_grid(size=8, num_instruments=2)
        # Set nonzero representation so weight update is nonzero
        state = grid.state.at[:, :, 0].set(1.0)
        _, new_weights = pc_relaxation_step(state, grid.weights, grid.params)
        # Weights should have changed
        assert not jnp.allclose(new_weights, grid.weights)

    def test_state_channels_are_finite(self):
        """All channels should remain finite after a step."""
        grid = create_grid(size=16, num_instruments=4)
        new_state, new_weights = pc_relaxation_step(
            grid.state, grid.weights, grid.params
        )
        assert jnp.isfinite(new_state).all()
        assert jnp.isfinite(new_weights).all()


class TestApplyClamping:
    def test_clamped_values_overwritten(self):
        grid = create_grid(size=16, num_instruments=4)
        clamp_values = jnp.ones((16,))  # all 1s for input column
        state = apply_clamping(
            grid.state, grid.input_mask, clamp_values, channel=0
        )
        # Column 0 representation should be 1.0
        assert (state[:, 0, 0] == 1.0).all()
        # Column 1 should be unchanged
        assert (state[:, 1, 0] == grid.state[:, 1, 0]).all()

    def test_non_masked_values_unchanged(self):
        """Non-masked neurons should remain untouched."""
        grid = create_grid(size=16, num_instruments=4)
        original_state = grid.state.copy()
        clamp_values = jnp.ones((16,))
        state = apply_clamping(
            grid.state, grid.input_mask, clamp_values, channel=0
        )
        # Columns 1-15 should be identical
        assert jnp.allclose(state[:, 1:, 0], original_state[:, 1:, 0])

    def test_clamp_different_channel(self):
        """Should be able to clamp channel 1 (error) as well."""
        grid = create_grid(size=8, num_instruments=2)
        clamp_values = jnp.full((8,), 0.5)
        state = apply_clamping(
            grid.state, grid.input_mask, clamp_values, channel=1
        )
        # Column 0 error channel should be 0.5
        assert jnp.allclose(state[:, 0, 1], 0.5)
