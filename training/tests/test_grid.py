import jax.numpy as jnp
import pytest
from training.engine.grid import GridState, create_grid


class TestCreateGrid:
    def test_creates_correct_shapes(self):
        grid = create_grid(size=128, num_instruments=8)
        assert grid.state.shape == (128, 128, 4)      # r, e, s, h
        assert grid.weights.shape == (128, 128, 4)     # w_left, w_right, w_up, w_down
        assert grid.params.shape == (128, 128, 4)      # alpha, beta, lr, bias

    def test_initial_representation_near_zero(self):
        grid = create_grid(size=128, num_instruments=8)
        r = grid.state[:, :, 0]  # representation channel
        assert jnp.abs(r).max() < 0.1

    def test_initial_weights_small(self):
        grid = create_grid(size=128, num_instruments=8)
        assert jnp.abs(grid.weights).max() < 1.0

    def test_alpha_initialized_in_range(self):
        grid = create_grid(size=128, num_instruments=8)
        alpha = grid.params[:, :, 0]
        assert (alpha >= 0.0).all()
        assert (alpha <= 1.0).all()

    def test_custom_size(self):
        grid = create_grid(size=32, num_instruments=4)
        assert grid.state.shape == (32, 32, 4)

    def test_clamp_masks(self):
        grid = create_grid(size=128, num_instruments=8)
        # Input mask: column 0
        assert grid.input_mask[0, 0] == True
        assert grid.input_mask[64, 0] == True
        assert grid.input_mask[0, 1] == False
        # Output mask: column 127
        assert grid.output_mask[0, 127] == True
        assert grid.output_mask[0, 0] == False
        # Conditioning mask: top-middle of column 0
        assert grid.conditioning_mask.sum() == 8  # num_instruments

    def test_conditioning_mask_centered(self):
        """Conditioning mask should be centered vertically."""
        grid = create_grid(size=128, num_instruments=8)
        # With size=128, num_instruments=8: cond_start = (128 - 8) // 2 = 60
        # So rows 60..67 in column 0 should be True
        cond_start = (128 - 8) // 2
        for i in range(8):
            assert grid.conditioning_mask[cond_start + i, 0] == True
        # Row before should be False
        assert grid.conditioning_mask[cond_start - 1, 0] == False

    def test_beta_initialized(self):
        grid = create_grid(size=32, num_instruments=4)
        beta = grid.params[:, :, 1]
        assert jnp.allclose(beta, 0.1)

    def test_learning_rate_initialized(self):
        grid = create_grid(size=32, num_instruments=4)
        lr = grid.params[:, :, 2]
        assert jnp.allclose(lr, 0.01)

    def test_bias_initialized_to_zero(self):
        grid = create_grid(size=32, num_instruments=4)
        bias = grid.params[:, :, 3]
        assert jnp.allclose(bias, 0.0)
