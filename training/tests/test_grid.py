import jax.numpy as jnp
import pytest
from training.engine.grid import GridState, create_grid


class TestCreateGrid:
    def test_creates_correct_shapes(self):
        grid = create_grid(width=128, height=128, num_instruments=8)
        assert grid.state.shape == (128, 128, 4)      # r, e, s, h
        assert grid.weights.shape == (128, 128, 4)     # w_left, w_right, w_up, w_down
        assert grid.params.shape == (128, 128, 4)      # alpha, beta, lr, bias

    def test_initial_representation_near_zero(self):
        grid = create_grid(width=128, height=128, num_instruments=8)
        r = grid.state[:, :, 0]
        assert jnp.abs(r).max() < 0.1

    def test_initial_weights_small(self):
        grid = create_grid(width=128, height=128, num_instruments=8)
        assert jnp.abs(grid.weights).max() < 1.0

    def test_alpha_initialized_in_range(self):
        grid = create_grid(width=128, height=128, num_instruments=8)
        alpha = grid.params[:, :, 0]
        assert (alpha >= 0.0).all()
        assert (alpha <= 1.0).all()

    def test_custom_size(self):
        grid = create_grid(width=32, height=32, num_instruments=4)
        assert grid.state.shape == (32, 32, 4)

    def test_clamp_masks(self):
        grid = create_grid(width=128, height=128, num_instruments=8)
        assert grid.input_mask[0, 0] == True
        assert grid.input_mask[64, 0] == True
        assert grid.input_mask[0, 1] == False
        assert grid.output_mask[0, 127] == True
        assert grid.output_mask[0, 0] == False
        assert grid.conditioning_mask.sum() == 8

    def test_conditioning_mask_centered(self):
        """Conditioning mask should be centered vertically."""
        grid = create_grid(width=128, height=128, num_instruments=8)
        cond_start = (128 - 8) // 2
        for i in range(8):
            assert grid.conditioning_mask[cond_start + i, 0] == True
        assert grid.conditioning_mask[cond_start - 1, 0] == False

    def test_beta_initialized(self):
        grid = create_grid(width=32, height=32, num_instruments=4)
        beta = grid.params[:, :, 1]
        assert jnp.allclose(beta, 0.1)

    def test_learning_rate_initialized(self):
        grid = create_grid(width=32, height=32, num_instruments=4)
        lr = grid.params[:, :, 2]
        assert jnp.allclose(lr, 0.005)

    def test_bias_initialized_to_zero(self):
        grid = create_grid(width=32, height=32, num_instruments=4)
        bias = grid.params[:, :, 3]
        assert jnp.allclose(bias, 0.0)

    def test_log_precision_shape(self):
        grid = create_grid(width=32, height=32, num_instruments=4)
        assert grid.log_precision.shape == (32, 32)

    def test_log_precision_initialized_to_zero(self):
        grid = create_grid(width=32, height=32, num_instruments=4)
        assert jnp.allclose(grid.log_precision, 0.0)

    def test_pos_weight_default(self):
        grid = create_grid(width=32, height=32, num_instruments=4)
        assert grid.pos_weight == 20.0

    def test_lambda_sparse_default(self):
        grid = create_grid(width=32, height=32, num_instruments=4)
        assert grid.lambda_sparse == 0.01

    def test_default_width_is_16(self):
        grid = create_grid(height=128, num_instruments=8)
        assert grid.state.shape == (128, 16, 4)

    def test_connectivity_neighbor_no_fc(self):
        grid = create_grid(width=8, height=8, num_instruments=2,
                           connectivity="neighbor")
        assert grid.connectivity == "neighbor"
        assert grid.fc_weights is None
        assert grid.fc_skip_weights is None

    def test_connectivity_fc_creates_weights(self):
        grid = create_grid(width=8, height=8, num_instruments=2,
                           connectivity="fc")
        assert grid.connectivity == "fc"
        assert grid.fc_weights is not None
        assert grid.fc_weights.shape == (8, 8, 8)
        assert grid.fc_skip_weights is None

    def test_connectivity_fc_double_creates_both(self):
        grid = create_grid(width=8, height=8, num_instruments=2,
                           connectivity="fc_double")
        assert grid.connectivity == "fc_double"
        assert grid.fc_weights is not None
        assert grid.fc_skip_weights is not None
        assert grid.fc_skip_weights.shape == (8, 8, 8)

    def test_w_temporal_initialized_to_beta(self):
        grid = create_grid(width=8, height=8, num_instruments=2)
        assert grid.w_temporal is not None
        assert grid.w_temporal.shape == (8, 8)
        assert jnp.allclose(grid.w_temporal, 0.1)

    def test_default_lr_is_0005(self):
        grid = create_grid(width=8, height=8, num_instruments=2)
        lr = grid.params[:, :, 2]
        assert jnp.allclose(lr, 0.005)

    def test_lr_w_default(self):
        grid = create_grid(width=8, height=8, num_instruments=2)
        assert grid.lr_w == 0.001

    def test_spike_boost_default(self):
        grid = create_grid(width=8, height=8, num_instruments=2)
        assert grid.spike_boost == 5.0

    def test_asl_defaults(self):
        grid = create_grid(width=8, height=8, num_instruments=2)
        assert grid.asl_gamma_neg == 4.0
        assert grid.asl_margin == 0.05

    def test_lr_amplification_center_boost(self):
        """lr_amplification > 0 should boost center columns more."""
        grid = create_grid(width=8, height=8, num_instruments=2,
                           lr_amplification=1.0)
        lr = grid.params[:, :, 2]
        # Center column should have higher lr than edge columns
        assert float(lr[0, 4]) > float(lr[0, 0])
