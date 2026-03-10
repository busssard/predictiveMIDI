import jax.numpy as jnp
from training.engine.hierarchical_grid import HierarchicalGridState, create_hierarchical_grid


class TestCreateHierarchicalGrid:
    def test_default_layer_sizes(self):
        grid = create_hierarchical_grid()
        assert grid.layer_sizes == [128, 64, 32, 64, 128]

    def test_representations_shapes(self):
        grid = create_hierarchical_grid()
        assert grid.representations[0].shape == (128,)
        assert grid.representations[1].shape == (64,)
        assert grid.representations[2].shape == (32,)
        assert grid.representations[3].shape == (64,)
        assert grid.representations[4].shape == (128,)

    def test_prediction_weight_shapes(self):
        """W_pred[i] predicts layer i from layer i+1."""
        grid = create_hierarchical_grid()
        # W_pred[0]: layer 1 (64) predicts layer 0 (128) -> shape (128, 64)
        assert grid.prediction_weights[0].shape == (128, 64)
        assert grid.prediction_weights[1].shape == (64, 32)
        assert grid.prediction_weights[2].shape == (32, 64)
        assert grid.prediction_weights[3].shape == (64, 128)

    def test_skip_connection_shapes(self):
        """Skip from encoder layer i to decoder layer (N-1-i)."""
        grid = create_hierarchical_grid()
        # Skip 0->4: both 128, so W_skip shape (128, 128)
        assert grid.skip_weights[0].shape == (128, 128)
        # Skip 1->3: both 64, so W_skip shape (64, 64)
        assert grid.skip_weights[1].shape == (64, 64)

    def test_custom_layer_sizes(self):
        grid = create_hierarchical_grid(layer_sizes=[32, 16, 32])
        assert grid.layer_sizes == [32, 16, 32]
        assert len(grid.prediction_weights) == 2

    def test_conditioning_vector_size(self):
        grid = create_hierarchical_grid(num_instruments=16)
        assert grid.conditioning_size == 16

    def test_temporal_weights_shapes(self):
        grid = create_hierarchical_grid()
        for i, size in enumerate(grid.layer_sizes):
            assert grid.temporal_weights[i].shape == (size, size)

    def test_all_arrays_are_finite(self):
        grid = create_hierarchical_grid()
        for r in grid.representations:
            assert jnp.all(jnp.isfinite(r))
        for w in grid.prediction_weights:
            assert jnp.all(jnp.isfinite(w))

    def test_errors_initialized_to_zero(self):
        grid = create_hierarchical_grid()
        for e in grid.errors:
            assert jnp.allclose(e, 0.0)

    def test_temporal_state_initialized_to_zero(self):
        grid = create_hierarchical_grid()
        for ts in grid.temporal_state:
            assert jnp.allclose(ts, 0.0)
