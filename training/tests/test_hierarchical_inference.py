import numpy as np
import jax.numpy as jnp
from training.engine.hierarchical_grid import create_hierarchical_grid
from training.engine.hierarchical_inference import run_hierarchical_inference


class TestHierarchicalInference:
    def test_output_shape(self):
        grid = create_hierarchical_grid(layer_sizes=[128, 64, 32, 64, 128])
        input_seq = np.random.rand(8, 128).astype(np.float32)
        cond = np.zeros(16, dtype=np.float32)
        cond[0] = 1.0

        output, _ = run_hierarchical_inference(
            grid, input_seq, cond, relaxation_steps=5)
        assert output.shape == (8, 128)

    def test_output_in_valid_range(self):
        grid = create_hierarchical_grid(layer_sizes=[128, 64, 32, 64, 128])
        input_seq = np.random.rand(8, 128).astype(np.float32)
        cond = np.zeros(16, dtype=np.float32)
        cond[0] = 1.0

        output, _ = run_hierarchical_inference(
            grid, input_seq, cond, relaxation_steps=5)
        assert np.all(output >= 0.0)
        assert np.all(output <= 1.0)

    def test_no_nan_divergence(self):
        """The critical test -- hierarchical inference must not NaN."""
        grid = create_hierarchical_grid(layer_sizes=[128, 64, 32, 64, 128])
        input_seq = np.zeros((16, 128), dtype=np.float32)
        input_seq[:, 60] = 1.0  # single note
        cond = np.zeros(16, dtype=np.float32)

        output, _ = run_hierarchical_inference(
            grid, input_seq, cond, relaxation_steps=32)
        assert np.all(np.isfinite(output))

    def test_layer_states_returned(self):
        grid = create_hierarchical_grid(layer_sizes=[128, 64, 32, 64, 128])
        input_seq = np.random.rand(4, 128).astype(np.float32)
        cond = np.zeros(16, dtype=np.float32)

        _, all_states = run_hierarchical_inference(
            grid, input_seq, cond, relaxation_steps=5)
        assert len(all_states) == 5
        assert all_states[0].shape == (4, 128)
        assert all_states[2].shape == (4, 32)

    def test_small_grid(self):
        """Test with smaller grid for speed."""
        grid = create_hierarchical_grid(layer_sizes=[32, 16, 32])
        input_seq = np.random.rand(4, 32).astype(np.float32)
        cond = np.zeros(16, dtype=np.float32)
        cond[0] = 1.0

        output, _ = run_hierarchical_inference(
            grid, input_seq, cond, relaxation_steps=5)
        assert output.shape == (4, 128)  # always outputs 128 (zero-padded)
        assert np.all(np.isfinite(output))
