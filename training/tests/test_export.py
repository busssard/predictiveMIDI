import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from training.engine.grid import create_grid
from training.engine.export import export_model, load_exported_model


class TestExportModel:
    def test_creates_binary_files(self, tmp_path):
        grid = create_grid(size=16, num_instruments=4)
        vocabulary = {"piano": 0, "bass": 1, "guitar": 2, "drums": 3}
        export_model(grid, vocabulary, str(tmp_path))
        assert (tmp_path / "state.bin").exists()
        assert (tmp_path / "weights.bin").exists()
        assert (tmp_path / "params.bin").exists()
        assert (tmp_path / "config.json").exists()

    def test_binary_file_sizes_correct(self, tmp_path):
        grid = create_grid(size=16, num_instruments=4)
        vocabulary = {"piano": 0, "bass": 1, "guitar": 2, "drums": 3}
        export_model(grid, vocabulary, str(tmp_path))
        # Each texture: 16 * 16 * 4 channels * 4 bytes = 4096 bytes
        expected = 16 * 16 * 4 * 4
        assert (tmp_path / "state.bin").stat().st_size == expected
        assert (tmp_path / "weights.bin").stat().st_size == expected
        assert (tmp_path / "params.bin").stat().st_size == expected

    def test_config_json_correct(self, tmp_path):
        grid = create_grid(size=16, num_instruments=4)
        vocabulary = {"piano": 0, "bass": 1, "guitar": 2, "drums": 3}
        export_model(grid, vocabulary, str(tmp_path))
        config = json.loads((tmp_path / "config.json").read_text())
        assert config["grid_size"] == 16
        assert config["num_instruments"] == 4
        assert config["vocabulary"] == vocabulary

    def test_roundtrip_preserves_values(self, tmp_path):
        grid = create_grid(size=16, num_instruments=4)
        vocabulary = {"piano": 0, "bass": 1, "guitar": 2, "drums": 3}
        export_model(grid, vocabulary, str(tmp_path))
        loaded = load_exported_model(str(tmp_path))
        assert np.allclose(loaded["state"], np.array(grid.state), atol=1e-6)
        assert np.allclose(loaded["weights"], np.array(grid.weights), atol=1e-6)

    def test_creates_output_directory(self, tmp_path):
        """export_model should create the output directory if needed."""
        output_dir = tmp_path / "nested" / "model"
        grid = create_grid(size=8, num_instruments=2)
        vocabulary = {"piano": 0, "drums": 1}
        export_model(grid, vocabulary, str(output_dir))
        assert (output_dir / "state.bin").exists()

    def test_config_contains_dtype(self, tmp_path):
        grid = create_grid(size=8, num_instruments=2)
        vocabulary = {"piano": 0, "drums": 1}
        export_model(grid, vocabulary, str(tmp_path))
        config = json.loads((tmp_path / "config.json").read_text())
        assert config["dtype"] == "float32"
        assert config["channels_per_texture"] == 4


class TestLoadExportedModel:
    def test_loaded_shapes_correct(self, tmp_path):
        grid = create_grid(size=16, num_instruments=4)
        vocabulary = {"piano": 0, "bass": 1, "guitar": 2, "drums": 3}
        export_model(grid, vocabulary, str(tmp_path))
        loaded = load_exported_model(str(tmp_path))
        assert loaded["state"].shape == (16, 16, 4)
        assert loaded["weights"].shape == (16, 16, 4)
        assert loaded["params"].shape == (16, 16, 4)

    def test_loaded_config_present(self, tmp_path):
        grid = create_grid(size=8, num_instruments=2)
        vocabulary = {"piano": 0, "drums": 1}
        export_model(grid, vocabulary, str(tmp_path))
        loaded = load_exported_model(str(tmp_path))
        assert "config" in loaded
        assert loaded["config"]["grid_size"] == 8
