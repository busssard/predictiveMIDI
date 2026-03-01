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
        grid = create_grid(width=16, height=16, num_instruments=4)
        vocabulary = {"piano": 0, "bass": 1, "guitar": 2, "drums": 3}
        export_model(grid, vocabulary, str(tmp_path))
        assert (tmp_path / "state.bin").exists()
        assert (tmp_path / "weights.bin").exists()
        assert (tmp_path / "params.bin").exists()
        assert (tmp_path / "config.json").exists()

    def test_binary_file_sizes_correct(self, tmp_path):
        grid = create_grid(width=16, height=16, num_instruments=4)
        vocabulary = {"piano": 0, "bass": 1, "guitar": 2, "drums": 3}
        export_model(grid, vocabulary, str(tmp_path))
        expected = 16 * 16 * 4 * 4
        assert (tmp_path / "state.bin").stat().st_size == expected
        assert (tmp_path / "weights.bin").stat().st_size == expected
        assert (tmp_path / "params.bin").stat().st_size == expected

    def test_config_json_correct(self, tmp_path):
        grid = create_grid(width=16, height=16, num_instruments=4)
        vocabulary = {"piano": 0, "bass": 1, "guitar": 2, "drums": 3}
        export_model(grid, vocabulary, str(tmp_path))
        config = json.loads((tmp_path / "config.json").read_text())
        assert config["grid_size"] == 16
        assert config["num_instruments"] == 4
        assert config["vocabulary"] == vocabulary

    def test_roundtrip_preserves_values(self, tmp_path):
        grid = create_grid(width=16, height=16, num_instruments=4)
        vocabulary = {"piano": 0, "bass": 1, "guitar": 2, "drums": 3}
        export_model(grid, vocabulary, str(tmp_path))
        loaded = load_exported_model(str(tmp_path))
        assert np.allclose(loaded["state"], np.array(grid.state), atol=1e-6)
        assert np.allclose(loaded["weights"], np.array(grid.weights), atol=1e-6)

    def test_creates_output_directory(self, tmp_path):
        """export_model should create the output directory if needed."""
        output_dir = tmp_path / "nested" / "model"
        grid = create_grid(width=8, height=8, num_instruments=2)
        vocabulary = {"piano": 0, "drums": 1}
        export_model(grid, vocabulary, str(output_dir))
        assert (output_dir / "state.bin").exists()

    def test_config_contains_dtype(self, tmp_path):
        grid = create_grid(width=8, height=8, num_instruments=2)
        vocabulary = {"piano": 0, "drums": 1}
        export_model(grid, vocabulary, str(tmp_path))
        config = json.loads((tmp_path / "config.json").read_text())
        assert config["dtype"] == "float32"
        assert config["channels_per_texture"] == 4

    def test_log_precision_exported(self, tmp_path):
        """export_model should write log_precision.bin when present."""
        grid = create_grid(width=16, height=16, num_instruments=4)
        vocabulary = {"piano": 0, "bass": 1, "guitar": 2, "drums": 3}
        export_model(grid, vocabulary, str(tmp_path))
        assert (tmp_path / "log_precision.bin").exists()
        assert (tmp_path / "log_precision.bin").stat().st_size == 16 * 16 * 4

    def test_log_precision_roundtrip(self, tmp_path):
        """log_precision should survive export/load roundtrip."""
        grid = create_grid(width=8, height=8, num_instruments=2)
        vocabulary = {"piano": 0, "drums": 1}
        export_model(grid, vocabulary, str(tmp_path))
        loaded = load_exported_model(str(tmp_path))
        assert "log_precision" in loaded
        assert np.allclose(loaded["log_precision"],
                           np.array(grid.log_precision), atol=1e-6)

    def test_fc_weights_exported(self, tmp_path):
        """FC weights should be exported when connectivity=fc."""
        grid = create_grid(width=8, height=8, num_instruments=2,
                           connectivity="fc")
        vocabulary = {"piano": 0, "drums": 1}
        export_model(grid, vocabulary, str(tmp_path))
        assert (tmp_path / "fc_weights.bin").exists()
        # W * H * H * 4 bytes
        assert (tmp_path / "fc_weights.bin").stat().st_size == 8 * 8 * 8 * 4

    def test_fc_weights_roundtrip(self, tmp_path):
        """FC weights should survive export/load roundtrip."""
        grid = create_grid(width=8, height=8, num_instruments=2,
                           connectivity="fc")
        vocabulary = {"piano": 0, "drums": 1}
        export_model(grid, vocabulary, str(tmp_path))
        loaded = load_exported_model(str(tmp_path))
        assert "fc_weights" in loaded
        assert np.allclose(loaded["fc_weights"],
                           np.array(grid.fc_weights), atol=1e-6)

    def test_fc_skip_weights_roundtrip(self, tmp_path):
        """FC skip weights should survive export/load roundtrip."""
        grid = create_grid(width=8, height=8, num_instruments=2,
                           connectivity="fc_double")
        vocabulary = {"piano": 0, "drums": 1}
        export_model(grid, vocabulary, str(tmp_path))
        loaded = load_exported_model(str(tmp_path))
        assert "fc_weights" in loaded
        assert "fc_skip_weights" in loaded
        assert np.allclose(loaded["fc_skip_weights"],
                           np.array(grid.fc_skip_weights), atol=1e-6)

    def test_w_temporal_exported(self, tmp_path):
        """w_temporal should be exported."""
        grid = create_grid(width=8, height=8, num_instruments=2)
        vocabulary = {"piano": 0, "drums": 1}
        export_model(grid, vocabulary, str(tmp_path))
        assert (tmp_path / "w_temporal.bin").exists()
        assert (tmp_path / "w_temporal.bin").stat().st_size == 8 * 8 * 4

    def test_w_temporal_roundtrip(self, tmp_path):
        """w_temporal should survive export/load roundtrip."""
        grid = create_grid(width=8, height=8, num_instruments=2)
        vocabulary = {"piano": 0, "drums": 1}
        export_model(grid, vocabulary, str(tmp_path))
        loaded = load_exported_model(str(tmp_path))
        assert "w_temporal" in loaded
        assert np.allclose(loaded["w_temporal"],
                           np.array(grid.w_temporal), atol=1e-6)

    def test_config_contains_connectivity(self, tmp_path):
        """config.json should include connectivity field."""
        grid = create_grid(width=8, height=8, num_instruments=2,
                           connectivity="fc")
        vocabulary = {"piano": 0, "drums": 1}
        export_model(grid, vocabulary, str(tmp_path))
        config = json.loads((tmp_path / "config.json").read_text())
        assert config["connectivity"] == "fc"


class TestLoadExportedModel:
    def test_loaded_shapes_correct(self, tmp_path):
        grid = create_grid(width=16, height=16, num_instruments=4)
        vocabulary = {"piano": 0, "bass": 1, "guitar": 2, "drums": 3}
        export_model(grid, vocabulary, str(tmp_path))
        loaded = load_exported_model(str(tmp_path))
        assert loaded["state"].shape == (16, 16, 4)
        assert loaded["weights"].shape == (16, 16, 4)
        assert loaded["params"].shape == (16, 16, 4)

    def test_loaded_config_present(self, tmp_path):
        grid = create_grid(width=8, height=8, num_instruments=2)
        vocabulary = {"piano": 0, "drums": 1}
        export_model(grid, vocabulary, str(tmp_path))
        loaded = load_exported_model(str(tmp_path))
        assert "config" in loaded
        assert loaded["config"]["grid_size"] == 8
