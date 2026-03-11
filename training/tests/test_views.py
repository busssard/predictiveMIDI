"""Tests for training views: checkpoint listing and export with hierarchical support."""
import json
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from django.test import RequestFactory
from rest_framework.test import APIRequestFactory

from training.views import CheckpointListView, ExportCheckpointView
from training.engine.export import export_hierarchical_model


@pytest.fixture
def factory():
    return APIRequestFactory()


@pytest.fixture
def flat_checkpoint(tmp_path):
    """Create a mock flat checkpoint directory."""
    ckpt_dir = tmp_path / "checkpoints"
    step = ckpt_dir / "step_10"
    step.mkdir(parents=True)
    # Flat checkpoints have state.npy
    np.save(step / "state.npy", np.zeros((128, 16, 4), dtype=np.float32))
    np.save(step / "weights.npy", np.zeros((128, 16, 4), dtype=np.float32))
    np.save(step / "params.npy", np.zeros((128, 16, 4), dtype=np.float32))
    return ckpt_dir


@pytest.fixture
def hierarchical_checkpoint(tmp_path):
    """Create a mock hierarchical checkpoint directory."""
    ckpt_dir = tmp_path / "checkpoints_hierarchical"
    step = ckpt_dir / "step_50"
    step.mkdir(parents=True)
    final = ckpt_dir / "final"
    final.mkdir(parents=True)

    layer_sizes = [128, 64, 32, 64, 128]
    for d in [step, final]:
        metadata = {
            "architecture": "hierarchical",
            "layer_sizes": layer_sizes,
            "vocabulary": {"piano": 0, "drums": 1},
            "conditioning_size": 2,
            "lr": 0.01,
            "lr_w": 0.001,
            "alpha": 0.8,
            "lambda_sparse": 0.005,
        }
        (d / "metadata.json").write_text(json.dumps(metadata))
        # Save minimal weight files
        for i in range(len(layer_sizes)):
            np.save(d / f"layer_{i}_rep.npy", np.zeros(layer_sizes[i], dtype=np.float32))
            np.save(d / f"temporal_weight_{i}.npy",
                    np.zeros((layer_sizes[i], layer_sizes[i]), dtype=np.float32))
        for i in range(len(layer_sizes) - 1):
            np.save(d / f"pred_weight_{i}.npy",
                    np.zeros((layer_sizes[i], layer_sizes[i + 1]), dtype=np.float32))
        n_skip = len(layer_sizes) // 2
        for i in range(n_skip):
            j = len(layer_sizes) - 1 - i
            np.save(d / f"skip_weight_{i}.npy",
                    np.zeros((layer_sizes[j], layer_sizes[i]), dtype=np.float32))

    return ckpt_dir


@pytest.mark.django_db
class TestCheckpointListView:
    """Test that CheckpointListView returns both flat and hierarchical checkpoints."""

    def test_lists_flat_checkpoints(self, factory, flat_checkpoint):
        """Flat checkpoints with state.npy are listed with architecture='flat'."""
        view = CheckpointListView.as_view()
        request = factory.get("/api/training/checkpoints/")
        with patch("training.views.settings") as mock_settings:
            mock_settings.CHECKPOINT_DIR = str(flat_checkpoint)
            mock_settings.BASE_DIR = flat_checkpoint.parent
            response = view(request)
        assert response.status_code == 200
        ckpts = response.data["checkpoints"]
        assert len(ckpts) == 1
        assert ckpts[0]["name"] == "step_10"
        assert ckpts[0]["step"] == 10
        assert ckpts[0]["architecture"] == "flat"

    def test_lists_hierarchical_checkpoints(self, factory, hierarchical_checkpoint, tmp_path):
        """Hierarchical checkpoints with metadata.json are listed with architecture='hierarchical'."""
        view = CheckpointListView.as_view()
        request = factory.get("/api/training/checkpoints/")
        # The flat checkpoint dir may not exist, that's ok
        flat_dir = tmp_path / "checkpoints"
        with patch("training.views.settings") as mock_settings:
            mock_settings.CHECKPOINT_DIR = str(flat_dir)
            mock_settings.BASE_DIR = tmp_path
            response = view(request)
        assert response.status_code == 200
        ckpts = response.data["checkpoints"]
        # Should find step_50 and final
        names = {c["name"] for c in ckpts}
        assert "checkpoints_hierarchical/step_50" in names
        assert "checkpoints_hierarchical/final" in names
        for c in ckpts:
            assert c["architecture"] == "hierarchical"

    def test_lists_both_flat_and_hierarchical(self, factory, flat_checkpoint,
                                              hierarchical_checkpoint, tmp_path):
        """When both types exist, both are returned with correct architecture labels."""
        # Move both into the same parent dir
        # flat_checkpoint is already at tmp_path/checkpoints
        # hierarchical_checkpoint is at tmp_path/checkpoints_hierarchical
        view = CheckpointListView.as_view()
        request = factory.get("/api/training/checkpoints/")
        with patch("training.views.settings") as mock_settings:
            mock_settings.CHECKPOINT_DIR = str(flat_checkpoint)
            mock_settings.BASE_DIR = tmp_path
            response = view(request)
        assert response.status_code == 200
        ckpts = response.data["checkpoints"]
        archs = {c["architecture"] for c in ckpts}
        assert "flat" in archs
        assert "hierarchical" in archs
        assert len(ckpts) >= 3  # 1 flat + 2 hierarchical (step_50 + final)

    def test_hierarchical_includes_final(self, factory, hierarchical_checkpoint, tmp_path):
        """The 'final' subdirectory is also detected as a hierarchical checkpoint."""
        view = CheckpointListView.as_view()
        request = factory.get("/api/training/checkpoints/")
        with patch("training.views.settings") as mock_settings:
            mock_settings.CHECKPOINT_DIR = str(tmp_path / "checkpoints")
            mock_settings.BASE_DIR = tmp_path
            response = view(request)
        assert response.status_code == 200
        ckpts = response.data["checkpoints"]
        final_ckpts = [c for c in ckpts if c["name"].endswith("/final")]
        assert len(final_ckpts) == 1
        assert final_ckpts[0]["architecture"] == "hierarchical"

    def test_empty_checkpoint_dirs(self, factory, tmp_path):
        """No checkpoints returns empty list without error."""
        view = CheckpointListView.as_view()
        request = factory.get("/api/training/checkpoints/")
        with patch("training.views.settings") as mock_settings:
            mock_settings.CHECKPOINT_DIR = str(tmp_path / "nonexistent")
            mock_settings.BASE_DIR = tmp_path
            response = view(request)
        assert response.status_code == 200
        assert response.data["checkpoints"] == []

    def test_hierarchical_checkpoint_has_layer_sizes(self, factory, hierarchical_checkpoint,
                                                     tmp_path):
        """Hierarchical checkpoints include layer_sizes metadata."""
        view = CheckpointListView.as_view()
        request = factory.get("/api/training/checkpoints/")
        with patch("training.views.settings") as mock_settings:
            mock_settings.CHECKPOINT_DIR = str(tmp_path / "checkpoints")
            mock_settings.BASE_DIR = tmp_path
            response = view(request)
        assert response.status_code == 200
        ckpts = response.data["checkpoints"]
        hier_ckpts = [c for c in ckpts if c["architecture"] == "hierarchical"]
        assert len(hier_ckpts) > 0
        for c in hier_ckpts:
            assert c["layer_sizes"] == [128, 64, 32, 64, 128]


class TestExportHierarchicalModel:
    """Test the export_hierarchical_model function."""

    def test_creates_weight_files(self, hierarchical_checkpoint, tmp_path):
        """Export creates .bin files for each weight array."""
        ckpt_path = hierarchical_checkpoint / "step_50"
        output_dir = tmp_path / "export_out"
        export_hierarchical_model(str(ckpt_path), str(output_dir))
        assert (output_dir / "config.json").exists()
        config = json.loads((output_dir / "config.json").read_text())
        assert config["architecture"] == "hierarchical"
        assert config["layer_sizes"] == [128, 64, 32, 64, 128]
        # Check weight files exist
        for i in range(4):  # 5 layers - 1 = 4 prediction weights
            assert (output_dir / f"pred_weight_{i}.bin").exists()
        for i in range(2):  # 5 // 2 = 2 skip weights
            assert (output_dir / f"skip_weight_{i}.bin").exists()
        for i in range(5):  # 5 temporal weights
            assert (output_dir / f"temporal_weight_{i}.bin").exists()

    def test_config_has_vocabulary(self, hierarchical_checkpoint, tmp_path):
        """Exported config.json contains the vocabulary from metadata."""
        ckpt_path = hierarchical_checkpoint / "step_50"
        output_dir = tmp_path / "export_out"
        export_hierarchical_model(str(ckpt_path), str(output_dir))
        config = json.loads((output_dir / "config.json").read_text())
        assert config["vocabulary"] == {"piano": 0, "drums": 1}

    def test_weight_file_sizes(self, hierarchical_checkpoint, tmp_path):
        """Weight .bin files have correct byte sizes (float32 = 4 bytes per element)."""
        ckpt_path = hierarchical_checkpoint / "step_50"
        output_dir = tmp_path / "export_out"
        export_hierarchical_model(str(ckpt_path), str(output_dir))
        # pred_weight_0: (128, 64) => 128*64*4 bytes
        assert (output_dir / "pred_weight_0.bin").stat().st_size == 128 * 64 * 4
        # skip_weight_0: (128, 128) => 128*128*4 bytes
        assert (output_dir / "skip_weight_0.bin").stat().st_size == 128 * 128 * 4
        # temporal_weight_0: (128, 128) => 128*128*4 bytes
        assert (output_dir / "temporal_weight_0.bin").stat().st_size == 128 * 128 * 4

    def test_creates_output_directory(self, hierarchical_checkpoint, tmp_path):
        """export_hierarchical_model creates nested output dirs if needed."""
        ckpt_path = hierarchical_checkpoint / "step_50"
        output_dir = tmp_path / "nested" / "deep" / "export"
        export_hierarchical_model(str(ckpt_path), str(output_dir))
        assert (output_dir / "config.json").exists()


@pytest.mark.django_db
class TestExportCheckpointView:
    """Test ExportCheckpointView handles both flat and hierarchical checkpoints."""

    def test_export_flat_checkpoint(self, factory, flat_checkpoint, tmp_path):
        """Exporting a flat checkpoint works as before."""
        view = ExportCheckpointView.as_view()
        request = factory.post(
            "/api/training/export-checkpoint/",
            {"checkpoint": "step_10", "name": "test_flat"},
            format="json",
        )
        model_dir = tmp_path / "frontend" / "model"
        with patch("training.views.settings") as mock_settings:
            mock_settings.CHECKPOINT_DIR = str(flat_checkpoint)
            mock_settings.BASE_DIR = tmp_path
            response = view(request)
        assert response.status_code == 200
        assert response.data["status"] == "exported"

    def test_export_hierarchical_checkpoint(self, factory, hierarchical_checkpoint, tmp_path):
        """Exporting a hierarchical checkpoint uses the hierarchical export path."""
        view = ExportCheckpointView.as_view()
        request = factory.post(
            "/api/training/export-checkpoint/",
            {"checkpoint": "checkpoints_hierarchical/step_50", "name": "test_hier"},
            format="json",
        )
        with patch("training.views.settings") as mock_settings:
            mock_settings.CHECKPOINT_DIR = str(tmp_path / "checkpoints")
            mock_settings.BASE_DIR = tmp_path
            response = view(request)
        assert response.status_code == 200
        assert response.data["status"] == "exported"
        # Verify hierarchical export files exist
        export_dir = tmp_path / "frontend" / "model" / "test_hier"
        assert (export_dir / "config.json").exists()
        config = json.loads((export_dir / "config.json").read_text())
        assert config["architecture"] == "hierarchical"

    def test_export_missing_checkpoint_returns_404(self, factory, tmp_path):
        """Requesting a non-existent checkpoint returns 404."""
        view = ExportCheckpointView.as_view()
        request = factory.post(
            "/api/training/export-checkpoint/",
            {"checkpoint": "step_999", "name": "nope"},
            format="json",
        )
        with patch("training.views.settings") as mock_settings:
            mock_settings.CHECKPOINT_DIR = str(tmp_path / "checkpoints")
            mock_settings.BASE_DIR = tmp_path
            response = view(request)
        assert response.status_code == 404
