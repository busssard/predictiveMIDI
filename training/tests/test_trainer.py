import json
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from corpus.tests.test_scanner import make_test_midi
from corpus.tests.test_batch_generator import make_corpus
from training.engine.trainer import Trainer


class TestTrainerMetricsIntegration:
    def test_trainer_logs_metrics_when_metrics_dir_set(self, tmp_path):
        """Trainer should write JSONL metrics when metrics_dir is provided."""
        midi_dir = tmp_path / "midi"
        midi_dir.mkdir()
        make_corpus(midi_dir)
        metrics_dir = tmp_path / "metrics"
        trainer = Trainer(
            midi_dir=str(midi_dir),
            grid_size=16,
            relaxation_steps=5,
            fs=8.0,
            metrics_dir=str(metrics_dir),
        )
        trainer.train_step(batch_size=2, step=1)
        trainer.train_step(batch_size=2, step=2)

        from training.engine.metrics_logger import MetricsLogger
        logger = MetricsLogger(str(metrics_dir))
        metrics = logger.load_metrics()
        assert len(metrics) == 2
        assert metrics[0]["step"] == 1
        assert metrics[1]["step"] == 2
        assert "error" in metrics[0]
        assert "f1" in metrics[0]
        assert "precision" in metrics[0]
        assert "recall" in metrics[0]

    def test_trainer_no_metrics_without_dir(self, tmp_path):
        """Trainer should not crash when metrics_dir is not provided."""
        make_corpus(tmp_path)
        trainer = Trainer(
            midi_dir=str(tmp_path),
            grid_size=16,
            relaxation_steps=5,
            fs=8.0,
        )
        error, meta = trainer.train_step(batch_size=2)
        assert np.isfinite(error)


class TestTrainer:
    def test_train_one_batch_produces_finite_error(self, tmp_path):
        make_corpus(tmp_path)
        trainer = Trainer(
            midi_dir=str(tmp_path),
            grid_size=16,
            relaxation_steps=5,
            fs=8.0,
        )
        error, meta = trainer.train_step(batch_size=2)
        assert np.isfinite(error)

    def test_train_multiple_steps(self, tmp_path):
        make_corpus(tmp_path)
        trainer = Trainer(
            midi_dir=str(tmp_path),
            grid_size=16,
            relaxation_steps=5,
            fs=8.0,
        )
        errors = []
        for _ in range(3):
            err, meta = trainer.train_step(batch_size=2)
            errors.append(err)
        assert all(np.isfinite(e) for e in errors)

    def test_evaluate_error(self, tmp_path):
        make_corpus(tmp_path)
        trainer = Trainer(
            midi_dir=str(tmp_path),
            grid_size=16,
            relaxation_steps=5,
            fs=8.0,
        )
        error = trainer.evaluate_error(batch_size=2)
        assert np.isfinite(error)

    def test_curriculum_integration(self, tmp_path):
        make_corpus(tmp_path)
        trainer = Trainer(
            midi_dir=str(tmp_path),
            grid_size=16,
            relaxation_steps=5,
            fs=8.0,
        )
        assert trainer.curriculum.current_phase == 1

    def test_save_and_load_checkpoint(self, tmp_path):
        midi_dir = tmp_path / "midi"
        midi_dir.mkdir()
        make_corpus(midi_dir)
        trainer = Trainer(
            midi_dir=str(midi_dir),
            grid_size=16,
            relaxation_steps=5,
            fs=8.0,
        )
        trainer.train_step(batch_size=2)
        ckpt_path = tmp_path / "checkpoint"
        trainer.save_checkpoint(str(ckpt_path))

        trainer2 = Trainer(
            midi_dir=str(midi_dir),
            grid_size=16,
            relaxation_steps=5,
            fs=8.0,
        )
        trainer2.load_checkpoint(str(ckpt_path))
        assert jnp.allclose(trainer.grid.weights, trainer2.grid.weights)

    def test_active_error_returned(self, tmp_path):
        """train_step should return active_error in meta dict."""
        make_corpus(tmp_path)
        trainer = Trainer(
            midi_dir=str(tmp_path),
            grid_size=16,
            relaxation_steps=5,
            fs=8.0,
        )
        error, meta = trainer.train_step(batch_size=2)
        assert "active_error" in meta
        assert np.isfinite(meta["active_error"])

    def test_checkpoint_saves_vocabulary(self, tmp_path):
        """save_checkpoint should write metadata.json with vocabulary."""
        midi_dir = tmp_path / "midi"
        midi_dir.mkdir()
        make_corpus(midi_dir)
        trainer = Trainer(
            midi_dir=str(midi_dir),
            grid_size=16,
            relaxation_steps=5,
            fs=8.0,
        )
        ckpt_path = tmp_path / "checkpoint"
        trainer.save_checkpoint(str(ckpt_path))

        metadata = json.loads((ckpt_path / "metadata.json").read_text())
        assert "vocabulary" in metadata
        assert metadata["vocabulary"] == trainer.batch_gen.vocabulary

    def test_checkpoint_saves_log_precision(self, tmp_path):
        """save_checkpoint should write log_precision.npy."""
        midi_dir = tmp_path / "midi"
        midi_dir.mkdir()
        make_corpus(midi_dir)
        trainer = Trainer(
            midi_dir=str(midi_dir),
            grid_size=16,
            relaxation_steps=5,
            fs=8.0,
        )
        trainer.train_step(batch_size=2)
        ckpt_path = tmp_path / "checkpoint"
        trainer.save_checkpoint(str(ckpt_path))

        assert (ckpt_path / "log_precision.npy").exists()

        trainer2 = Trainer(
            midi_dir=str(midi_dir),
            grid_size=16,
            relaxation_steps=5,
            fs=8.0,
        )
        trainer2.load_checkpoint(str(ckpt_path))
        assert jnp.allclose(trainer.grid.log_precision, trainer2.grid.log_precision)

    def test_col_energy_in_meta(self, tmp_path):
        """train_step should return col_energy in meta dict."""
        make_corpus(tmp_path)
        trainer = Trainer(
            midi_dir=str(tmp_path),
            grid_size=16,
            relaxation_steps=5,
            fs=8.0,
        )
        error, meta = trainer.train_step(batch_size=2)
        assert "col_energy" in meta
        assert isinstance(meta["col_energy"], dict)

    def test_fc_connectivity_trains(self, tmp_path):
        """FC connectivity mode should produce finite errors."""
        make_corpus(tmp_path)
        trainer = Trainer(
            midi_dir=str(tmp_path),
            grid_size=16,
            relaxation_steps=5,
            fs=8.0,
            connectivity="fc",
        )
        error, meta = trainer.train_step(batch_size=2)
        assert np.isfinite(error)

    def test_checkpoint_saves_w_temporal(self, tmp_path):
        """save_checkpoint should write w_temporal.npy."""
        midi_dir = tmp_path / "midi"
        midi_dir.mkdir()
        make_corpus(midi_dir)
        trainer = Trainer(
            midi_dir=str(midi_dir),
            grid_size=16,
            relaxation_steps=5,
            fs=8.0,
        )
        trainer.train_step(batch_size=2)
        ckpt_path = tmp_path / "checkpoint"
        trainer.save_checkpoint(str(ckpt_path))

        assert (ckpt_path / "w_temporal.npy").exists()

        trainer2 = Trainer(
            midi_dir=str(midi_dir),
            grid_size=16,
            relaxation_steps=5,
            fs=8.0,
        )
        trainer2.load_checkpoint(str(ckpt_path))
        assert jnp.allclose(trainer.grid.w_temporal, trainer2.grid.w_temporal)

    def test_f1_metrics_in_meta(self, tmp_path):
        """train_step should return precision, recall, F1 in meta dict."""
        make_corpus(tmp_path)
        trainer = Trainer(
            midi_dir=str(tmp_path),
            grid_size=16,
            relaxation_steps=5,
            fs=8.0,
        )
        error, meta = trainer.train_step(batch_size=2)
        assert "f1" in meta
        assert "precision" in meta
        assert "recall" in meta
        assert 0.0 <= meta["f1"] <= 1.0
        assert 0.0 <= meta["precision"] <= 1.0
        assert 0.0 <= meta["recall"] <= 1.0

    def test_checkpoint_saves_fc_weights(self, tmp_path):
        """FC weights should be saved and loaded from checkpoints."""
        midi_dir = tmp_path / "midi"
        midi_dir.mkdir()
        make_corpus(midi_dir)
        trainer = Trainer(
            midi_dir=str(midi_dir),
            grid_size=16,
            relaxation_steps=5,
            fs=8.0,
            connectivity="fc",
        )
        trainer.train_step(batch_size=2)
        ckpt_path = tmp_path / "checkpoint"
        trainer.save_checkpoint(str(ckpt_path))

        assert (ckpt_path / "fc_weights.npy").exists()

        trainer2 = Trainer(
            midi_dir=str(midi_dir),
            grid_size=16,
            relaxation_steps=5,
            fs=8.0,
            connectivity="fc",
        )
        trainer2.load_checkpoint(str(ckpt_path))
        assert jnp.allclose(trainer.grid.fc_weights, trainer2.grid.fc_weights)
