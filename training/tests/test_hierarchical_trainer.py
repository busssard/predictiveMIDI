import numpy as np
import pytest
from pathlib import Path
from corpus.tests.test_scanner import make_test_midi
from training.engine.hierarchical_trainer import HierarchicalTrainer


@pytest.fixture
def midi_corpus(tmp_path):
    """Create a small test MIDI corpus."""
    for i in range(5):
        make_test_midi(tmp_path / f"song{i}.mid", [
            ("Piano", 0, False, [
                (60 + j, j * 0.25, (j + 1) * 0.25, 100) for j in range(32)
            ]),
            ("Bass", 32, False, [
                (36 + j % 4, j * 0.5, (j + 1) * 0.5, 80) for j in range(16)
            ]),
            ("Drums", 0, True, [
                (36, j * 0.25, j * 0.25 + 0.1, 100) for j in range(32)
            ]),
        ])
    return tmp_path


class TestHierarchicalTrainer:
    def test_init_creates_grid(self, midi_corpus):
        trainer = HierarchicalTrainer(
            midi_dir=str(midi_corpus),
            layer_sizes=[128, 64, 32, 64, 128],
            relaxation_steps=5,
            fs=4.0,
        )
        assert trainer.grid.layer_sizes == [128, 64, 32, 64, 128]

    def test_train_step_returns_finite_error(self, midi_corpus):
        trainer = HierarchicalTrainer(
            midi_dir=str(midi_corpus),
            layer_sizes=[32, 16, 32],
            relaxation_steps=5,
            fs=4.0,
        )
        error, meta = trainer.train_step(batch_size=2)
        assert np.isfinite(error)
        assert error > 0

    def test_train_step_returns_f1(self, midi_corpus):
        trainer = HierarchicalTrainer(
            midi_dir=str(midi_corpus),
            layer_sizes=[32, 16, 32],
            relaxation_steps=5,
            fs=4.0,
        )
        _, meta = trainer.train_step(batch_size=2)
        assert "f1" in meta
        assert "precision" in meta
        assert "recall" in meta

    def test_multiple_steps_no_crash(self, midi_corpus):
        trainer = HierarchicalTrainer(
            midi_dir=str(midi_corpus),
            layer_sizes=[32, 16, 32],
            relaxation_steps=5,
            fs=4.0,
        )
        errors = []
        for _ in range(3):
            err, _ = trainer.train_step(batch_size=2)
            errors.append(err)
        assert all(np.isfinite(e) for e in errors)

    def test_save_load_checkpoint(self, midi_corpus, tmp_path):
        trainer = HierarchicalTrainer(
            midi_dir=str(midi_corpus),
            layer_sizes=[32, 16, 32],
            relaxation_steps=5,
            fs=4.0,
        )
        trainer.train_step(batch_size=2)

        ckpt_path = tmp_path / "ckpt"
        trainer.save_checkpoint(str(ckpt_path))

        assert (ckpt_path / "metadata.json").exists()
        assert (ckpt_path / "layer_0_rep.npy").exists()
        assert (ckpt_path / "pred_weight_0.npy").exists()

        # Load into a new trainer and verify weights match
        trainer2 = HierarchicalTrainer(
            midi_dir=str(midi_corpus),
            layer_sizes=[32, 16, 32],
            relaxation_steps=5,
            fs=4.0,
        )
        trainer2.load_checkpoint(str(ckpt_path))
        for w1, w2 in zip(trainer.grid.prediction_weights, trainer2.grid.prediction_weights):
            assert np.allclose(np.array(w1), np.array(w2), atol=1e-6)

    def test_teacher_forcing_partial(self, midi_corpus):
        """With teacher_forcing_ratio < 1.0, training should still work."""
        trainer = HierarchicalTrainer(
            midi_dir=str(midi_corpus),
            layer_sizes=[32, 16, 32],
            relaxation_steps=5,
            fs=4.0,
            teacher_forcing_ratio=0.5,
        )
        error, _ = trainer.train_step(batch_size=2)
        assert np.isfinite(error)

    def test_teacher_forcing_zero(self, midi_corpus):
        """With teacher_forcing_ratio=0, output is never clamped."""
        trainer = HierarchicalTrainer(
            midi_dir=str(midi_corpus),
            layer_sizes=[32, 16, 32],
            relaxation_steps=5,
            fs=4.0,
            teacher_forcing_ratio=0.0,
        )
        error, _ = trainer.train_step(batch_size=2)
        assert np.isfinite(error)

    def test_weight_update_hebbian(self, midi_corpus):
        """Trainer accepts weight_update='hebbian' (default)."""
        trainer = HierarchicalTrainer(
            midi_dir=str(midi_corpus),
            layer_sizes=[32, 16, 32],
            relaxation_steps=5,
            fs=4.0,
            weight_update="hebbian",
        )
        assert trainer.weight_update == "hebbian"
        error, _ = trainer.train_step(batch_size=2)
        assert np.isfinite(error)

    def test_weight_update_autodiff(self, midi_corpus):
        """Trainer accepts weight_update='autodiff' and produces finite error."""
        trainer = HierarchicalTrainer(
            midi_dir=str(midi_corpus),
            layer_sizes=[32, 16, 32],
            relaxation_steps=5,
            fs=4.0,
            weight_update="autodiff",
        )
        assert trainer.weight_update == "autodiff"
        error, _ = trainer.train_step(batch_size=2)
        assert np.isfinite(error)

    def test_weight_update_invalid_raises(self, midi_corpus):
        """Trainer rejects invalid weight_update values."""
        with pytest.raises(ValueError, match="weight_update"):
            HierarchicalTrainer(
                midi_dir=str(midi_corpus),
                layer_sizes=[32, 16, 32],
                relaxation_steps=5,
                fs=4.0,
                weight_update="invalid",
            )
