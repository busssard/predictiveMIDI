import jax
import jax.numpy as jnp
import numpy as np
import pytest
from corpus.tests.test_scanner import make_test_midi
from corpus.tests.test_batch_generator import make_corpus
from training.engine.trainer import Trainer


class TestTrainer:
    def test_train_one_batch_produces_finite_error(self, tmp_path):
        make_corpus(tmp_path)
        trainer = Trainer(
            midi_dir=str(tmp_path),
            grid_size=16,
            relaxation_steps=5,
            fs=8.0,
        )
        error = trainer.train_step(batch_size=2)
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
            err = trainer.train_step(batch_size=2)
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
