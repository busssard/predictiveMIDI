import json
from pathlib import Path
import numpy as np
import pytest
from corpus.tests.test_scanner import make_test_midi
from corpus.services.scanner import scan_directory
from corpus.services.vocabulary import build_vocabulary
from corpus.services.batch_generator import BatchGenerator
from training.engine.trainer import Trainer
from training.engine.export import export_model, load_exported_model


class TestEndToEnd:
    def test_full_pipeline(self, tmp_path):
        """Scan -> batch -> train -> export -> verify."""
        # 1. Create test corpus
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

        # 2. Scan and build vocabulary
        results = scan_directory(str(tmp_path))
        assert len(results) == 5
        vocab = build_vocabulary(results)
        assert len(vocab) >= 2

        # 3. Train for a few steps
        trainer = Trainer(
            midi_dir=str(tmp_path),
            grid_size=16,
            relaxation_steps=5,
            fs=4.0,
        )
        errors = []
        for _ in range(3):
            err, meta = trainer.train_step(batch_size=2)
            errors.append(err)
            assert np.isfinite(err)

        # 4. Export
        export_dir = tmp_path / "export"
        export_model(trainer.grid, trainer.batch_gen.vocabulary, str(export_dir))
        assert (export_dir / "state.bin").exists()
        assert (export_dir / "weights.bin").exists()
        assert (export_dir / "config.json").exists()

        # 5. Verify export
        loaded = load_exported_model(str(export_dir))
        assert loaded["config"]["grid_size"] == 16
        assert np.allclose(loaded["weights"], np.array(trainer.grid.weights), atol=1e-6)
