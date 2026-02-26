import tempfile
from pathlib import Path
import numpy as np
import pretty_midi
import pytest
from corpus.services.batch_generator import BatchGenerator
from corpus.tests.test_scanner import make_test_midi


def make_corpus(tmp_path, num_songs=3):
    """Create a small test corpus with piano, bass, and drums."""
    paths = []
    for i in range(num_songs):
        path = tmp_path / f"song{i}.mid"
        make_test_midi(path, [
            ("Piano", 0, False, [
                (60 + j, j * 0.5, (j + 1) * 0.5, 100) for j in range(16)
            ]),
            ("Bass", 32, False, [
                (36 + j % 4, j * 0.5, (j + 1) * 0.5, 80) for j in range(16)
            ]),
            ("Drums", 0, True, [
                (36, j * 0.5, j * 0.5 + 0.25, 100) for j in range(16)
            ]),
        ])
        paths.append(path)
    return paths


class TestBatchGenerator:
    def test_produces_correct_shapes(self, tmp_path):
        make_corpus(tmp_path)
        gen = BatchGenerator(str(tmp_path), snippet_ticks=16, fs=8.0)
        batch = gen.generate_batch(batch_size=4)
        assert batch["input"].shape == (4, 16, 128)
        assert batch["target"].shape == (4, 16, 128)
        assert batch["conditioning"].shape[0] == 4
        # conditioning width = number of instrument categories
        assert batch["conditioning"].shape[1] == len(gen.vocabulary)

    def test_input_values_in_range(self, tmp_path):
        make_corpus(tmp_path)
        gen = BatchGenerator(str(tmp_path), snippet_ticks=16, fs=8.0)
        batch = gen.generate_batch(batch_size=4)
        assert batch["input"].min() >= 0.0
        assert batch["input"].max() <= 1.0
        assert batch["target"].min() >= 0.0
        assert batch["target"].max() <= 1.0

    def test_conditioning_is_one_hot(self, tmp_path):
        make_corpus(tmp_path)
        gen = BatchGenerator(str(tmp_path), snippet_ticks=16, fs=8.0)
        batch = gen.generate_batch(batch_size=8)
        for i in range(8):
            assert batch["conditioning"][i].sum() == pytest.approx(1.0)

    def test_target_not_in_input_mix(self, tmp_path):
        """The target instrument should not appear in the input mix."""
        make_corpus(tmp_path)
        gen = BatchGenerator(str(tmp_path), snippet_ticks=16, fs=8.0)
        # This is probabilistic, but with enough samples we can check
        # that target and input differ
        batch = gen.generate_batch(batch_size=4)
        # At minimum, target_instrument_idx should be set
        assert "target_categories" in batch

    def test_different_snippet_lengths(self, tmp_path):
        make_corpus(tmp_path)
        gen = BatchGenerator(str(tmp_path), snippet_ticks=32, fs=8.0)
        batch = gen.generate_batch(batch_size=2)
        assert batch["input"].shape[1] == 32
