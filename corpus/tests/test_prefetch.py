import time
import pytest
from corpus.services.batch_generator import BatchGenerator
from corpus.services.prefetch import PrefetchBatchGenerator
from corpus.tests.test_batch_generator import make_corpus


class TestPrefetchBatchGenerator:
    def test_produces_correct_batch_shapes(self, tmp_path):
        make_corpus(tmp_path)
        gen = BatchGenerator(str(tmp_path), snippet_ticks=16, fs=8.0)
        with PrefetchBatchGenerator(gen, queue_depth=2) as pg:
            batch = pg.generate_batch(batch_size=2)
            assert batch["input"].shape == (2, 16, 128)
            assert batch["target"].shape == (2, 16, 128)
            assert batch["conditioning"].shape[0] == 2

    def test_snippet_ticks_change(self, tmp_path):
        make_corpus(tmp_path)
        gen = BatchGenerator(str(tmp_path), snippet_ticks=16, fs=8.0)
        with PrefetchBatchGenerator(gen, queue_depth=2) as pg:
            batch = pg.generate_batch(batch_size=2)
            assert batch["input"].shape[1] == 16

            pg.snippet_ticks = 8
            batch = pg.generate_batch(batch_size=2)
            assert batch["input"].shape[1] == 8

    def test_close_stops_worker(self, tmp_path):
        make_corpus(tmp_path)
        gen = BatchGenerator(str(tmp_path), snippet_ticks=16, fs=8.0)
        pg = PrefetchBatchGenerator(gen, queue_depth=2)
        pg.generate_batch(batch_size=2)  # starts worker
        assert pg._thread is not None
        assert pg._thread.is_alive()
        pg.close()
        assert pg._thread is None

    def test_context_manager(self, tmp_path):
        make_corpus(tmp_path)
        gen = BatchGenerator(str(tmp_path), snippet_ticks=16, fs=8.0)
        with PrefetchBatchGenerator(gen, queue_depth=2) as pg:
            batch = pg.generate_batch(batch_size=2)
            assert batch is not None
        # After exiting, should be cleaned up
        assert pg._thread is None

    def test_proxied_vocabulary(self, tmp_path):
        make_corpus(tmp_path)
        gen = BatchGenerator(str(tmp_path), snippet_ticks=16, fs=8.0)
        pg = PrefetchBatchGenerator(gen, queue_depth=2)
        assert pg.vocabulary == gen.vocabulary
        pg.close()
