import json
from pathlib import Path
import pytest
from corpus.tests.test_scanner import make_test_midi
from corpus.tests.test_batch_generator import make_corpus
from corpus.services.batch_generator import BatchGenerator
from corpus.services.vocabulary import build_vocabulary


def make_index(tmp_path, songs):
    """Write a minimal corpus_index.json and return its path."""
    vocabulary = build_vocabulary(songs)
    index = {
        "version": 1,
        "created_at": "2026-01-01T00:00:00Z",
        "fs": 8.0,
        "vocabulary": vocabulary,
        "stats": {"total_songs": len(songs)},
        "songs": songs,
    }
    index_path = tmp_path / "corpus_index.json"
    with open(index_path, "w") as f:
        json.dump(index, f)
    return index_path


def make_scan_results(tmp_path, num_songs=3):
    """Create test MIDI files and return scan-result-like dicts."""
    songs = []
    for i in range(num_songs):
        path = tmp_path / "midi" / f"song{i}.mid"
        path.parent.mkdir(parents=True, exist_ok=True)
        make_test_midi(path, [
            ("Piano", 0, False, [
                (60 + j, j * 0.5, (j + 1) * 0.5, 100) for j in range(8)
            ]),
            ("Bass", 32, False, [
                (36, j * 0.5, (j + 1) * 0.5, 80) for j in range(8)
            ]),
            ("Drums", 0, True, [
                (36, j * 0.5, j * 0.5 + 0.25, 100) for j in range(8)
            ]),
        ])
        songs.append({
            "source_paths": [str(path)],
            "instruments": [
                {"program": 0, "is_drum": False, "note_count": 8, "pitch_range": [60, 67]},
                {"program": 32, "is_drum": False, "note_count": 8, "pitch_range": [36, 36]},
                {"program": 0, "is_drum": True, "note_count": 8, "pitch_range": [36, 36]},
            ],
            "tempo": 120.0,
            "duration": 4.0,
            "dataset": "test",
        })
    return songs


class TestCorpusIndexFormat:
    def test_index_has_required_keys(self, tmp_path):
        songs = make_scan_results(tmp_path)
        index_path = make_index(tmp_path, songs)
        with open(index_path) as f:
            index = json.load(f)
        assert index["version"] == 1
        assert "vocabulary" in index
        assert "songs" in index
        assert len(index["songs"]) == 3

    def test_song_entry_has_required_fields(self, tmp_path):
        songs = make_scan_results(tmp_path)
        index_path = make_index(tmp_path, songs)
        with open(index_path) as f:
            index = json.load(f)
        song = index["songs"][0]
        assert "source_paths" in song
        assert "instruments" in song
        assert "tempo" in song
        assert "duration" in song
        assert isinstance(song["source_paths"], list)

    def test_pitch_range_stored_as_list(self, tmp_path):
        songs = make_scan_results(tmp_path)
        index_path = make_index(tmp_path, songs)
        with open(index_path) as f:
            index = json.load(f)
        inst = index["songs"][0]["instruments"][0]
        assert isinstance(inst["pitch_range"], list)

    def test_vocabulary_is_deterministic(self, tmp_path):
        songs = make_scan_results(tmp_path)
        index_path = make_index(tmp_path, songs)
        with open(index_path) as f:
            index = json.load(f)
        vocab = index["vocabulary"]
        assert list(vocab.keys()) == sorted(vocab.keys())


class TestBatchGeneratorFromIndex:
    def test_loads_from_index_path(self, tmp_path):
        songs = make_scan_results(tmp_path)
        index_path = make_index(tmp_path, songs)
        gen = BatchGenerator(
            midi_dir=str(tmp_path / "midi"),
            index_path=str(index_path),
            test_fraction=0,
        )
        assert len(gen.song_paths) == 3
        assert len(gen.vocabulary) > 0

    def test_vocabulary_matches_stored(self, tmp_path):
        songs = make_scan_results(tmp_path)
        index_path = make_index(tmp_path, songs)
        with open(index_path) as f:
            expected_vocab = json.load(f)["vocabulary"]
        gen = BatchGenerator(
            midi_dir=str(tmp_path / "midi"),
            index_path=str(index_path),
        )
        assert gen.vocabulary == expected_vocab

    def test_generate_batch_from_index(self, tmp_path):
        songs = make_scan_results(tmp_path)
        index_path = make_index(tmp_path, songs)
        gen = BatchGenerator(
            midi_dir=str(tmp_path / "midi"),
            snippet_ticks=8,
            fs=8.0,
            index_path=str(index_path),
        )
        batch = gen.generate_batch(batch_size=2)
        assert batch["input"].shape == (2, 8, 128)
        assert batch["target"].shape == (2, 8, 128)

    def test_scan_results_takes_priority_over_index(self, tmp_path):
        """scan_results param should be used even if index_path is given."""
        songs = make_scan_results(tmp_path, num_songs=2)
        index_path = make_index(tmp_path, songs)
        # Make a different scan_results with 1 song
        one_song = make_scan_results(tmp_path, num_songs=1)
        gen = BatchGenerator(
            midi_dir=str(tmp_path / "midi"),
            scan_results=one_song,
            index_path=str(index_path),
            test_fraction=0,
        )
        # scan_results has 1 song, index has 2 — scan_results should win
        assert len(gen.song_paths) == 1
