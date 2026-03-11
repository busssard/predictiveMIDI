"""Tests for quality filtering and instrument selection."""

import json
import pytest
from corpus.management.commands.build_corpus_index import (
    select_top_instruments,
    compute_quality_score,
)
from corpus.services.batch_generator import BatchGenerator
from corpus.services.vocabulary import build_vocabulary
from corpus.tests.test_scanner import make_test_midi


def _make_instrument(program=0, is_drum=False, note_count=50, pitch_range=None):
    """Helper to create an instrument dict."""
    if pitch_range is None:
        pitch_range = [36, 84]
    return {
        "program": program,
        "is_drum": is_drum,
        "note_count": note_count,
        "pitch_range": pitch_range,
    }


def _make_song(tempo=120.0, duration=60.0, instruments=None, dataset="test"):
    """Helper to create a song dict."""
    if instruments is None:
        instruments = [
            _make_instrument(program=0, note_count=100),
            _make_instrument(program=32, note_count=80),
            _make_instrument(program=0, is_drum=True, note_count=60),
        ]
    return {
        "source_paths": ["/fake/path.mid"],
        "instruments": instruments,
        "tempo": tempo,
        "duration": duration,
        "dataset": dataset,
    }


class TestSelectTopInstruments:
    def test_instruments_capped_at_12(self):
        """Songs with >12 instruments should keep only the 12 with most notes."""
        instruments = [
            _make_instrument(program=i, note_count=100 - i * 4)
            for i in range(20)
        ]
        result = select_top_instruments(instruments, max_instruments=12, min_notes=10)
        assert len(result) == 12
        # Should keep the 12 with highest note_count
        note_counts = [inst["note_count"] for inst in result]
        assert note_counts == sorted(note_counts, reverse=True)
        # The smallest in the result should be program=11, note_count=56
        assert result[-1]["note_count"] == 56

    def test_sparse_instruments_excluded(self):
        """Instruments with <10 notes should be dropped before selection."""
        instruments = [
            _make_instrument(program=0, note_count=100),
            _make_instrument(program=1, note_count=5),   # too few
            _make_instrument(program=2, note_count=3),   # too few
            _make_instrument(program=3, note_count=50),
        ]
        result = select_top_instruments(instruments, max_instruments=12, min_notes=10)
        assert len(result) == 2
        assert result[0]["note_count"] == 100
        assert result[1]["note_count"] == 50

    def test_keeps_all_when_under_max(self):
        """If fewer valid instruments than max, keep them all."""
        instruments = [
            _make_instrument(program=0, note_count=100),
            _make_instrument(program=1, note_count=50),
        ]
        result = select_top_instruments(instruments, max_instruments=12, min_notes=10)
        assert len(result) == 2

    def test_empty_input_returns_empty(self):
        result = select_top_instruments([], max_instruments=12, min_notes=10)
        assert result == []

    def test_all_sparse_returns_empty(self):
        instruments = [
            _make_instrument(program=0, note_count=5),
            _make_instrument(program=1, note_count=3),
        ]
        result = select_top_instruments(instruments, max_instruments=12, min_notes=10)
        assert result == []


class TestComputeQualityScore:
    def test_quality_score_computed(self):
        """Each song should have a quality_score field 0.0-1.0."""
        song = _make_song(tempo=120.0, duration=60.0, instruments=[
            _make_instrument(note_count=100),
            _make_instrument(note_count=80),
        ])
        score = compute_quality_score(song)
        assert 0.0 <= score <= 1.0

    def test_normal_song_high_score(self):
        """A normal song should score 1.0."""
        song = _make_song(tempo=120.0, duration=60.0, instruments=[
            _make_instrument(note_count=100),
            _make_instrument(note_count=80),
        ])
        score = compute_quality_score(song)
        assert score == 1.0

    def test_extreme_tempo_penalized(self):
        """Songs with tempo <40 or >240 should be penalized."""
        fast = _make_song(tempo=260.0, duration=60.0, instruments=[
            _make_instrument(note_count=100),
        ])
        slow = _make_song(tempo=35.0, duration=60.0, instruments=[
            _make_instrument(note_count=100),
        ])
        normal = _make_song(tempo=120.0, duration=60.0, instruments=[
            _make_instrument(note_count=100),
        ])
        assert compute_quality_score(fast) < compute_quality_score(normal)
        assert compute_quality_score(slow) < compute_quality_score(normal)

    def test_very_short_song_penalized(self):
        """Songs <5s duration should be penalized."""
        short = _make_song(tempo=120.0, duration=3.0, instruments=[
            _make_instrument(note_count=100),
        ])
        normal = _make_song(tempo=120.0, duration=60.0, instruments=[
            _make_instrument(note_count=100),
        ])
        assert compute_quality_score(short) < compute_quality_score(normal)

    def test_very_long_song_penalized(self):
        """Songs >300s duration should be penalized."""
        long = _make_song(tempo=120.0, duration=400.0, instruments=[
            _make_instrument(note_count=100),
        ])
        normal = _make_song(tempo=120.0, duration=60.0, instruments=[
            _make_instrument(note_count=100),
        ])
        assert compute_quality_score(long) < compute_quality_score(normal)

    def test_sparse_note_density_penalized(self):
        """Songs with very low note density should be heavily penalized."""
        sparse = _make_song(tempo=120.0, duration=100.0, instruments=[
            _make_instrument(note_count=10),  # 0.1 notes/sec
        ])
        dense = _make_song(tempo=120.0, duration=100.0, instruments=[
            _make_instrument(note_count=500),  # 5 notes/sec
        ])
        assert compute_quality_score(sparse) < compute_quality_score(dense)

    def test_score_is_rounded(self):
        """Score should be rounded to 3 decimal places."""
        song = _make_song(tempo=35.0, duration=3.0, instruments=[
            _make_instrument(note_count=1),
        ])
        score = compute_quality_score(song)
        assert score == round(score, 3)


class TestQualityFilterIntegration:
    """Test that quality filters are applied during scanning (via _make_song + filter logic)."""

    def test_extreme_tempo_filtered(self):
        """Songs with tempo <30 or >300 should be excluded by CLI bounds."""
        from corpus.management.commands.build_corpus_index import song_passes_filters

        too_fast = _make_song(tempo=310.0)
        too_slow = _make_song(tempo=25.0)
        normal = _make_song(tempo=120.0)

        assert not song_passes_filters(too_fast, min_tempo=30.0, max_tempo=300.0,
                                        max_duration=600.0)
        assert not song_passes_filters(too_slow, min_tempo=30.0, max_tempo=300.0,
                                        max_duration=600.0)
        assert song_passes_filters(normal, min_tempo=30.0, max_tempo=300.0,
                                   max_duration=600.0)

    def test_extreme_duration_filtered(self):
        """Songs longer than 600s should be excluded."""
        from corpus.management.commands.build_corpus_index import song_passes_filters

        too_long = _make_song(duration=700.0)
        normal = _make_song(duration=60.0)

        assert not song_passes_filters(too_long, min_tempo=30.0, max_tempo=300.0,
                                        max_duration=600.0)
        assert song_passes_filters(normal, min_tempo=30.0, max_tempo=300.0,
                                   max_duration=600.0)

    def test_border_values_pass(self):
        """Songs exactly at the boundary should pass."""
        from corpus.management.commands.build_corpus_index import song_passes_filters

        at_min_tempo = _make_song(tempo=30.0)
        at_max_tempo = _make_song(tempo=300.0)
        at_max_dur = _make_song(duration=600.0)

        assert song_passes_filters(at_min_tempo, min_tempo=30.0, max_tempo=300.0,
                                   max_duration=600.0)
        assert song_passes_filters(at_max_tempo, min_tempo=30.0, max_tempo=300.0,
                                   max_duration=600.0)
        assert song_passes_filters(at_max_dur, min_tempo=30.0, max_tempo=300.0,
                                   max_duration=600.0)


def _make_index_with_quality(tmp_path, songs):
    """Write a corpus index with quality_score fields and return its path."""
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


def _make_real_songs(tmp_path, quality_scores):
    """Create real MIDI files and return scan-result-like dicts with quality scores."""
    songs = []
    for i, qs in enumerate(quality_scores):
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
            "quality_score": qs,
        })
    return songs


class TestQualityAwareBatchSampling:
    def test_quality_threshold_filters_songs(self, tmp_path):
        """Songs below quality_threshold should be excluded from training."""
        songs = _make_real_songs(tmp_path, [0.3, 0.5, 0.8, 1.0])
        index_path = _make_index_with_quality(tmp_path, songs)
        gen = BatchGenerator(
            midi_dir=str(tmp_path / "midi"),
            index_path=str(index_path),
            test_fraction=0,
            quality_threshold=0.6,
        )
        # Only songs with quality_score >= 0.6 should remain (0.8 and 1.0)
        assert len(gen.song_paths) == 2

    def test_quality_threshold_zero_keeps_all(self, tmp_path):
        """Default threshold 0.0 should keep all songs."""
        songs = _make_real_songs(tmp_path, [0.1, 0.5, 0.9])
        index_path = _make_index_with_quality(tmp_path, songs)
        gen = BatchGenerator(
            midi_dir=str(tmp_path / "midi"),
            index_path=str(index_path),
            test_fraction=0,
            quality_threshold=0.0,
        )
        assert len(gen.song_paths) == 3

    def test_quality_threshold_with_scan_results(self, tmp_path):
        """Quality filtering should work with scan_results too."""
        songs = _make_real_songs(tmp_path, [0.2, 0.4, 0.7])
        gen = BatchGenerator(
            midi_dir=str(tmp_path / "midi"),
            scan_results=songs,
            test_fraction=0,
            quality_threshold=0.5,
        )
        # Only song with quality_score 0.7 should remain
        assert len(gen.song_paths) == 1

    def test_songs_without_quality_score_always_included(self, tmp_path):
        """Songs missing quality_score (legacy index) should be included."""
        path = tmp_path / "midi" / "song0.mid"
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
        songs = [{
            "source_paths": [str(path)],
            "instruments": [
                {"program": 0, "is_drum": False, "note_count": 8, "pitch_range": [60, 67]},
                {"program": 32, "is_drum": False, "note_count": 8, "pitch_range": [36, 36]},
                {"program": 0, "is_drum": True, "note_count": 8, "pitch_range": [36, 36]},
            ],
            "tempo": 120.0,
            "duration": 4.0,
            "dataset": "test",
            # No quality_score field
        }]
        gen = BatchGenerator(
            midi_dir=str(tmp_path / "midi"),
            scan_results=songs,
            test_fraction=0,
            quality_threshold=0.5,
        )
        assert len(gen.song_paths) == 1

    def test_piano_rolls_respect_instrument_list(self, tmp_path):
        """_load_piano_rolls should only load instruments listed in scan entry."""
        # Create a MIDI with 3 instruments but index only lists 2
        path = tmp_path / "midi" / "song0.mid"
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
        # Index entry only lists Piano and Bass (instruments at index 0, 1)
        songs = [{
            "source_paths": [str(path)],
            "instruments": [
                {"program": 0, "is_drum": False, "note_count": 8, "pitch_range": [60, 67]},
                {"program": 32, "is_drum": False, "note_count": 8, "pitch_range": [36, 36]},
                # Drums deliberately excluded from the index entry
            ],
            "tempo": 120.0,
            "duration": 4.0,
            "dataset": "test",
            "quality_score": 1.0,
        }]
        gen = BatchGenerator(
            midi_dir=str(tmp_path / "midi"),
            scan_results=songs,
            test_fraction=0,
        )
        rolls = gen._load_piano_rolls(str(path))
        # Should only get 2 instruments (Piano and Bass), not 3
        assert len(rolls) == 2
