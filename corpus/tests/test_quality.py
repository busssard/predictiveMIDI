"""Tests for quality filtering and instrument selection."""

import pytest
from corpus.management.commands.build_corpus_index import (
    select_top_instruments,
    compute_quality_score,
)


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
