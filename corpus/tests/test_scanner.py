import tempfile
from pathlib import Path
import pretty_midi
import pytest
from corpus.services.scanner import scan_midi_file, scan_directory


def make_test_midi(path, instruments):
    """Create a MIDI file with given instruments.

    instruments: list of (name, program, is_drum, notes) tuples.
    notes: list of (pitch, start, end, velocity) tuples.
    """
    midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    for name, program, is_drum, notes in instruments:
        inst = pretty_midi.Instrument(program=program, is_drum=is_drum, name=name)
        for pitch, start, end, velocity in notes:
            inst.notes.append(pretty_midi.Note(
                velocity=velocity, pitch=pitch, start=start, end=end,
            ))
        midi.instruments.append(inst)
    midi.write(str(path))


class TestScanMidiFile:
    def test_extracts_instruments(self, tmp_path):
        path = tmp_path / "test.mid"
        make_test_midi(path, [
            ("Piano", 0, False, [(60, 0.0, 1.0, 100)]),
            ("Bass", 32, False, [(36, 0.0, 2.0, 80)]),
        ])
        result = scan_midi_file(path)
        assert result["path"] == str(path)
        assert len(result["instruments"]) == 2
        assert result["instruments"][0]["name"] == "Piano"
        assert result["instruments"][0]["program"] == 0
        assert result["instruments"][1]["name"] == "Bass"
        assert result["instruments"][1]["program"] == 32

    def test_extracts_tempo(self, tmp_path):
        path = tmp_path / "test.mid"
        make_test_midi(path, [("Piano", 0, False, [(60, 0.0, 1.0, 100)])])
        result = scan_midi_file(path)
        assert result["tempo"] == pytest.approx(120.0, abs=1.0)

    def test_extracts_duration(self, tmp_path):
        path = tmp_path / "test.mid"
        make_test_midi(path, [
            ("Piano", 0, False, [(60, 0.0, 3.5, 100)]),
        ])
        result = scan_midi_file(path)
        assert result["duration"] == pytest.approx(3.5, abs=0.1)

    def test_identifies_drums(self, tmp_path):
        path = tmp_path / "test.mid"
        make_test_midi(path, [
            ("Drums", 0, True, [(36, 0.0, 0.5, 100)]),
        ])
        result = scan_midi_file(path)
        assert result["instruments"][0]["is_drum"] is True


class TestScanDirectory:
    def test_scans_all_midi_files(self, tmp_path):
        for i in range(3):
            make_test_midi(
                tmp_path / f"song{i}.mid",
                [("Piano", 0, False, [(60, 0.0, 1.0, 100)])],
            )
        # Also create a non-MIDI file that should be ignored
        (tmp_path / "readme.txt").write_text("not midi")
        results = scan_directory(str(tmp_path))
        assert len(results) == 3

    def test_returns_empty_for_empty_dir(self, tmp_path):
        results = scan_directory(str(tmp_path))
        assert results == []
