import pytest
from pathlib import Path
from corpus.services.dataset_scanner import (
    _parse_aam_filename,
    scan_lakh,
    scan_aam,
    scan_slakh,
    scan_datasets,
)
from corpus.services.batch_generator import BatchGenerator
from corpus.tests.test_scanner import make_test_midi


PIANO_NOTES = [(60 + j, j * 0.5, (j + 1) * 0.5, 100) for j in range(8)]
BASS_NOTES = [(36 + j % 4, j * 0.5, (j + 1) * 0.5, 80) for j in range(8)]
DRUM_NOTES = [(36, j * 0.5, j * 0.5 + 0.25, 100) for j in range(8)]


class TestParseAamFilename:
    def test_standard_filename(self):
        assert _parse_aam_filename("0001_Drums.mid") == (1, "Drums")

    def test_multi_word_instrument(self):
        assert _parse_aam_filename("0042_ElectricPiano.mid") == (42, "ElectricPiano")

    def test_non_matching(self):
        assert _parse_aam_filename("readme.txt") is None

    def test_no_underscore(self):
        assert _parse_aam_filename("0001.mid") is None


class TestScanLakh:
    def test_finds_files_in_hash_subdirs(self, tmp_path):
        lmd = tmp_path / "lmd_full" / "a"
        lmd.mkdir(parents=True)
        make_test_midi(
            lmd / "abc123.mid",
            [("Piano", 0, False, PIANO_NOTES)],
        )
        results = scan_lakh(str(tmp_path))
        assert len(results) == 1
        assert results[0]["dataset"] == "lakh"
        assert results[0]["source_paths"] == [str(lmd / "abc123.mid")]

    def test_empty_dir(self, tmp_path):
        lmd = tmp_path / "lmd_full"
        lmd.mkdir()
        assert scan_lakh(str(tmp_path)) == []

    def test_missing_dir(self, tmp_path):
        assert scan_lakh(str(tmp_path)) == []


class TestScanAam:
    def test_groups_by_song_id(self, tmp_path):
        make_test_midi(
            tmp_path / "0001_Piano.mid",
            [("Piano", 0, False, PIANO_NOTES)],
        )
        make_test_midi(
            tmp_path / "0001_Bass.mid",
            [("Bass", 32, False, BASS_NOTES)],
        )
        make_test_midi(
            tmp_path / "0002_Piano.mid",
            [("Piano", 0, False, PIANO_NOTES)],
        )
        results = scan_aam(str(tmp_path))
        assert len(results) == 2
        # Song 1 should have 2 instruments from 2 files
        song1 = [r for r in results if "0001" in r["path"]][0]
        assert len(song1["instruments"]) == 2
        assert len(song1["source_paths"]) == 2
        assert song1["dataset"] == "aam"

    def test_excludes_demo(self, tmp_path):
        make_test_midi(
            tmp_path / "0007_Piano.mid",
            [("Piano", 0, False, PIANO_NOTES)],
        )
        make_test_midi(
            tmp_path / "0007_Demo.mid",
            [("Demo", 0, False, PIANO_NOTES)],
        )
        results = scan_aam(str(tmp_path))
        assert len(results) == 1
        assert len(results[0]["source_paths"]) == 1
        assert "Demo" not in results[0]["source_paths"][0]

    def test_duration_is_max(self, tmp_path):
        # Piano: 4 seconds, Bass: 2 seconds
        make_test_midi(
            tmp_path / "0001_Piano.mid",
            [("Piano", 0, False, [(60, 0.0, 4.0, 100)])],
        )
        make_test_midi(
            tmp_path / "0001_Bass.mid",
            [("Bass", 32, False, [(36, 0.0, 2.0, 80)])],
        )
        results = scan_aam(str(tmp_path))
        assert results[0]["duration"] == pytest.approx(4.0, abs=0.1)

    def test_missing_dir(self, tmp_path):
        assert scan_aam(str(tmp_path / "nonexistent")) == []


class TestScanSlakh:
    def _make_track(self, split_dir, track_name, use_all_src=True):
        """Helper to create a Slakh track directory."""
        track_dir = split_dir / track_name
        track_dir.mkdir(parents=True)

        if use_all_src:
            make_test_midi(
                track_dir / "all_src.mid",
                [
                    ("Piano", 0, False, PIANO_NOTES),
                    ("Bass", 32, False, BASS_NOTES),
                ],
            )
        else:
            midi_dir = track_dir / "MIDI"
            midi_dir.mkdir()
            make_test_midi(
                midi_dir / "S00.mid",
                [("Piano", 0, False, PIANO_NOTES)],
            )
            make_test_midi(
                midi_dir / "S01.mid",
                [("Bass", 32, False, BASS_NOTES)],
            )
        return track_dir

    def test_uses_all_src(self, tmp_path):
        train = tmp_path / "train"
        train.mkdir()
        self._make_track(train, "Track00001", use_all_src=True)
        results = scan_slakh(str(tmp_path))
        assert len(results) == 1
        assert results[0]["dataset"] == "slakh"
        assert "all_src.mid" in results[0]["source_paths"][0]

    def test_falls_back_to_stems(self, tmp_path):
        train = tmp_path / "train"
        train.mkdir()
        self._make_track(train, "Track00001", use_all_src=False)
        results = scan_slakh(str(tmp_path))
        assert len(results) == 1
        assert len(results[0]["source_paths"]) == 2
        assert len(results[0]["instruments"]) == 2

    def test_respects_splits(self, tmp_path):
        for split in ("train", "validation", "test"):
            s = tmp_path / split
            s.mkdir()
            self._make_track(s, "Track00001", use_all_src=True)

        results = scan_slakh(str(tmp_path), splits=["train", "test"])
        assert len(results) == 2
        splits_found = {r["split"] for r in results}
        assert splits_found == {"train", "test"}

    def test_includes_split_metadata(self, tmp_path):
        val = tmp_path / "validation"
        val.mkdir()
        self._make_track(val, "Track00001", use_all_src=True)
        results = scan_slakh(str(tmp_path))
        assert results[0]["split"] == "validation"

    def test_missing_dir(self, tmp_path):
        assert scan_slakh(str(tmp_path / "nonexistent")) == []


class TestScanDatasets:
    def test_auto_detects_datasets(self, tmp_path):
        # Set up AAM
        aam = tmp_path / "aam_midi"
        aam.mkdir()
        make_test_midi(
            aam / "0001_Piano.mid",
            [("Piano", 0, False, PIANO_NOTES)],
        )

        # Set up Lakh
        lmd = tmp_path / "lakh_midi" / "lmd_full" / "a"
        lmd.mkdir(parents=True)
        make_test_midi(
            lmd / "abc.mid",
            [("Piano", 0, False, PIANO_NOTES)],
        )

        results = scan_datasets(str(tmp_path))
        datasets_found = {r["dataset"] for r in results}
        assert "lakh" in datasets_found
        assert "aam" in datasets_found

    def test_selective_scanning(self, tmp_path):
        # Set up both
        aam = tmp_path / "aam_midi"
        aam.mkdir()
        make_test_midi(
            aam / "0001_Piano.mid",
            [("Piano", 0, False, PIANO_NOTES)],
        )
        lmd = tmp_path / "lakh_midi" / "lmd_full" / "a"
        lmd.mkdir(parents=True)
        make_test_midi(
            lmd / "abc.mid",
            [("Piano", 0, False, PIANO_NOTES)],
        )

        results = scan_datasets(str(tmp_path), datasets=["aam"])
        assert all(r["dataset"] == "aam" for r in results)

    def test_missing_data_dir(self, tmp_path):
        results = scan_datasets(str(tmp_path / "nonexistent"))
        assert results == []


class TestBatchGeneratorMultiSource:
    def test_loads_multi_file_song(self, tmp_path):
        # Create two separate MIDI files for one "song"
        make_test_midi(
            tmp_path / "piano.mid",
            [("Piano", 0, False, PIANO_NOTES)],
        )
        make_test_midi(
            tmp_path / "bass.mid",
            [("Bass", 32, False, BASS_NOTES)],
        )
        make_test_midi(
            tmp_path / "drums.mid",
            [("Drums", 0, True, DRUM_NOTES)],
        )

        scan_results = [{
            "path": "multi_song",
            "instruments": [
                {"name": "Piano", "program": 0, "is_drum": False,
                 "note_count": 8, "pitch_range": (60, 67)},
                {"name": "Bass", "program": 32, "is_drum": False,
                 "note_count": 8, "pitch_range": (36, 39)},
                {"name": "Drums", "program": 0, "is_drum": True,
                 "note_count": 8, "pitch_range": (36, 36)},
            ],
            "tempo": 120.0,
            "duration": 4.0,
            "source_paths": [
                str(tmp_path / "piano.mid"),
                str(tmp_path / "bass.mid"),
                str(tmp_path / "drums.mid"),
            ],
            "dataset": "aam",
        }]

        gen = BatchGenerator(
            str(tmp_path), snippet_ticks=16, fs=8.0,
            scan_results=scan_results, rng_seed=42,
        )
        rolls = gen._load_piano_rolls("multi_song")
        assert len(rolls) == 3

    def test_backward_compat_no_scan_results(self, tmp_path):
        """Without scan_results, BatchGenerator scans the directory as before."""
        make_test_midi(
            tmp_path / "song.mid",
            [
                ("Piano", 0, False, PIANO_NOTES),
                ("Bass", 32, False, BASS_NOTES),
            ],
        )
        gen = BatchGenerator(str(tmp_path), snippet_ticks=16, fs=8.0, test_fraction=0)
        batch = gen.generate_batch(batch_size=1)
        assert batch["input"].shape == (1, 16, 128)

    def test_generate_batch_with_multi_source(self, tmp_path):
        """Full batch generation with multi-source scan results."""
        make_test_midi(
            tmp_path / "piano.mid",
            [("Piano", 0, False, PIANO_NOTES)],
        )
        make_test_midi(
            tmp_path / "bass.mid",
            [("Bass", 32, False, BASS_NOTES)],
        )
        make_test_midi(
            tmp_path / "drums.mid",
            [("Drums", 0, True, DRUM_NOTES)],
        )

        scan_results = [{
            "path": "multi_song",
            "instruments": [
                {"name": "Piano", "program": 0, "is_drum": False,
                 "note_count": 8, "pitch_range": (60, 67)},
                {"name": "Bass", "program": 32, "is_drum": False,
                 "note_count": 8, "pitch_range": (36, 39)},
                {"name": "Drums", "program": 0, "is_drum": True,
                 "note_count": 8, "pitch_range": (36, 36)},
            ],
            "tempo": 120.0,
            "duration": 4.0,
            "source_paths": [
                str(tmp_path / "piano.mid"),
                str(tmp_path / "bass.mid"),
                str(tmp_path / "drums.mid"),
            ],
            "dataset": "aam",
        }]

        gen = BatchGenerator(
            str(tmp_path), snippet_ticks=16, fs=8.0,
            scan_results=scan_results, rng_seed=42, test_fraction=0,
        )
        batch = gen.generate_batch(batch_size=2)
        assert batch["input"].shape == (2, 16, 128)
        assert batch["target"].shape == (2, 16, 128)
