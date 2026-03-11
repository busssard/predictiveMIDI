import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from django.core.management.base import BaseCommand
from django.conf import settings

from corpus.services.scanner import scan_midi_file
from corpus.services.vocabulary import build_vocabulary


def select_top_instruments(instruments, max_instruments=12, min_notes=10):
    """Keep up to max_instruments, sorted by note_count descending.
    Drop instruments with fewer than min_notes."""
    valid = [inst for inst in instruments if inst["note_count"] >= min_notes]
    valid.sort(key=lambda x: x["note_count"], reverse=True)
    return valid[:max_instruments]


def compute_quality_score(song):
    """Compute a quality score 0.0-1.0 for a song based on tempo, duration, density."""
    score = 1.0
    if song["tempo"] < 40 or song["tempo"] > 240:
        score *= 0.7
    if song["duration"] < 5:
        score *= 0.5
    if song["duration"] > 300:
        score *= 0.7
    total_notes = sum(i["note_count"] for i in song["instruments"])
    if total_notes / max(song["duration"], 1) < 0.5:
        score *= 0.3
    return round(score, 3)


def song_passes_filters(song, min_tempo=30.0, max_tempo=300.0, max_duration=600.0):
    """Check whether a song passes the tempo and duration filters."""
    if song["tempo"] < min_tempo or song["tempo"] > max_tempo:
        return False
    if song["duration"] > max_duration:
        return False
    return True


def _parse_aam_filename(filename):
    m = re.match(r"^(\d+)_(.+)\.mid$", filename, re.IGNORECASE)
    if m:
        return int(m.group(1)), m.group(2)
    return None


def _error_category(e):
    """Classify an exception into a short label."""
    msg = str(e)
    if "MThd not found" in msg:
        return "not_midi"
    if "key" in msg.lower() and "signature" in msg.lower():
        return "bad_key_sig"
    if "data byte" in msg:
        return "bad_data_byte"
    if "EOFError" in type(e).__name__ or "unexpected end" in msg.lower():
        return "truncated"
    return type(e).__name__


class ProgressTracker:
    """Live progress display for scanning."""

    def __init__(self, dataset, total, stream):
        self.dataset = dataset
        self.total = total
        self.stream = stream
        self.ok = 0
        self.failed = 0
        self.skipped = 0
        self.errors = defaultdict(int)
        self._t0 = time.time()
        self._last_print = 0

    def record_ok(self):
        self.ok += 1
        self._maybe_print()

    def record_skip(self, reason="filtered"):
        self.skipped += 1
        self._maybe_print()

    def record_error(self, e):
        self.failed += 1
        cat = _error_category(e)
        self.errors[cat] += 1
        self._maybe_print()

    def _maybe_print(self):
        now = time.time()
        if now - self._last_print < 0.3:
            return
        self._last_print = now
        self._print_line()

    def _print_line(self):
        done = self.ok + self.failed + self.skipped
        pct = done / self.total * 100 if self.total else 0
        elapsed = time.time() - self._t0
        rate = done / elapsed if elapsed > 0 else 0

        err_parts = " | ".join(f"{k}:{v}" for k, v in sorted(self.errors.items()))
        err_str = f" | errors: {err_parts}" if err_parts else ""

        line = (
            f"\r[{self.dataset}] {done}/{self.total} ({pct:.1f}%) "
            f"ok:{self.ok} skip:{self.skipped} fail:{self.failed}"
            f"{err_str} [{rate:.0f}/s]"
        )
        self.stream.write(line)
        self.stream.flush()

    def finish(self):
        self._print_line()
        self.stream.write("\n")
        self.stream.flush()


class Command(BaseCommand):
    help = "Build a unified corpus index from all MIDI datasets"

    def add_arguments(self, parser):
        parser.add_argument("--midi-dir", default=settings.MIDI_DATA_DIR)
        parser.add_argument(
            "--output",
            default=str(Path(settings.BASE_DIR) / "data" / "corpus_index.json"),
        )
        parser.add_argument(
            "--datasets", nargs="*", default=None,
            help="Which datasets to scan (lakh, aam, slakh). Default: all detected.",
        )
        parser.add_argument("--min-instruments", type=int, default=2)
        parser.add_argument("--min-duration", type=float, default=2.0)
        parser.add_argument("--max-duration", type=float, default=600.0)
        parser.add_argument("--min-tempo", type=float, default=30.0)
        parser.add_argument("--max-tempo", type=float, default=300.0)
        parser.add_argument("--max-instruments", type=int, default=12)
        parser.add_argument("--min-notes-per-instrument", type=int, default=10)

    def handle(self, *args, **options):
        midi_dir = Path(options["midi_dir"])
        output_path = Path(options["output"])
        min_inst = options["min_instruments"]
        min_dur = options["min_duration"]

        # Quality filter options
        quality_opts = {
            "max_duration": options["max_duration"],
            "min_tempo": options["min_tempo"],
            "max_tempo": options["max_tempo"],
            "max_instruments": options["max_instruments"],
            "min_notes_per_instrument": options["min_notes_per_instrument"],
        }

        # Load existing index for resumability
        existing_paths = set()
        existing_songs = []
        if output_path.exists():
            self.stdout.write(f"Loading existing index for resume...")
            with open(output_path) as f:
                existing = json.load(f)
            existing_songs = existing.get("songs", [])
            existing_paths = {s["source_paths"][0] for s in existing_songs}
            self.stdout.write(f"  {len(existing_songs)} songs already indexed\n")

        new_songs = []
        stats_by_dataset = {}

        selected = options["datasets"]

        # --- Lakh ---
        if selected is None or "lakh" in selected:
            lakh_songs = self._scan_lakh(
                midi_dir / "lakh_midi", existing_paths, min_inst, min_dur,
                quality_opts,
            )
            new_songs.extend(lakh_songs)
            stats_by_dataset["lakh"] = len(lakh_songs)

        # --- AAM ---
        if selected is None or "aam" in selected:
            aam_songs = self._scan_aam(
                midi_dir / "aam_midi", existing_paths, min_inst, min_dur,
                quality_opts,
            )
            new_songs.extend(aam_songs)
            stats_by_dataset["aam"] = len(aam_songs)

        # --- Slakh ---
        if selected is None or "slakh" in selected:
            slakh_songs = self._scan_slakh(
                midi_dir / "slakh2100_midi" / "slakh2100_flac_redux",
                existing_paths, min_inst, min_dur, quality_opts,
            )
            new_songs.extend(slakh_songs)
            stats_by_dataset["slakh"] = len(slakh_songs)

        # Merge with existing
        all_songs = existing_songs + new_songs
        self.stdout.write(f"\nTotal songs: {len(all_songs)}\n")

        if not all_songs:
            self.stderr.write("No songs to index. Check --midi-dir path.\n")
            return

        vocabulary = build_vocabulary(all_songs)

        index = {
            "version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "fs": 8.0,
            "vocabulary": vocabulary,
            "stats": {
                "total_songs": len(all_songs),
                "by_dataset": stats_by_dataset,
            },
            "songs": all_songs,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output_path.with_suffix(".json.tmp")
        with open(tmp_path, "w") as f:
            json.dump(index, f, separators=(",", ":"))
        os.replace(tmp_path, output_path)

        size_mb = output_path.stat().st_size / 1024 / 1024
        self.stdout.write(
            f"Index written to {output_path}\n"
            f"  Songs: {len(all_songs)}\n"
            f"  Vocabulary: {list(vocabulary.keys())}\n"
            f"  File size: {size_mb:.1f} MB\n"
        )

    # ------------------------------------------------------------------
    # Per-dataset scanners with progress
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_instruments(instruments):
        """Convert instrument dicts to JSON-safe native Python types."""
        return [
            {
                "program": int(inst["program"]),
                "is_drum": bool(inst["is_drum"]),
                "note_count": int(inst["note_count"]),
                "pitch_range": [int(x) for x in inst["pitch_range"]],
            }
            for inst in instruments
        ]

    def _make_song(self, result, dataset, quality_opts=None, **extra):
        """Normalize a scan_midi_file result into the index format.

        Applies instrument selection and quality score if quality_opts given.
        """
        instruments = self._normalize_instruments(result["instruments"])

        if quality_opts:
            instruments = select_top_instruments(
                instruments,
                max_instruments=quality_opts["max_instruments"],
                min_notes=quality_opts["min_notes_per_instrument"],
            )

        song = {
            "source_paths": result.get("source_paths", [result["path"]]),
            "instruments": instruments,
            "tempo": float(result["tempo"]),
            "duration": float(result["duration"]),
            "dataset": dataset,
        }
        song.update(extra)
        song["quality_score"] = compute_quality_score(song)
        return song

    def _scan_lakh(self, base_dir, existing_paths, min_inst, min_dur,
                   quality_opts=None):
        """Lakh: each .mid under lmd_full/ is one song."""
        lmd_dir = base_dir / "lmd_full"
        if not lmd_dir.is_dir():
            self.stdout.write(f"[lakh] not found at {lmd_dir}, skipping\n")
            return []

        files = sorted(lmd_dir.rglob("*.mid"))
        progress = ProgressTracker("lakh", len(files), sys.stderr)
        songs = []

        for path in files:
            spath = str(path)
            if spath in existing_paths:
                progress.record_skip("indexed")
                continue

            try:
                result = scan_midi_file(path)
            except Exception as e:
                progress.record_error(e)
                continue

            if len(result["instruments"]) < min_inst:
                progress.record_skip("few_instruments")
                continue
            if result["duration"] < min_dur:
                progress.record_skip("too_short")
                continue

            result["source_paths"] = [spath]
            song = self._make_song(result, "lakh", quality_opts=quality_opts)

            # Apply quality filters (tempo/duration bounds)
            if quality_opts and not song_passes_filters(
                song,
                min_tempo=quality_opts["min_tempo"],
                max_tempo=quality_opts["max_tempo"],
                max_duration=quality_opts["max_duration"],
            ):
                progress.record_skip("quality_filtered")
                continue

            # Reject songs with <min_inst valid instruments after selection
            if len(song["instruments"]) < min_inst:
                progress.record_skip("few_instruments_after_filter")
                continue

            songs.append(song)
            progress.record_ok()

        progress.finish()
        return songs

    def _scan_aam(self, base_dir, existing_paths, min_inst, min_dur,
                  quality_opts=None):
        """AAM: group {songID}_{Instrument}.mid files into songs."""
        if not base_dir.is_dir():
            self.stdout.write(f"[aam] not found at {base_dir}, skipping\n")
            return []

        # Group files by song ID
        song_files = {}
        for path in sorted(base_dir.glob("*.mid")):
            parsed = _parse_aam_filename(path.name)
            if parsed is None:
                continue
            song_id, inst_name = parsed
            if inst_name == "Demo":
                continue
            song_files.setdefault(song_id, []).append(path)

        total_files = sum(len(v) for v in song_files.values())
        progress = ProgressTracker("aam", total_files, sys.stderr)
        songs = []

        for song_id in sorted(song_files):
            paths = song_files[song_id]
            primary = str(paths[0])
            if primary in existing_paths:
                for _ in paths:
                    progress.record_skip("indexed")
                continue

            all_instruments = []
            tempos = []
            durations = []
            ok_paths = []

            for path in paths:
                try:
                    scan = scan_midi_file(path)
                    all_instruments.extend(scan["instruments"])
                    tempos.append(scan["tempo"])
                    durations.append(scan["duration"])
                    ok_paths.append(str(path))
                    progress.record_ok()
                except Exception as e:
                    progress.record_error(e)

            if not ok_paths:
                continue
            if len(all_instruments) < min_inst:
                progress.record_skip("few_instruments")
                continue
            if max(durations) < min_dur:
                progress.record_skip("too_short")
                continue

            # Build song dict, then apply quality filters
            instruments = self._normalize_instruments(all_instruments)
            if quality_opts:
                instruments = select_top_instruments(
                    instruments,
                    max_instruments=quality_opts["max_instruments"],
                    min_notes=quality_opts["min_notes_per_instrument"],
                )

            song = {
                "source_paths": ok_paths,
                "instruments": instruments,
                "tempo": float(tempos[0]),
                "duration": float(max(durations)),
                "dataset": "aam",
            }
            song["quality_score"] = compute_quality_score(song)

            if quality_opts and not song_passes_filters(
                song,
                min_tempo=quality_opts["min_tempo"],
                max_tempo=quality_opts["max_tempo"],
                max_duration=quality_opts["max_duration"],
            ):
                progress.record_skip("quality_filtered")
                continue

            if len(song["instruments"]) < min_inst:
                progress.record_skip("few_instruments_after_filter")
                continue

            songs.append(song)

        progress.finish()
        return songs

    def _scan_slakh(self, base_dir, existing_paths, min_inst, min_dur,
                    quality_opts=None):
        """Slakh: use all_src.mid, fallback to MIDI/S*.mid stems."""
        if not base_dir.is_dir():
            self.stdout.write(f"[slakh] not found at {base_dir}, skipping\n")
            return []

        # Count tracks for progress
        track_dirs = []
        for split in ("train", "validation", "test"):
            split_dir = base_dir / split
            if split_dir.is_dir():
                for d in sorted(split_dir.iterdir()):
                    if d.is_dir():
                        track_dirs.append((split, d))

        progress = ProgressTracker("slakh", len(track_dirs), sys.stderr)
        songs = []

        for split, track_dir in track_dirs:
            all_src = track_dir / "all_src.mid"

            if all_src.exists():
                spath = str(all_src)
                if spath in existing_paths:
                    progress.record_skip("indexed")
                    continue

                try:
                    result = scan_midi_file(all_src)
                except Exception as e:
                    progress.record_error(e)
                    continue

                if len(result["instruments"]) < min_inst:
                    progress.record_skip("few_instruments")
                    continue
                if result["duration"] < min_dur:
                    progress.record_skip("too_short")
                    continue

                result["source_paths"] = [spath]
                song = self._make_song(
                    result, "slakh", quality_opts=quality_opts, split=split,
                )

                if quality_opts and not song_passes_filters(
                    song,
                    min_tempo=quality_opts["min_tempo"],
                    max_tempo=quality_opts["max_tempo"],
                    max_duration=quality_opts["max_duration"],
                ):
                    progress.record_skip("quality_filtered")
                    continue

                if len(song["instruments"]) < min_inst:
                    progress.record_skip("few_instruments_after_filter")
                    continue

                songs.append(song)
                progress.record_ok()
            else:
                # Fallback: merge stems
                midi_dir = track_dir / "MIDI"
                if not midi_dir.is_dir():
                    progress.record_skip("no_midi_dir")
                    continue

                stems = sorted(midi_dir.glob("S*.mid"))
                if not stems:
                    progress.record_skip("no_stems")
                    continue

                primary = str(stems[0])
                if primary in existing_paths:
                    progress.record_skip("indexed")
                    continue

                all_instruments = []
                tempos = []
                durations = []
                ok_stems = []

                for stem in stems:
                    try:
                        scan = scan_midi_file(stem)
                        all_instruments.extend(scan["instruments"])
                        tempos.append(scan["tempo"])
                        durations.append(scan["duration"])
                        ok_stems.append(str(stem))
                    except Exception as e:
                        progress.record_error(e)

                if not ok_stems:
                    progress.record_error(Exception("all_stems_failed"))
                    continue
                if len(all_instruments) < min_inst:
                    progress.record_skip("few_instruments")
                    continue
                if max(durations) < min_dur:
                    progress.record_skip("too_short")
                    continue

                # Build song dict, then apply quality filters
                instruments = self._normalize_instruments(all_instruments)
                if quality_opts:
                    instruments = select_top_instruments(
                        instruments,
                        max_instruments=quality_opts["max_instruments"],
                        min_notes=quality_opts["min_notes_per_instrument"],
                    )

                song = {
                    "source_paths": ok_stems,
                    "instruments": instruments,
                    "tempo": float(tempos[0]),
                    "duration": float(max(durations)),
                    "dataset": "slakh",
                    "split": split,
                }
                song["quality_score"] = compute_quality_score(song)

                if quality_opts and not song_passes_filters(
                    song,
                    min_tempo=quality_opts["min_tempo"],
                    max_tempo=quality_opts["max_tempo"],
                    max_duration=quality_opts["max_duration"],
                ):
                    progress.record_skip("quality_filtered")
                    continue

                if len(song["instruments"]) < min_inst:
                    progress.record_skip("few_instruments_after_filter")
                    continue

                songs.append(song)
                progress.record_ok()

        progress.finish()
        return songs
