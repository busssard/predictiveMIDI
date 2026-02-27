"""Dataset-specific MIDI scanners for Lakh, AAM, and Slakh datasets.

Each scanner produces the standard scan result format from scanner.py
plus additional keys:
    source_paths: list of file paths to load (for multi-file songs)
    dataset: provenance tag ("lakh", "aam", "slakh")
"""

import re
from pathlib import Path

from corpus.services.scanner import scan_midi_file


def _parse_aam_filename(filename):
    """Parse an AAM filename like '0001_Drums.mid' into (song_id, instrument_name).

    Returns None if the filename doesn't match the expected pattern.
    """
    m = re.match(r"^(\d+)_(.+)\.mid$", filename, re.IGNORECASE)
    if m:
        return int(m.group(1)), m.group(2)
    return None


def scan_lakh(base_dir):
    """Scan the Lakh MIDI Dataset (lmd_full/).

    Each .mid file under lmd_full/ is a complete song.
    Files are organized in hash-prefix subdirectories (0-9, a-f).

    Returns list of scan results.
    """
    lmd_dir = Path(base_dir) / "lmd_full"
    if not lmd_dir.is_dir():
        return []

    results = []
    for path in sorted(lmd_dir.rglob("*.mid")):
        try:
            result = scan_midi_file(path)
            result["source_paths"] = [str(path)]
            result["dataset"] = "lakh"
            results.append(result)
        except Exception as e:
            print(f"Warning: could not parse {path}: {e}")
    return results


def scan_aam(base_dir):
    """Scan the AAM (Audio-Aligned MIDI) dataset.

    Files are named {songID}_{Instrument}.mid. Each song ID may have
    multiple instrument files that should be grouped into one song result.
    Demo.mid files are excluded.

    Returns list of scan results (one per song).
    """
    aam_dir = Path(base_dir)
    if not aam_dir.is_dir():
        return []

    # Group files by song ID
    songs = {}
    for path in sorted(aam_dir.glob("*.mid")):
        parsed = _parse_aam_filename(path.name)
        if parsed is None:
            continue
        song_id, instrument_name = parsed
        if instrument_name == "Demo":
            continue
        songs.setdefault(song_id, []).append(path)

    results = []
    for song_id in sorted(songs):
        paths = songs[song_id]
        all_instruments = []
        tempos = []
        durations = []
        scan_results_per_file = []

        for path in paths:
            try:
                scan = scan_midi_file(path)
                scan_results_per_file.append(scan)
                all_instruments.extend(scan["instruments"])
                tempos.append(scan["tempo"])
                durations.append(scan["duration"])
            except Exception as e:
                print(f"Warning: could not parse {path}: {e}")

        if not scan_results_per_file:
            continue

        results.append({
            "path": str(paths[0].parent / f"{song_id:04d}"),
            "instruments": all_instruments,
            "tempo": tempos[0],
            "duration": max(durations),
            "source_paths": [str(p) for p in paths],
            "dataset": "aam",
        })

    return results


def scan_slakh(base_dir, splits=None):
    """Scan the Slakh2100 dataset.

    Iterates train/validation/test splits. Uses all_src.mid (combined
    multitrack) when present, falls back to merging individual MIDI/S*.mid
    stems.

    Args:
        base_dir: path to the slakh2100_flac_redux directory
        splits: list of splits to scan (default: ["train", "validation", "test"])

    Returns list of scan results.
    """
    slakh_dir = Path(base_dir)
    if not slakh_dir.is_dir():
        return []

    if splits is None:
        splits = ["train", "validation", "test"]

    results = []
    for split in splits:
        split_dir = slakh_dir / split
        if not split_dir.is_dir():
            continue

        for track_dir in sorted(split_dir.iterdir()):
            if not track_dir.is_dir():
                continue

            all_src = track_dir / "all_src.mid"
            if all_src.exists():
                try:
                    result = scan_midi_file(all_src)
                    result["source_paths"] = [str(all_src)]
                    result["dataset"] = "slakh"
                    result["split"] = split
                    results.append(result)
                except Exception as e:
                    print(f"Warning: could not parse {all_src}: {e}")
            else:
                # Fall back to merging individual stems
                midi_dir = track_dir / "MIDI"
                if not midi_dir.is_dir():
                    continue

                stems = sorted(midi_dir.glob("S*.mid"))
                if not stems:
                    continue

                all_instruments = []
                tempos = []
                durations = []
                valid_stems = []

                for stem in stems:
                    try:
                        scan = scan_midi_file(stem)
                        all_instruments.extend(scan["instruments"])
                        tempos.append(scan["tempo"])
                        durations.append(scan["duration"])
                        valid_stems.append(stem)
                    except Exception as e:
                        print(f"Warning: could not parse {stem}: {e}")

                if not valid_stems:
                    continue

                results.append({
                    "path": str(track_dir),
                    "instruments": all_instruments,
                    "tempo": tempos[0],
                    "duration": max(durations),
                    "source_paths": [str(s) for s in valid_stems],
                    "dataset": "slakh",
                    "split": split,
                })

    return results


def scan_datasets(data_dir, datasets=None):
    """Auto-detect and scan MIDI datasets under data_dir.

    Looks for known directory structures:
        - lakh:  data_dir/lakh_midi/lmd_full/
        - aam:   data_dir/aam_midi/
        - slakh: data_dir/slakh2100_midi/slakh2100_flac_redux/

    Args:
        data_dir: root data directory containing dataset subdirectories
        datasets: optional list of dataset names to scan (default: all detected)

    Returns combined list of scan results from all datasets.
    """
    data_dir = Path(data_dir)
    all_datasets = {
        "lakh": ("lakh_midi", lambda d: scan_lakh(d)),
        "aam": ("aam_midi", lambda d: scan_aam(d)),
        "slakh": (
            "slakh2100_midi/slakh2100_flac_redux",
            lambda d: scan_slakh(d),
        ),
    }

    results = []
    for name, (subdir, scanner) in all_datasets.items():
        if datasets is not None and name not in datasets:
            continue
        full_path = data_dir / subdir
        if full_path.is_dir():
            results.extend(scanner(full_path))

    return results
