#!/usr/bin/env python3
"""
Download multitrack MIDI datasets for predictive coding experiment.

Datasets:
  1. Lakh MIDI Dataset (LMD) — 176K multitrack MIDI files, ~1.65 GB
  2. AAM (Artificial Audio Multitracks) — 3,000 multitrack MIDIs, ~10 MB (MIDI-only)
  3. Slakh2100 — 2,100 songs with per-instrument MIDI (MIDI-only extraction)

All datasets are CC-BY 4.0 licensed.
"""

import os
import sys
import subprocess
import tarfile
import zipfile
import shutil
import json
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

BASE_DIR = Path(__file__).parent / "data" / "midi"

DATASETS = {
    "lakh": {
        "name": "Lakh MIDI Dataset (Full)",
        "url": "http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz",
        "dest": "lakh_midi",
        "description": "176,581 multitrack MIDI files",
    },
    "aam": {
        "name": "AAM MIDI (Artificial Audio Multitracks)",
        "urls": [],  # populated dynamically from Zenodo API
        "zenodo_record": "5794629",
        "dest": "aam_midi",
        "description": "3,000 multitrack MIDIs with annotations",
    },
    "slakh": {
        "name": "Slakh2100 (MIDI only)",
        "zenodo_record": "4599666",
        "dest": "slakh2100_midi",
        "description": "2,100 songs with per-instrument MIDI tracks",
    },
}


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def download_with_wget(url: str, dest_path: Path):
    """Download a file using wget with progress bar and resume support."""
    cmd = [
        "wget",
        "--continue",
        "--show-progress",
        "--progress=bar:force:noscroll",
        "-O", str(dest_path),
        url,
    ]
    print(f"  Downloading: {url}")
    print(f"  To: {dest_path}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  ERROR: wget failed with code {result.returncode}")
        return False
    return True


def get_zenodo_files(record_id: str) -> list[dict]:
    """Fetch file listing from Zenodo API."""
    api_url = f"https://zenodo.org/api/records/{record_id}"
    print(f"  Fetching file list from Zenodo record {record_id}...")
    req = Request(api_url, headers={"Accept": "application/json"})
    with urlopen(req) as resp:
        data = json.loads(resp.read().decode())
    return data.get("files", [])


def extract_tar_gz(archive: Path, dest: Path):
    """Extract a .tar.gz archive."""
    print(f"  Extracting {archive.name}...")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(path=dest)
    print(f"  Extracted to {dest}")


def extract_zip(archive: Path, dest: Path):
    """Extract a .zip archive."""
    print(f"  Extracting {archive.name}...")
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(path=dest)
    print(f"  Extracted to {dest}")


def extract_tar(archive: Path, dest: Path):
    """Extract a .tar archive (uncompressed or any compression)."""
    print(f"  Extracting {archive.name}...")
    with tarfile.open(archive) as tar:
        tar.extractall(path=dest)
    print(f"  Extracted to {dest}")


def extract_midi_from_tar(archive: Path, dest: Path):
    """Extract only .mid/.midi files from a tar archive."""
    print(f"  Extracting MIDI files only from {archive.name}...")
    count = 0
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if member.name.lower().endswith((".mid", ".midi")) and member.isfile():
                # Preserve directory structure under dest
                tar.extract(member, path=dest)
                count += 1
                if count % 500 == 0:
                    print(f"    Extracted {count} MIDI files...")
    print(f"  Extracted {count} MIDI files to {dest}")
    return count


def download_lakh():
    """Download and extract Lakh MIDI Dataset."""
    info = DATASETS["lakh"]
    dest = BASE_DIR / info["dest"]
    archive = BASE_DIR / "lmd_full.tar.gz"

    if dest.exists() and any(dest.rglob("*.mid")):
        count = sum(1 for _ in dest.rglob("*.mid"))
        print(f"  Already downloaded ({count} MIDI files found). Skipping.")
        return True

    ensure_dir(BASE_DIR)

    if not archive.exists() or archive.stat().st_size < 1_000_000:
        if not download_with_wget(info["url"], archive):
            return False

    ensure_dir(dest)
    extract_midi_from_tar(archive, dest)

    # Clean up archive
    print(f"  Removing archive {archive.name} to save space...")
    archive.unlink()
    return True


def download_aam():
    """Download AAM MIDI-only files from Zenodo."""
    info = DATASETS["aam"]
    dest = BASE_DIR / info["dest"]

    if dest.exists() and any(dest.rglob("*.mid")):
        count = sum(1 for _ in dest.rglob("*.mid"))
        print(f"  Already downloaded ({count} MIDI files found). Skipping.")
        return True

    ensure_dir(dest)

    files = get_zenodo_files(info["zenodo_record"])
    midi_files = [f for f in files if "midi" in f["key"].lower() or "mid" in f["key"].lower()]

    if not midi_files:
        # Fallback: look for small zip files that likely contain MIDI
        midi_files = [
            f for f in files
            if f["key"].lower().endswith(".zip") and f["size"] < 50_000_000
            and ("midi" in f["key"].lower() or "mid" in f["key"].lower())
        ]

    if not midi_files:
        # Show all files so we can pick the right ones
        print("  Available files in Zenodo record:")
        for f in sorted(files, key=lambda x: x["size"]):
            size_mb = f["size"] / 1_000_000
            print(f"    {f['key']} ({size_mb:.1f} MB)")
        # Download small files that might contain MIDI
        midi_files = [f for f in files if f["size"] < 50_000_000]
        print(f"  Downloading {len(midi_files)} small files that may contain MIDI...")

    for f in midi_files:
        url = f["links"]["self"]
        fname = f["key"]
        archive = BASE_DIR / fname

        if not archive.exists():
            if not download_with_wget(url, archive):
                continue

        if fname.endswith(".zip"):
            extract_zip(archive, dest)
            archive.unlink()
        elif fname.endswith(".tar.gz") or fname.endswith(".tgz"):
            extract_tar_gz(archive, dest)
            archive.unlink()
        elif fname.endswith(".tar"):
            extract_tar(archive, dest)
            archive.unlink()
        elif fname.lower().endswith((".mid", ".midi")):
            shutil.move(str(archive), str(dest / fname))

    count = sum(1 for _ in dest.rglob("*.mid")) + sum(1 for _ in dest.rglob("*.midi"))
    print(f"  Total: {count} MIDI files downloaded.")
    return True


def download_slakh():
    """Download Slakh2100 MIDI-only files.

    The full Slakh2100 archive is ~104 GB (includes audio).
    We try to download only MIDI files by streaming the tar and extracting .mid files.
    If that's not feasible, we download the full archive and extract MIDI.
    """
    info = DATASETS["slakh"]
    dest = BASE_DIR / info["dest"]

    if dest.exists() and any(dest.rglob("*.mid")):
        count = sum(1 for _ in dest.rglob("*.mid"))
        print(f"  Already downloaded ({count} MIDI files found). Skipping.")
        return True

    ensure_dir(dest)

    files = get_zenodo_files(info["zenodo_record"])

    print("  Available files in Zenodo record:")
    for f in sorted(files, key=lambda x: x["size"]):
        size_gb = f["size"] / 1_000_000_000
        size_mb = f["size"] / 1_000_000
        if size_gb >= 1:
            print(f"    {f['key']} ({size_gb:.1f} GB)")
        else:
            print(f"    {f['key']} ({size_mb:.1f} MB)")

    # Look for MIDI-only archives first
    midi_only = [f for f in files if "midi" in f["key"].lower() and f["size"] < 1_000_000_000]
    if midi_only:
        print(f"  Found MIDI-only archive(s)!")
        for f in midi_only:
            url = f["links"]["self"]
            archive = BASE_DIR / f["key"]
            if not download_with_wget(url, archive):
                continue
            if archive.name.endswith(".tar.gz") or archive.name.endswith(".tgz"):
                extract_tar_gz(archive, dest)
            elif archive.name.endswith(".zip"):
                extract_zip(archive, dest)
            elif archive.name.endswith(".tar"):
                extract_tar(archive, dest)
            archive.unlink()
    else:
        # Full archive — download and extract only MIDI
        # Find the flac_redux (smaller) or the main archive
        candidates = [f for f in files if f["key"].endswith((".tar.gz", ".tar"))]
        candidates.sort(key=lambda x: x["size"])

        if not candidates:
            print("  ERROR: No downloadable archives found.")
            return False

        # Pick smallest archive
        chosen = candidates[0]
        size_gb = chosen["size"] / 1_000_000_000
        print(f"\n  WARNING: No MIDI-only archive available.")
        print(f"  Smallest archive: {chosen['key']} ({size_gb:.1f} GB)")
        print(f"  This is a large download. We'll extract only .mid files.")

        resp = input(f"  Proceed with downloading {size_gb:.1f} GB? [y/N]: ").strip().lower()
        if resp != "y":
            print("  Skipping Slakh2100.")
            return False

        archive = BASE_DIR / chosen["key"]
        if not download_with_wget(chosen["links"]["self"], archive):
            return False

        extract_midi_from_tar(archive, dest)
        print(f"  Removing archive to save space...")
        archive.unlink()

    count = sum(1 for _ in dest.rglob("*.mid")) + sum(1 for _ in dest.rglob("*.midi"))
    print(f"  Total: {count} MIDI files downloaded.")
    return True


def count_midi_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.rglob("*.mid")) + sum(1 for _ in path.rglob("*.midi"))


def main():
    print("=" * 60)
    print("Multitrack MIDI Dataset Downloader")
    print("=" * 60)
    print(f"Download directory: {BASE_DIR}")
    print()

    ensure_dir(BASE_DIR)

    # Select datasets
    if len(sys.argv) > 1:
        selected = sys.argv[1:]
    else:
        selected = ["lakh", "aam", "slakh"]

    results = {}

    for name in selected:
        if name not in DATASETS:
            print(f"Unknown dataset: {name}")
            print(f"Available: {', '.join(DATASETS.keys())}")
            continue

        info = DATASETS[name]
        print(f"\n{'─' * 60}")
        print(f"[{name.upper()}] {info['name']}")
        print(f"  {info['description']}")
        print(f"{'─' * 60}")

        try:
            success = {"lakh": download_lakh, "aam": download_aam, "slakh": download_slakh}[name]()
            dest = BASE_DIR / info["dest"]
            count = count_midi_files(dest)
            results[name] = {"success": success, "count": count}
        except (URLError, HTTPError) as e:
            print(f"  ERROR: Network error: {e}")
            results[name] = {"success": False, "count": 0}
        except Exception as e:
            print(f"  ERROR: {e}")
            results[name] = {"success": False, "count": 0}

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    total = 0
    for name, res in results.items():
        status = "OK" if res["success"] else "FAILED"
        print(f"  [{status}] {DATASETS[name]['name']}: {res['count']} MIDI files")
        total += res["count"]
    print(f"\n  Total: {total} MIDI files in {BASE_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
