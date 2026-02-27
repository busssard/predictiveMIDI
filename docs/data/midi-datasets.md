# MIDI Datasets

Three MIDI datasets stored in `data/midi/`. Each uses a different structure but all are consumed through a unified scanner API.

## Datasets

### Lakh MIDI Dataset (LMD)
- **Path**: `data/midi/lakh_midi/lmd_full/`
- **Size**: ~178K files, 6 GB
- **Structure**: Flat files in 16 hex-prefix subdirectories (0–f)
  ```
  lmd_full/
    0/ abc123.mid
    a/ def456.mid
    ...
  ```
- **Format**: Each `.mid` file is a complete multi-instrument song using GM standard programs. Channel 10 = drums.
- **Source**: https://colinraffel.com/projects/lmd/

### AAM (Audio-Aligned MIDI)
- **Path**: `data/midi/aam_midi/`
- **Size**: ~22K files across ~3K songs, 121 MB
- **Structure**: Flat directory, one instrument per file
  ```
  aam_midi/
    0001_Drums.mid
    0001_Piano.mid
    0001_Bass.mid
    0002_ElectricPiano.mid
    ...
  ```
- **Naming**: `{songID}_{InstrumentName}.mid` — numeric prefix groups files into songs
- **Multi-file convention**: All files sharing a song ID belong to the same song. The scanner groups them and creates a single result with `source_paths` listing all files.
- **Demo files**: `{songID}_Demo.mid` files are excluded by the scanner — they contain overview mixes, not individual instruments.

### Slakh2100
- **Path**: `data/midi/slakh2100_midi/slakh2100_flac_redux/`
- **Size**: ~24K files across ~2.1K songs, 279 MB
- **Structure**: Train/validation/test splits with per-track directories
  ```
  slakh2100_flac_redux/
    train/
      Track00001/
        all_src.mid        ← combined multitrack (preferred)
        MIDI/
          S00.mid          ← individual stems
          S01.mid
          ...
    validation/
    test/
    omitted/               ← not scanned by default
  ```
- **Loading priority**: `all_src.mid` is used when present (single file, all instruments). Falls back to merging `MIDI/S*.mid` stems.
- **Metadata**: Scan results include a `split` field (`"train"`, `"validation"`, `"test"`).

## Scan Result Format

All scanners produce dicts with the standard fields from `scan_midi_file()` plus dataset-specific extras:

```python
{
    "path": str,            # canonical path (file or directory)
    "instruments": [...],   # list of instrument dicts
    "tempo": float,         # BPM
    "duration": float,      # seconds
    "source_paths": [str],  # actual files to load (may be multiple)
    "dataset": str,         # "lakh", "aam", or "slakh"
    "split": str,           # (slakh only) "train"/"validation"/"test"
}
```

For multi-file songs (AAM, Slakh stem fallback), `source_paths` lists all files. `BatchGenerator` loads instruments from every file in the list.

## Usage

```python
from corpus.services.dataset_scanner import scan_datasets

# Scan all detected datasets
results = scan_datasets("data/midi")

# Scan specific datasets
results = scan_datasets("data/midi", datasets=["aam", "slakh"])

# Pass to BatchGenerator
from corpus.services.batch_generator import BatchGenerator
gen = BatchGenerator("data/midi", scan_results=results)
```

Individual scanners are also available:

```python
from corpus.services.dataset_scanner import scan_lakh, scan_aam, scan_slakh

lakh = scan_lakh("data/midi/lakh_midi")
aam = scan_aam("data/midi/aam_midi")
slakh = scan_slakh("data/midi/slakh2100_midi/slakh2100_flac_redux")
```

## GM Standard

All datasets use General MIDI (GM) program numbers 0–127 for instrument assignment. The `vocabulary.py` module maps these to categories (piano, bass, strings, etc.). Channel 10/drums are identified by `is_drum=True`.
