# PCJAM

A predictive coding network on a 128x128 grid that learns to generate instrument parts from multitrack MIDI — no backpropagation, only local prediction errors.

## Overview

This project explores whether predictive coding (PC) — a neuroscience theory of cortical computation — can learn musical structure when arranged on a 2D grid rather than the usual layered hierarchy. Each cell predicts its spatial neighbours, computes prediction errors, and updates through iterative relaxation using incremental PC (iPC) with concurrent weight updates.

The grid is trained on multitrack MIDI from three public datasets (~180K songs total). Input instruments are clamped to the left edge, a one-hot instrument selector conditions the top edge, and the grid learns to produce the target instrument's part on the right edge. Interior cells self-organise to connect input to output.

A WebGL frontend visualizes the grid in real time: prediction errors map to colour, connection weights to gap size between cells.

**Status**: Early-stage research code. The training pipeline and WebGL visualization are functional. Musical output quality is under active experimentation.

## Method

### Grid structure

128x128 cells, each storing four values:

| Value | Role |
|-------|------|
| **r** | Representation — the cell's current activation |
| **e** | Prediction error — mismatch between neighbours' predictions and r |
| **s** | Leaky integrator — slow-decaying memory (learned decay rate alpha) |
| **h** | Recurrent state — active temporal memory (learned gain beta) |

Cells connect to their 4 spatial neighbours (up/down/left/right) with learned weights.

### Update rule

Each musical time step runs 128 relaxation steps. Per relaxation step, every cell simultaneously:

1. **Predicts** each neighbour: `prediction = weight * tanh(r)`
2. **Computes error**: `e = r - sum(incoming_predictions)`
3. **Updates memory**: `s = alpha * s + (1-alpha) * r` and `h = beta * tanh(h_prev)`
4. **Updates representation**: `r += lr * (-e + sum(neighbour_errors * weights) + h + s - r)`
5. **Updates weights** (iPC): `w += lr_w * e * tanh(r_neighbour)` — weights learn at every relaxation step, not just at convergence

### Training

- MIDI is sampled at 8 ticks/second into 128-dim piano roll vectors (one row per pitch)
- Each batch randomly selects songs, time offsets, and input/target instrument splits
- Curriculum starts with short 16-tick snippets (~2 bars) and advances to longer sequences as error drops
- Entire update rule is vectorized with `jax.vmap` (batch) and `lax.scan` (ticks/relaxation)

## Prerequisites

- Python >= 3.10
- [uv](https://docs.astral.sh/uv/) package manager
- CUDA 12 + compatible GPU (for training at reasonable speed; CPU works but is slow)

## Quick start

```bash
git clone https://github.com/busssard/predictiveMIDI.git
cd predictiveMIDI

# Install with GPU support
uv sync --extra cuda --extra dev

# Or CPU-only
uv sync --extra cpu --extra dev

# Download a dataset (AAM is smallest, ~120 MB)
uv run python download_midi_datasets.py aam

# Build corpus index (or use the included pre-built one with ~4,700 songs)
uv run python manage.py build_corpus_index

# Train
uv run python manage.py train --num-steps 100 --batch-size 8
```

## Datasets

Three CC-BY 4.0 licensed MIDI datasets:

| Dataset | Songs | Structure | Size |
|---------|-------|-----------|------|
| [Lakh MIDI](https://colinraffel.com/projects/lmd/) [1] | ~178K | One multi-instrument file per song | ~6 GB |
| [AAM](https://zenodo.org/records/5794629) [2] | ~3K | One file per instrument, grouped by song ID | ~120 MB |
| [Slakh2100](https://zenodo.org/records/4599666) [3] | ~2.1K | Per-song directories with stems + combined file | varies |

All three use General MIDI program numbers, providing a unified instrument vocabulary.

```bash
uv run python download_midi_datasets.py aam          # just AAM
uv run python download_midi_datasets.py aam slakh    # AAM + Slakh
uv run python download_midi_datasets.py              # all three
```

A pre-built corpus index (`data/corpus_index.json`) with ~4,700 songs (AAM + Slakh) is included so the project structure can be explored without downloading the full datasets. Rebuild after downloading more data with `uv run python manage.py build_corpus_index`.

## Training

```bash
uv run python manage.py train \
    --num-steps 1000 \
    --batch-size 16 \
    --grid-size 128 \
    --relaxation-steps 128 \
    --fs 8.0 \
    --checkpoint-every 50 \
    --datasets aam slakh
```

| Flag | Default | Description |
|------|---------|-------------|
| `--num-steps` | 1000 | Total training steps |
| `--batch-size` | 16 | Songs per step |
| `--grid-size` | 128 | Grid dimension (NxN) |
| `--relaxation-steps` | 128 | PC relaxation iterations per tick |
| `--fs` | 8.0 | Sampling rate in ticks/second |
| `--checkpoint-every` | 50 | Checkpoint interval |
| `--datasets` | all detected | Which datasets to use |

Checkpoints save to `checkpoints/`. The model auto-exports to `frontend/model/` on completion.

### Dashboard

A web dashboard shows live loss curves, training phase, and grid state. Start the Django server and open `frontend/dashboard.html`:

```bash
uv run python manage.py runserver
```

## Tests

```bash
uv run python -m pytest           # 91 tests
uv run python -m pytest -v        # verbose
```

## Project structure

```
corpus/                     MIDI data pipeline
    services/
        scanner.py              MIDI file scanner
        dataset_scanner.py      Dataset-aware scanners (Lakh, AAM, Slakh)
        batch_generator.py      Training batch generation
        vocabulary.py           General MIDI instrument clustering
        curriculum.py           Curriculum scheduler
        prefetch.py             Background batch prefetching
    management/commands/
        build_corpus_index.py   Corpus index builder

training/                   Grid training engine
    engine/
        grid.py                 Grid state initialization (JAX)
        update_rule.py          PC relaxation step + iPC weight update
        trainer.py              Vectorized training loop
        export.py               Model export for WebGL

frontend/                   WebGL visualization + jam mode
    dashboard.html              Training dashboard
    js/                         Grid renderer, loss chart, MIDI I/O
    shaders/                    PC compute + rendering shaders

data/
    corpus_index.json           Pre-built song metadata index
    midi/                       Downloaded datasets (not in repo)

docs/                       Design documents and dataset documentation
```

## References

[1] C. Raffel. "Learning-Based Methods for Comparing Sequences, with Applications to Audio-to-MIDI Alignment and Matching." PhD Thesis, Columbia University, 2016. https://colinraffel.com/projects/lmd/

[2] M. Perez Zarazaga, et al. "AAM: Artificial Audio Multitracks Dataset." Zenodo, 2022. https://doi.org/10.5281/zenodo.5794629

[3] E. Manilow, G. Wichern, P. Seetharaman, J. Le Roux. "Cutting Music Source Separation Some Slakh: A Dataset to Study the Impact of Training Data and Evaluation Methods." ICASSP, 2019. https://doi.org/10.5281/zenodo.4599666

[4] R. P. Rao and D. H. Ballard. "Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects." Nature Neuroscience, 2(1):79-87, 1999.

[5] K. Friston. "A theory of cortical responses." Philosophical Transactions of the Royal Society B, 360(1456):815-836, 2005.

[6] T. Salvatori, et al. "Incremental Predictive Coding: A Parallel and Fully Automatic Learning Algorithm." NeurIPS Workshop on Associative Memory & Hopfield Networks, 2023.

## License

Code is MIT licensed. See [LICENSE](LICENSE).

The MIDI datasets are CC-BY 4.0 licensed by their respective authors.
