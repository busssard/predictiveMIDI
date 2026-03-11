# PCJAM

A hierarchical predictive coding network that learns to generate instrument parts from multitrack MIDI — no backpropagation, only local prediction errors and Hebbian learning.

## Overview

This project explores whether predictive coding (PC) — a neuroscience theory of cortical computation — can learn musical structure. A hierarchical hourglass network computes top-down predictions, bottom-up prediction errors, and updates representations through iterative relaxation. Weights are updated either via local Hebbian outer products or via autodiff gradients of the free energy.

The network is trained on multitrack MIDI from three public datasets (~160K songs after quality filtering). Input instruments are clamped to the bottom layer, the network relaxes to minimize prediction error, and the output layer produces the target instrument's part.

A WebGL frontend visualizes the grid in real time: prediction errors map to colour, connection weights to gap size between cells.

**Status**: Research code. Architecture overhauled March 2026 — bounded predictions, quality pipeline, lax.scan optimization. Training and evaluation ongoing.

## Architecture

### Hierarchical hourglass

Five-layer hourglass: `[128 → 64 → 32 → 64 → 128]` with dense connections:

| Connection | Direction | Purpose |
|-----------|-----------|---------|
| **Prediction weights** | Top-down (layer i → i-1) | Generate predictions of lower-layer activity |
| **Skip connections** | Input → output (layer 0 → last) | Direct pathway for salient features |
| **Temporal weights** | Same layer, previous tick | Temporal continuity across musical time steps |

All predictions are bounded: `pred = 3 · tanh(W · tanh(x) + b)`, ensuring stable attractor dynamics during inference (Rao & Ballard 1999).

### Update rule

Each musical time step runs 32 relaxation steps. Per relaxation step, every layer simultaneously:

1. **Receives predictions** from the layer above (top-down), skip connections, and temporal context
2. **Computes error**: `e = r - prediction`
3. **Updates representation**: gradient descent on `sum(e²)` w.r.t. representations
4. **Updates weights** via one of:
   - **Hebbian** (default): rank-1 outer product `ΔW = e · x^T` with activation derivatives
   - **Autodiff**: `jax.grad(F)` where `F = sum(||e_i||²)` — exact free energy gradient

### Training

- MIDI sampled at 8 ticks/second into 128-dim piano roll vectors
- Batches randomly sample songs, time offsets, and input/target instrument splits
- Teacher forcing: output clamped to target on a fraction of ticks (annealed from 0.8 → 0.0)
- Quality filtering: tempo 30–300 BPM, duration ≤ 600s, top-12 instruments by note count
- Entire update compiled via `lax.scan` (ticks) + `lax.fori_loop` (relaxation) — 1 JIT dispatch per example

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

# Build corpus index with quality filters
uv run python manage.py build_corpus_index --midi-dir data/midi

# Train (Hebbian, 1000 steps)
uv run python manage.py train_hierarchical \
    --midi-dir data/midi \
    --num-steps 1000 \
    --batch-size 16 \
    --weight-update hebbian
```

### Full pipeline

A single script runs the entire pipeline: rebuild corpus → train Hebbian → train autodiff → plot → compare.

```bash
bash scripts/run_all.sh
```

Individual steps are available as separate scripts in `scripts/01_rebuild_corpus.sh` through `scripts/06_compare.sh`.

## Datasets

Three CC-BY 4.0 licensed MIDI datasets:

| Dataset | Songs | Size |
|---------|-------|------|
| [Lakh MIDI](https://colinraffel.com/projects/lmd/) [1] | ~178K | ~6 GB |
| [AAM](https://zenodo.org/records/5794629) [2] | ~3K | ~120 MB |
| [Slakh2100](https://zenodo.org/records/4599666) [3] | ~2.1K | varies |

All three use General MIDI program numbers, providing a unified instrument vocabulary.

```bash
uv run python download_midi_datasets.py aam          # just AAM
uv run python download_midi_datasets.py aam slakh    # AAM + Slakh
uv run python download_midi_datasets.py              # all three
```

A pre-built corpus index is included as split files (`data/corpus_index_1.json` + `data/corpus_index_2.json`). Rebuild after downloading more data:

```bash
uv run python manage.py build_corpus_index \
    --midi-dir data/midi \
    --refilter \
    --max-instruments 12 \
    --min-notes-per-instrument 10 \
    --max-duration 600 \
    --min-tempo 30 \
    --max-tempo 300
```

The `--refilter` flag re-applies quality filters to existing entries without re-scanning. Index building is resumable (Ctrl+C safe with incremental saves).

## Training

### Hierarchical model (current)

```bash
uv run python manage.py train_hierarchical \
    --midi-dir data/midi \
    --num-steps 1000 \
    --batch-size 16 \
    --layer-sizes 128 64 32 64 128 \
    --relaxation-steps 32 \
    --teacher-forcing 0.8 \
    --lr 0.01 \
    --lr-w 0.001 \
    --weight-update hebbian \
    --checkpoint-dir runs/my_run/checkpoints \
    --checkpoint-every 100 \
    --metrics-dir runs/my_run
```

| Flag | Default | Description |
|------|---------|-------------|
| `--midi-dir` | `data/midi` | Path to MIDI datasets |
| `--num-steps` | 1000 | Total training steps |
| `--batch-size` | 16 | Songs per step |
| `--layer-sizes` | 128 64 32 64 128 | Hourglass layer dimensions |
| `--relaxation-steps` | 32 | PC relaxation iterations per tick |
| `--teacher-forcing` | 0.8 | Fraction of ticks with output clamped |
| `--lr` | 0.01 | Representation learning rate |
| `--lr-w` | 0.001 | Weight learning rate |
| `--weight-update` | hebbian | `hebbian` or `autodiff` |
| `--checkpoint-dir` | `checkpoints_hierarchical` | Checkpoint output directory |
| `--checkpoint-every` | 50 | Steps between checkpoints |
| `--metrics-dir` | (same as checkpoint-dir) | JSONL metrics output directory |

### Flat grid (legacy)

```bash
uv run python manage.py train --num-steps 1000 --batch-size 16
```

### Monitoring

```bash
# View live training status
bash scripts/04_monitor.sh runs/my_run

# Plot loss curves from metrics
uv run python scripts/plot_metrics.py runs/my_run/metrics.jsonl runs/my_run
```

### Dashboard

A web dashboard shows live loss curves and training state:

```bash
uv run python manage.py runserver
```

## Inference

```bash
# Generate MIDI from a trained checkpoint
uv run python manage.py jam --checkpoint runs/my_run/checkpoints/final
```

Auto-detects flat vs hierarchical architecture from checkpoint metadata.

## Tests

```bash
uv run python -m pytest training/tests/ corpus/tests/ -v   # ~230 tests
```

## Project structure

```
corpus/                         MIDI data pipeline
    services/
        scanner.py                  MIDI file scanner
        dataset_scanner.py          Dataset-aware scanners (Lakh, AAM, Slakh)
        batch_generator.py          Training batch generation (quality_threshold)
        vocabulary.py               General MIDI instrument clustering
        curriculum.py               Curriculum scheduler
        prefetch.py                 Background batch prefetching
    management/commands/
        build_corpus_index.py       Corpus index with quality filters + --refilter
    tests/
        test_quality.py             Quality filtering tests (20 tests)

training/                       Grid training engine
    engine/
        hierarchical_grid.py        HierarchicalGridState, create_hierarchical_grid()
        hierarchical_update.py      Hebbian relaxation step (bounded predictions)
        hierarchical_update_autodiff.py  Autodiff relaxation step (jax.grad)
        hierarchical_trainer.py     Training loop (lax.scan + fori_loop)
        hierarchical_inference.py   Inference / jam mode
        metrics_logger.py           JSONL metrics logger
        export.py                   Model export (flat + hierarchical)
        grid.py                     Flat grid state (legacy)
        update_rule.py              Flat PC update (legacy)
        trainer.py                  Flat training loop (legacy)
    management/commands/
        train_hierarchical.py       Hierarchical training CLI
        train.py                    Flat training CLI (legacy)
        jam.py                      Inference CLI (auto-detects architecture)
        diagnose.py                 Signal propagation diagnostics
    tests/                          ~210 tests
    views.py                        REST API (checkpoint list + export)

frontend/                       WebGL visualization + jam mode
    dashboard.html                  Training dashboard
    js/                             Grid renderer, loss chart, MIDI I/O
    shaders/                        PC compute + rendering shaders

scripts/                        Pipeline scripts
    run_all.sh                      Full pipeline (corpus → train → plot → compare)
    01_rebuild_corpus.sh            Rebuild corpus index with quality filters
    02_train_hebbian.sh             Train with Hebbian weight updates
    03_train_autodiff.sh            Train with autodiff weight updates
    04_monitor.sh                   Monitor running training
    05_plot.sh                      Plot metrics from JSONL
    06_compare.sh                   Compare Hebbian vs autodiff results
    plot_metrics.py                 Matplotlib plotting from metrics JSONL

data/
    corpus_index.json               Song metadata index (~160K songs)
    corpus_index_1.json             Pre-built index split (for GitHub)
    corpus_index_2.json             (merged automatically at load time)
    midi/                           Downloaded datasets (not in repo)
```

## References

[1] C. Raffel. "Learning-Based Methods for Comparing Sequences, with Applications to Audio-to-MIDI Alignment and Matching." PhD Thesis, Columbia University, 2016.

[2] M. Perez Zarazaga, et al. "AAM: Artificial Audio Multitracks Dataset." Zenodo, 2022.

[3] E. Manilow, G. Wichern, P. Seetharaman, J. Le Roux. "Cutting Music Source Separation Some Slakh." ICASSP, 2019.

[4] R. P. Rao and D. H. Ballard. "Predictive coding in the visual cortex." Nature Neuroscience, 2(1):79-87, 1999.

[5] K. Friston. "A theory of cortical responses." Phil. Trans. R. Soc. B, 360(1456):815-836, 2005.

[6] B. Millidge, A. Tschantz, C. L. Buckley. "Predictive Coding Approximates Backprop along Arbitrary Computation Graphs." NeurIPS, 2022.

[7] T. Salvatori, et al. "Incremental Predictive Coding: A Parallel and Fully Automatic Learning Algorithm." NeurIPS Workshop, 2023.

## License

Code is MIT licensed. See [LICENSE](LICENSE).

The MIDI datasets are CC-BY 4.0 licensed by their respective authors.
