# Hierarchical PC Architecture Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the flat 128x16 grid with a hierarchical encoder-decoder PC network that can propagate signals from input to output and generate music.

**Architecture:** A layered PC network where each layer predicts the layer below. Layers have decreasing spatial resolution (128→64→32→64→128) forming an hourglass. Inter-layer connections use dense matrices instead of scalar neighbor weights. Skip connections from encoder to decoder preserve pitch resolution (U-Net style). Training uses scheduled output unclamping to learn generation, not just interpolation.

**Tech Stack:** JAX (jit, vmap, lax.scan), NumPy, Django management commands, pytest

---

## Background: Why This Change

Diagnostics on the current step_250 checkpoint show:
- Signal propagation ratio: 0.007 (signal dies by column 6 of 16)
- Output column: sigmoid(~0) = 0.5008 (no meaningful output)
- 67.7 notes/tick during inference (random noise)
- Flat 4-neighbor connectivity cannot carry information across 16 hops

The literature (Friston 2005, Millidge 2021, Qi 2025) says:
- PC networks MUST be hierarchical — higher layers predict lower layers
- Errors flow UP (low→high), predictions flow DOWN (high→low)
- Without hierarchy, the grid finds trivial minima (copy-neighbor)

## Design: HierarchicalPC

```
Layer 0 (input):  128 neurons — clamped to input piano roll
Layer 1:           64 neurons — encoder
Layer 2:           32 neurons — bottleneck/latent
Layer 3:           64 neurons — decoder
Layer 4 (output): 128 neurons — free during inference, clamped during training

Inter-layer: dense prediction matrices W_pred (H_high, H_low)
Skip connections: encoder layer i → decoder layer (N-1-i)
```

Each layer stores: `x` (representation), `e` (prediction error)
Each inter-layer connection stores: `W_pred` (prediction matrix)

**PC dynamics per relaxation step:**
```
pred_l = activation(W_pred[l+1→l] @ x[l+1])      # top-down prediction
e_l = x_l - pred_l                                  # prediction error
x[l+1] += lr * W_pred[l+1→l].T @ e_l               # error drives higher layer
W_pred[l+1→l] += lr_w * e_l @ x[l+1].T             # Hebbian weight update
```

---

### Task 1: Hierarchical Grid State

**Files:**
- Create: `training/engine/hierarchical_grid.py`
- Test: `training/tests/test_hierarchical_grid.py`

**Step 1: Write the failing test**

```python
# training/tests/test_hierarchical_grid.py
import jax.numpy as jnp
import jax
from training.engine.hierarchical_grid import HierarchicalGridState, create_hierarchical_grid


class TestCreateHierarchicalGrid:
    def test_default_layer_sizes(self):
        grid = create_hierarchical_grid()
        assert grid.layer_sizes == [128, 64, 32, 64, 128]

    def test_representations_shapes(self):
        grid = create_hierarchical_grid()
        assert grid.representations[0].shape == (128,)
        assert grid.representations[1].shape == (64,)
        assert grid.representations[2].shape == (32,)
        assert grid.representations[3].shape == (64,)
        assert grid.representations[4].shape == (128,)

    def test_prediction_weight_shapes(self):
        """W_pred[i] predicts layer i from layer i+1."""
        grid = create_hierarchical_grid()
        # W_pred[0]: layer 1 (64) predicts layer 0 (128) → shape (128, 64)
        assert grid.prediction_weights[0].shape == (128, 64)
        assert grid.prediction_weights[1].shape == (64, 32)
        assert grid.prediction_weights[2].shape == (32, 64)
        assert grid.prediction_weights[3].shape == (64, 128)

    def test_skip_connection_shapes(self):
        """Skip from encoder layer i to decoder layer (N-1-i)."""
        grid = create_hierarchical_grid()
        # Skip 0→4: both 128, so W_skip shape (128, 128)
        assert grid.skip_weights[0].shape == (128, 128)
        # Skip 1→3: both 64, so W_skip shape (64, 64)
        assert grid.skip_weights[1].shape == (64, 64)

    def test_custom_layer_sizes(self):
        grid = create_hierarchical_grid(layer_sizes=[32, 16, 32])
        assert grid.layer_sizes == [32, 16, 32]
        assert len(grid.prediction_weights) == 2

    def test_conditioning_vector_size(self):
        grid = create_hierarchical_grid(num_instruments=16)
        assert grid.conditioning_size == 16

    def test_temporal_weights_shapes(self):
        grid = create_hierarchical_grid()
        for i, size in enumerate(grid.layer_sizes):
            assert grid.temporal_weights[i].shape == (size, size)

    def test_all_arrays_are_finite(self):
        grid = create_hierarchical_grid()
        for r in grid.representations:
            assert jnp.all(jnp.isfinite(r))
        for w in grid.prediction_weights:
            assert jnp.all(jnp.isfinite(w))
```

**Step 2: Run test to verify it fails**

Run: `uv run python -m pytest training/tests/test_hierarchical_grid.py -v`
Expected: ImportError — module not found

**Step 3: Write minimal implementation**

```python
# training/engine/hierarchical_grid.py
from dataclasses import dataclass, field
from typing import List, Optional
import jax
import jax.numpy as jnp


@dataclass
class HierarchicalGridState:
    """Hierarchical PC network state.

    Layer 0 = input (clamped), Layer N-1 = output (free during inference).
    Predictions flow outward from bottleneck, errors flow inward.

    representations: list of (H_l,) arrays — one per layer
    errors: list of (H_l,) arrays — prediction errors per layer
    prediction_weights: list of (H_low, H_high) arrays — W_pred[i] predicts layer i from i+1
    skip_weights: list of (H_dec, H_enc) arrays — skip from encoder to decoder
    temporal_weights: list of (H_l, H_l) arrays — temporal prediction per layer
    temporal_state: list of (H_l,) arrays — previous tick's representation
    layer_sizes: list of ints
    conditioning_size: int
    lr: float
    lr_w: float
    alpha: float — temporal decay
    lambda_sparse: float — L1 sparsity
    """
    representations: List[jnp.ndarray]
    errors: List[jnp.ndarray]
    prediction_weights: List[jnp.ndarray]
    skip_weights: List[jnp.ndarray]
    temporal_weights: List[jnp.ndarray]
    temporal_state: List[jnp.ndarray]
    layer_sizes: List[int]
    conditioning_size: int = 16
    lr: float = 0.01
    lr_w: float = 0.001
    alpha: float = 0.8
    lambda_sparse: float = 0.005


def create_hierarchical_grid(
    layer_sizes=None,
    num_instruments=16,
    key=None,
    lr=0.01,
    lr_w=0.001,
    alpha=0.8,
    lambda_sparse=0.005,
):
    if layer_sizes is None:
        layer_sizes = [128, 64, 32, 64, 128]
    if key is None:
        key = jax.random.PRNGKey(42)

    num_layers = len(layer_sizes)

    representations = [jnp.zeros(s) for s in layer_sizes]
    errors = [jnp.zeros(s) for s in layer_sizes]
    temporal_state = [jnp.zeros(s) for s in layer_sizes]

    # Prediction weights: W_pred[i] has shape (layer_sizes[i], layer_sizes[i+1])
    # — layer i+1 predicts layer i
    prediction_weights = []
    for i in range(num_layers - 1):
        key, subkey = jax.random.split(key)
        h_low, h_high = layer_sizes[i], layer_sizes[i + 1]
        scale = 0.1 / jnp.sqrt(float(h_high))
        w = jax.random.normal(subkey, (h_low, h_high)) * scale
        prediction_weights.append(w)

    # Skip connections: encoder layer i → decoder layer (N-1-i)
    # Only for symmetric pairs (0↔N-1, 1↔N-2, etc.)
    skip_weights = []
    n_skip = num_layers // 2
    for i in range(n_skip):
        j = num_layers - 1 - i
        key, subkey = jax.random.split(key)
        h_enc, h_dec = layer_sizes[i], layer_sizes[j]
        scale = 0.01 / jnp.sqrt(float(h_enc))
        w = jax.random.normal(subkey, (h_dec, h_enc)) * scale
        skip_weights.append(w)

    # Temporal weights: predict current from previous tick
    temporal_weights = []
    for i in range(num_layers):
        key, subkey = jax.random.split(key)
        s = layer_sizes[i]
        scale = 0.1 / jnp.sqrt(float(s))
        w = jax.random.normal(subkey, (s, s)) * scale
        temporal_weights.append(w)

    return HierarchicalGridState(
        representations=representations,
        errors=errors,
        prediction_weights=prediction_weights,
        skip_weights=skip_weights,
        temporal_weights=temporal_weights,
        temporal_state=temporal_state,
        layer_sizes=layer_sizes,
        conditioning_size=num_instruments,
        lr=lr,
        lr_w=lr_w,
        alpha=alpha,
        lambda_sparse=lambda_sparse,
    )
```

**Step 4: Run test to verify it passes**

Run: `uv run python -m pytest training/tests/test_hierarchical_grid.py -v`
Expected: All 8 tests PASS

**Step 5: Commit**

```bash
git add training/engine/hierarchical_grid.py training/tests/test_hierarchical_grid.py
git commit -m "feat: add HierarchicalGridState dataclass and create_hierarchical_grid"
```

---

### Task 2: Hierarchical PC Update Rule

**Files:**
- Create: `training/engine/hierarchical_update.py`
- Test: `training/tests/test_hierarchical_update.py`

**Step 1: Write the failing test**

```python
# training/tests/test_hierarchical_update.py
import jax
import jax.numpy as jnp
from training.engine.hierarchical_grid import create_hierarchical_grid
from training.engine.hierarchical_update import hierarchical_relaxation_step


class TestHierarchicalRelaxationStep:
    def test_output_shapes_match_input(self):
        grid = create_hierarchical_grid(layer_sizes=[16, 8, 16])
        new_reps, new_errors, new_pred_w, new_skip_w = hierarchical_relaxation_step(
            grid.representations,
            grid.prediction_weights,
            grid.skip_weights,
            grid.temporal_weights,
            grid.temporal_state,
            grid.layer_sizes,
            lr=0.01, lr_w=0.001, lambda_sparse=0.005,
        )
        assert len(new_reps) == 3
        assert new_reps[0].shape == (16,)
        assert new_reps[1].shape == (8,)
        assert new_reps[2].shape == (16,)

    def test_all_outputs_finite(self):
        grid = create_hierarchical_grid(layer_sizes=[16, 8, 16])
        # Set some non-zero representations
        reps = [jnp.ones(s) * 0.5 for s in grid.layer_sizes]
        new_reps, new_errors, new_pred_w, new_skip_w = hierarchical_relaxation_step(
            reps,
            grid.prediction_weights,
            grid.skip_weights,
            grid.temporal_weights,
            grid.temporal_state,
            grid.layer_sizes,
            lr=0.01, lr_w=0.001, lambda_sparse=0.005,
        )
        for r in new_reps:
            assert jnp.all(jnp.isfinite(r))
        for e in new_errors:
            assert jnp.all(jnp.isfinite(e))

    def test_error_decreases_over_steps(self):
        """Running multiple relaxation steps should reduce total error."""
        grid = create_hierarchical_grid(layer_sizes=[16, 8, 16])
        # Initialize with different values to create prediction error
        reps = [jax.random.normal(jax.random.PRNGKey(i), (s,)) * 0.5
                for i, s in enumerate(grid.layer_sizes)]

        errors_over_time = []
        pred_w = grid.prediction_weights
        skip_w = grid.skip_weights
        for step in range(20):
            reps, errors, pred_w, skip_w = hierarchical_relaxation_step(
                reps, pred_w, skip_w,
                grid.temporal_weights, grid.temporal_state,
                grid.layer_sizes,
                lr=0.01, lr_w=0.001, lambda_sparse=0.0,
            )
            total_error = sum(jnp.mean(e ** 2) for e in errors)
            errors_over_time.append(float(total_error))

        # Error at step 19 should be less than step 0
        assert errors_over_time[-1] < errors_over_time[0]

    def test_clamped_layer_unchanged(self):
        """If we clamp layer 0, its representation shouldn't change."""
        grid = create_hierarchical_grid(layer_sizes=[16, 8, 16])
        clamp_val = jnp.ones(16) * 0.7
        reps = list(grid.representations)
        reps[0] = clamp_val

        new_reps, _, _, _ = hierarchical_relaxation_step(
            reps,
            grid.prediction_weights,
            grid.skip_weights,
            grid.temporal_weights,
            grid.temporal_state,
            grid.layer_sizes,
            lr=0.01, lr_w=0.001, lambda_sparse=0.005,
            clamp_mask={0: clamp_val},
        )
        assert jnp.allclose(new_reps[0], clamp_val)

    def test_prediction_weights_change(self):
        """Weights should update when there are errors."""
        grid = create_hierarchical_grid(layer_sizes=[16, 8, 16])
        reps = [jax.random.normal(jax.random.PRNGKey(i), (s,))
                for i, s in enumerate(grid.layer_sizes)]
        old_w = [w.copy() for w in grid.prediction_weights]

        _, _, new_pred_w, _ = hierarchical_relaxation_step(
            reps,
            grid.prediction_weights,
            grid.skip_weights,
            grid.temporal_weights,
            grid.temporal_state,
            grid.layer_sizes,
            lr=0.01, lr_w=0.01, lambda_sparse=0.0,
        )
        for old, new in zip(old_w, new_pred_w):
            assert not jnp.allclose(old, new)

    def test_is_jit_compilable(self):
        """The function should work under jax.jit."""
        grid = create_hierarchical_grid(layer_sizes=[16, 8, 16])

        @jax.jit
        def step(reps, pred_w, skip_w, temp_w, temp_s):
            return hierarchical_relaxation_step(
                reps, pred_w, skip_w, temp_w, temp_s,
                [16, 8, 16],
                lr=0.01, lr_w=0.001, lambda_sparse=0.005,
            )

        new_reps, _, _, _ = step(
            grid.representations, grid.prediction_weights,
            grid.skip_weights, grid.temporal_weights,
            grid.temporal_state,
        )
        assert new_reps[0].shape == (16,)
```

**Step 2: Run test to verify it fails**

Run: `uv run python -m pytest training/tests/test_hierarchical_update.py -v`
Expected: ImportError

**Step 3: Write implementation**

```python
# training/engine/hierarchical_update.py
"""Hierarchical PC update rule.

Predictions flow from higher layers to lower layers (top-down).
Errors flow from lower layers to higher layers (bottom-up).

Per relaxation step:
  For each adjacent layer pair (l, l+1):
    pred_l = tanh(W_pred[l] @ x[l+1])           # top-down prediction
    e_l = x[l] - pred_l - skip_pred_l            # prediction error
    x[l+1] += lr * W_pred[l].T @ e_l             # update higher layer
    W_pred[l] += lr_w * outer(e_l, x[l+1])       # Hebbian weight update
"""
import jax
import jax.numpy as jnp
from typing import Dict, List, Optional, Tuple


def hierarchical_relaxation_step(
    representations: List[jnp.ndarray],
    prediction_weights: List[jnp.ndarray],
    skip_weights: List[jnp.ndarray],
    temporal_weights: List[jnp.ndarray],
    temporal_state: List[jnp.ndarray],
    layer_sizes: List[int],
    lr: float = 0.01,
    lr_w: float = 0.001,
    lambda_sparse: float = 0.005,
    clamp_mask: Optional[Dict[int, jnp.ndarray]] = None,
) -> Tuple[List[jnp.ndarray], List[jnp.ndarray],
           List[jnp.ndarray], List[jnp.ndarray]]:
    """One relaxation step of the hierarchical PC network.

    Args:
        representations: list of (H_l,) per layer
        prediction_weights: list of (H_low, H_high) — W_pred[i] predicts layer i from i+1
        skip_weights: list of (H_dec, H_enc) — skip from encoder i to decoder N-1-i
        temporal_weights: list of (H_l, H_l) — temporal prediction per layer
        temporal_state: list of (H_l,) — previous tick representations
        layer_sizes: list of ints
        lr: representation learning rate
        lr_w: weight learning rate
        lambda_sparse: L1 sparsity penalty
        clamp_mask: dict {layer_idx: clamp_values} — layers to hold fixed

    Returns:
        new_representations, new_errors, new_prediction_weights, new_skip_weights
    """
    if clamp_mask is None:
        clamp_mask = {}

    num_layers = len(layer_sizes)
    activation_fn = jnp.tanh

    # Compute predictions and errors for each layer
    errors = [jnp.zeros(s) for s in layer_sizes]
    predictions = [jnp.zeros(s) for s in layer_sizes]

    # Top-down predictions: layer i+1 predicts layer i
    for i in range(num_layers - 1):
        x_high = activation_fn(representations[i + 1])
        pred = activation_fn(prediction_weights[i] @ x_high)
        predictions[i] = predictions[i] + pred

    # Skip connection predictions: encoder i predicts decoder (N-1-i)
    n_skip = len(skip_weights)
    for si in range(n_skip):
        enc_idx = si
        dec_idx = num_layers - 1 - si
        if enc_idx != dec_idx:  # skip if same layer (odd number of layers)
            x_enc = activation_fn(representations[enc_idx])
            skip_pred = skip_weights[si] @ x_enc
            predictions[dec_idx] = predictions[dec_idx] + skip_pred

    # Temporal predictions
    for i in range(num_layers):
        temp_pred = temporal_weights[i] @ activation_fn(temporal_state[i])
        predictions[i] = predictions[i] + temp_pred * 0.1  # mild temporal influence

    # Compute errors
    for i in range(num_layers):
        errors[i] = representations[i] - predictions[i]

    # Update representations (error minimization)
    new_reps = list(representations)
    for i in range(num_layers):
        if i in clamp_mask:
            new_reps[i] = clamp_mask[i]
            continue

        # Error-driven update
        update = -errors[i]

        # Bottom-up error propagation: errors from layer i drive layer i+1
        # But also: layer i receives error signal from layer i-1 (if it predicted i-1)
        if i > 0:
            # This layer predicted layer i-1, so gradient flows back
            e_below = errors[i - 1]
            update = update + prediction_weights[i - 1].T @ e_below

        # Sparsity
        update = update - lambda_sparse * jnp.sign(representations[i])

        # Stability: L2 damping
        update = update - 0.1 * representations[i]

        new_reps[i] = representations[i] + lr * update

        # Clip for stability
        new_reps[i] = jnp.clip(new_reps[i], -5.0, 5.0)

    # Update prediction weights (Hebbian)
    new_pred_w = list(prediction_weights)
    for i in range(num_layers - 1):
        x_high = activation_fn(representations[i + 1])
        dw = jnp.outer(errors[i], x_high)
        new_pred_w[i] = prediction_weights[i] + lr_w * dw

        # Spectral normalization: scale by max singular value estimate
        # (cheap approximation: normalize by Frobenius norm / sqrt(min(H,W)))
        frob = jnp.sqrt(jnp.sum(new_pred_w[i] ** 2))
        min_dim = min(new_pred_w[i].shape)
        max_sv_approx = frob / jnp.sqrt(float(min_dim))
        new_pred_w[i] = jnp.where(
            max_sv_approx > 1.0,
            new_pred_w[i] / max_sv_approx,
            new_pred_w[i],
        )

    # Update skip weights
    new_skip_w = list(skip_weights)
    for si in range(n_skip):
        enc_idx = si
        dec_idx = num_layers - 1 - si
        if enc_idx != dec_idx:
            x_enc = activation_fn(representations[enc_idx])
            e_dec = errors[dec_idx]
            dw = jnp.outer(e_dec, x_enc)
            new_skip_w[si] = skip_weights[si] + lr_w * 0.1 * dw

            # Same normalization
            frob = jnp.sqrt(jnp.sum(new_skip_w[si] ** 2))
            min_dim = min(new_skip_w[si].shape)
            max_sv_approx = frob / jnp.sqrt(float(min_dim))
            new_skip_w[si] = jnp.where(
                max_sv_approx > 1.0,
                new_skip_w[si] / max_sv_approx,
                new_skip_w[si],
            )

    return new_reps, errors, new_pred_w, new_skip_w
```

**Step 4: Run tests**

Run: `uv run python -m pytest training/tests/test_hierarchical_update.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add training/engine/hierarchical_update.py training/tests/test_hierarchical_update.py
git commit -m "feat: add hierarchical PC relaxation step with skip connections"
```

---

### Task 3: Hierarchical Trainer (JIT-compiled training loop)

**Files:**
- Create: `training/engine/hierarchical_trainer.py`
- Test: `training/tests/test_hierarchical_trainer.py`

**Step 1: Write the failing test**

```python
# training/tests/test_hierarchical_trainer.py
import numpy as np
import jax.numpy as jnp
import pytest
from pathlib import Path
from corpus.tests.test_scanner import make_test_midi
from training.engine.hierarchical_trainer import HierarchicalTrainer


@pytest.fixture
def midi_corpus(tmp_path):
    """Create a small test MIDI corpus."""
    for i in range(5):
        make_test_midi(tmp_path / f"song{i}.mid", [
            ("Piano", 0, False, [
                (60 + j, j * 0.25, (j + 1) * 0.25, 100) for j in range(32)
            ]),
            ("Bass", 32, False, [
                (36 + j % 4, j * 0.5, (j + 1) * 0.5, 80) for j in range(16)
            ]),
        ])
    return tmp_path


class TestHierarchicalTrainer:
    def test_init_creates_grid(self, midi_corpus):
        trainer = HierarchicalTrainer(
            midi_dir=str(midi_corpus),
            layer_sizes=[128, 64, 32, 64, 128],
            relaxation_steps=5,
            fs=4.0,
        )
        assert trainer.grid.layer_sizes == [128, 64, 32, 64, 128]

    def test_train_step_returns_finite_error(self, midi_corpus):
        trainer = HierarchicalTrainer(
            midi_dir=str(midi_corpus),
            layer_sizes=[32, 16, 32],
            relaxation_steps=5,
            fs=4.0,
        )
        error, meta = trainer.train_step(batch_size=2)
        assert np.isfinite(error)
        assert error > 0

    def test_train_step_returns_f1(self, midi_corpus):
        trainer = HierarchicalTrainer(
            midi_dir=str(midi_corpus),
            layer_sizes=[32, 16, 32],
            relaxation_steps=5,
            fs=4.0,
        )
        _, meta = trainer.train_step(batch_size=2)
        assert "f1" in meta
        assert "precision" in meta
        assert "recall" in meta

    def test_error_decreases_over_steps(self, midi_corpus):
        trainer = HierarchicalTrainer(
            midi_dir=str(midi_corpus),
            layer_sizes=[32, 16, 32],
            relaxation_steps=8,
            fs=4.0,
        )
        errors = []
        for _ in range(5):
            err, _ = trainer.train_step(batch_size=2)
            errors.append(err)

        # Not necessarily monotonic, but last should be less than first
        # (allow some tolerance — 5 steps is short)
        assert all(np.isfinite(e) for e in errors)

    def test_save_load_checkpoint(self, midi_corpus, tmp_path):
        trainer = HierarchicalTrainer(
            midi_dir=str(midi_corpus),
            layer_sizes=[32, 16, 32],
            relaxation_steps=5,
            fs=4.0,
        )
        trainer.train_step(batch_size=2)

        ckpt_path = tmp_path / "ckpt"
        trainer.save_checkpoint(str(ckpt_path))

        assert (ckpt_path / "metadata.json").exists()
        assert (ckpt_path / "layer_0_rep.npy").exists()
        assert (ckpt_path / "pred_weight_0.npy").exists()

    def test_output_unclamping_schedule(self, midi_corpus):
        """With teacher_forcing_ratio < 1.0, some ticks should be unclamped."""
        trainer = HierarchicalTrainer(
            midi_dir=str(midi_corpus),
            layer_sizes=[32, 16, 32],
            relaxation_steps=5,
            fs=4.0,
            teacher_forcing_ratio=0.5,
        )
        # Should not crash — unclamped ticks run without output clamping
        error, _ = trainer.train_step(batch_size=2)
        assert np.isfinite(error)
```

**Step 2: Run test to verify it fails**

Run: `uv run python -m pytest training/tests/test_hierarchical_trainer.py -v`
Expected: ImportError

**Step 3: Write implementation**

The hierarchical trainer wraps the grid + update rule into a training loop
compatible with the existing batch generator and management command interface.

This file is larger — implement in `training/engine/hierarchical_trainer.py`.
Key components:
- `HierarchicalTrainer.__init__`: creates grid + batch generator
- `HierarchicalTrainer.train_step`: generates batch, runs relaxation, returns metrics
- `HierarchicalTrainer.save_checkpoint` / `load_checkpoint`
- Internal `_process_example`: lax.scan over ticks, inner loop over relaxation steps
- Teacher forcing schedule: `teacher_forcing_ratio` controls what fraction of ticks have output clamped

**Step 4: Run tests**

Run: `uv run python -m pytest training/tests/test_hierarchical_trainer.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add training/engine/hierarchical_trainer.py training/tests/test_hierarchical_trainer.py
git commit -m "feat: add HierarchicalTrainer with teacher forcing schedule"
```

---

### Task 4: Management Command — train_hierarchical

**Files:**
- Create: `training/management/commands/train_hierarchical.py`
- Modify: (nothing — new command)

**Step 1: Create the command**

New Django management command that mirrors `train.py` but uses `HierarchicalTrainer`.
Key args: `--layer-sizes`, `--teacher-forcing`, `--relaxation-steps`, etc.

**Step 2: Smoke test**

Run: `uv run python manage.py train_hierarchical --help`
Expected: Shows help text with all arguments

Run: `uv run python manage.py train_hierarchical --num-steps 3 --batch-size 2 --layer-sizes 32 16 32 --relaxation-steps 5`
Expected: Completes 3 training steps without error

**Step 3: Commit**

```bash
git add training/management/commands/train_hierarchical.py
git commit -m "feat: add train_hierarchical management command"
```

---

### Task 5: Hierarchical Inference + Updated Jam Command

**Files:**
- Create: `training/engine/hierarchical_inference.py`
- Modify: `training/management/commands/jam.py` — detect grid type from checkpoint metadata
- Test: `training/tests/test_hierarchical_inference.py`

**Step 1: Write test**

```python
# training/tests/test_hierarchical_inference.py
import jax.numpy as jnp
import numpy as np
from training.engine.hierarchical_grid import create_hierarchical_grid
from training.engine.hierarchical_inference import run_hierarchical_inference


class TestHierarchicalInference:
    def test_output_shape(self):
        grid = create_hierarchical_grid(layer_sizes=[128, 64, 32, 64, 128])
        input_seq = np.random.rand(8, 128).astype(np.float32)
        cond = np.zeros(16, dtype=np.float32)
        cond[0] = 1.0

        output, _ = run_hierarchical_inference(
            grid, input_seq, cond, relaxation_steps=5)
        assert output.shape == (8, 128)

    def test_output_in_valid_range(self):
        grid = create_hierarchical_grid(layer_sizes=[128, 64, 32, 64, 128])
        input_seq = np.random.rand(8, 128).astype(np.float32)
        cond = np.zeros(16, dtype=np.float32)
        cond[0] = 1.0

        output, _ = run_hierarchical_inference(
            grid, input_seq, cond, relaxation_steps=5)
        assert np.all(output >= 0.0)
        assert np.all(output <= 1.0)

    def test_no_nan_divergence(self):
        """The critical test — hierarchical inference must not NaN."""
        grid = create_hierarchical_grid(layer_sizes=[128, 64, 32, 64, 128])
        input_seq = np.zeros((16, 128), dtype=np.float32)
        input_seq[:, 60] = 1.0  # single note
        cond = np.zeros(16, dtype=np.float32)

        output, _ = run_hierarchical_inference(
            grid, input_seq, cond, relaxation_steps=32)
        assert np.all(np.isfinite(output))
```

**Step 2-5:** Implement, test, commit.

```bash
git commit -m "feat: add hierarchical inference engine and update jam command"
```

---

### Task 6: Signal Propagation Verification

**Files:**
- Modify: `training/management/commands/diagnose.py` — add hierarchical mode

**Step 1: Add hierarchical diagnostic**

Update `diagnose.py` to detect grid type from checkpoint metadata and run
the appropriate signal propagation test.

**Step 2: Run diagnostic on a freshly-trained hierarchical grid**

```bash
uv run python manage.py train_hierarchical --num-steps 50 --batch-size 4 \
    --layer-sizes 128 64 32 64 128 --relaxation-steps 16
uv run python manage.py diagnose --checkpoint checkpoints/step_50
```

Expected: Signal propagation ratio significantly > 0.007 (the flat grid baseline).

**Step 3: Commit**

```bash
git commit -m "feat: add hierarchical mode to diagnose command"
```

---

### Task 7: Run Comparison Experiment

**No code changes — just execution and analysis.**

```bash
# Train hierarchical grid for 250 steps (matching flat grid checkpoint)
uv run python manage.py train_hierarchical \
    --num-steps 250 --batch-size 16 \
    --layer-sizes 128 64 32 64 128 \
    --relaxation-steps 32 \
    --teacher-forcing 0.8 \
    --checkpoint-every 50

# Run diagnostics
uv run python manage.py diagnose --checkpoint checkpoints/step_250

# Compare with flat grid
# Key metrics to compare:
# - Signal propagation ratio (target: > 0.1, flat was 0.007)
# - Notes per tick (target: 1-8, flat was 67.7)
# - Unique output patterns (target: > 10, flat was 4)
# - Temporal autocorrelation (target: 0.3-0.8, flat was 0.95)
```

---

## Implementation Order

1. **Task 1** — HierarchicalGridState (data structure)
2. **Task 2** — Update rule (the core math)
3. **Task 3** — Trainer (training loop)
4. **Task 4** — Management command (CLI interface)
5. **Task 5** — Inference + jam integration
6. **Task 6** — Diagnostics update
7. **Task 7** — Comparison experiment

Each task builds on the previous. Total: ~7 commits, each independently testable.

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| JAX JIT won't compile variable-length layer lists | Use fixed-size padded arrays or pytree-registered dataclass |
| Hierarchical grid too slow (5 dense matmuls per step) | Profile; reduce layer sizes or relaxation steps |
| New architecture still collapses | Fallback: add attention mechanism between layers |
| Teacher forcing schedule too aggressive | Start at 1.0, decrease by 0.1 every 100 steps |
