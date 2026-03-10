# Predictive Coding Music Grid — Improvement Plan

## Goal
Build a PC network that can **play music along a MIDI input** — given some instruments,
generate a complementary target instrument in real-time.

## Diagnosis: Why the Current Grid Collapses

### Problem 1: Signal Death (architectural)
- 4-neighbor connectivity + L2 weight normalization (||w||₂ ≤ 1)
- Each weight saturates at ~0.5 → after 15 hops: 0.5^15 ≈ 0.00003
- Input at column 0 cannot reach column 15 with meaningful amplitude
- FC inter-column helps but is still column-sequential

### Problem 2: Boundary Dominance (training objective)
- pos_weight=20 at output boundary overwhelms interior signals
- Grid learns to satisfy boundaries independently, not input→output mapping
- During training both sides are clamped — grid never learns to *generate*

### Problem 3: Trivial Local Minima (learning rule)
- Hebbian update w += lr · error · neighbor converges to "copy neighbor"
- This IS the energy minimum for a flat grid — correct PC behavior, wrong architecture
- Without hierarchy, no pressure to form structured representations

---

## Phase 1: Inference Pipeline & Diagnostics (Week 1)

**Goal**: See what the grid actually does, build generation capability.

### 1.1 Build `jam` Management Command
- New command: `python manage.py jam --checkpoint <path> --midi <input.mid>`
- Clamp only column 0 (input), let column W-1 freely evolve
- Run relaxation, read output column, threshold to binary (>0.5), write MIDI
- This is our first "does it work at all?" test

### 1.2 Diagnostic Visualizations
- **Signal propagation test**: Clamp a single note at input, measure activation
  magnitude at each column after full relaxation → plot decay curve
- **Weight magnitude heatmap**: Visualize learned weights across grid
- **Column energy profile**: Plot mean |error| per column during training
- **Temporal coherence**: Plot output activations across 64 ticks — do they change?

### 1.3 Baseline Metrics
- **Note F1** (existing): precision/recall on binary note predictions
- **Onset F1**: New — only count note onsets (transitions 0→1), not sustains
- **Pitch class distribution**: Compare output pitch histogram to training data
- **Temporal autocorrelation**: Measure how much output changes tick-to-tick
- **Signal propagation ratio**: max(|r|) at column W-1 / max(|r|) at column 0

---

## Phase 2: Architecture Overhaul (Weeks 2-3)

### 2.1 Reinterpret Grid as Encoder-Decoder Hierarchy

Current flat grid (all columns equivalent):
```
col0 → col1 → col2 → ... → col14 → col15
input                                 output
```

New: **Hourglass / Pyramid architecture** with columns as hierarchy levels:
```
col0    col1    col2    col3    col4    col5    col6    col7
128     128      64      64      32      32      64     128
input  encode  encode  encode  latent  decode  decode  output
  ↓       ↓       ↓       ↓       ↑       ↑       ↑       ↑
  ←←←←←← errors flow inward →→→→→→→→→→
  →→→→→→→ predictions flow outward ←←←←←←
```

**Key changes:**
- **Reduce columns to 8** (encoder: 4, decoder: 4) — fewer hops, each is meaningful
- **Spatial pooling** between columns (128→64→32 via learned 2:1 pooling)
- **Skip connections** from encoder to decoder (like U-Net) to preserve pitch resolution
- **Predictions flow outward** (high→low abstraction), errors flow inward (low→high)
- This maps directly to the PC formalism: higher levels predict lower levels

### 2.2 Proper PC Layer Connectivity

Replace flat neighbor weights with **inter-layer prediction matrices**:

```python
# Instead of 4 scalar neighbor weights per neuron:
# Use dense prediction matrices between adjacent layers

# Encoder (bottom-up / error propagation):
E_l = x_l - f(W_pred_l · x_{l+1})    # error at layer l
x_{l+1} += lr * W_pred_l.T · E_l       # update higher layer from errors below

# Decoder (top-down / prediction):
pred_l = f(W_pred_l · x_{l+1})          # top-down prediction of layer l
```

Each inter-column connection is a **(H_from, H_to) matrix**, not scalar weights.
This gives enough capacity for meaningful transformations.

### 2.3 Temporal Architecture Upgrade

Current: leaky integrator (s = 0.9·s + 0.1·r) — very weak temporal model.

New: **Predictive temporal model** — each layer predicts its own next state:
```python
# Per layer, per tick:
x_temporal_pred = W_temporal @ activation(x_prev_tick)  # predict current from previous
e_temporal = x_current - x_temporal_pred                 # temporal prediction error
x_current += lr * (-e_temporal)                          # minimize temporal surprise

# W_temporal learned via same Hebbian rule:
W_temporal += lr_t * e_temporal @ x_prev_tick.T
```

This gives the network a *reason* to produce temporally coherent output:
deviating from temporal predictions costs energy.

### 2.4 Training Regime Change: Teacher Forcing → Free Running

Current: Both input AND output clamped during all training.
Problem: Grid never learns to generate — only to interpolate between known boundaries.

New: **Scheduled output unclamping**
```
Phase A (steps 0-1000):     100% teacher forcing (clamp both, learn representations)
Phase B (steps 1000-3000):  50% teacher forcing (randomly unclamp output on half the ticks)
Phase C (steps 3000+):      10% teacher forcing (mostly free-running generation)
```

During unclamped ticks, the output column evolves freely via relaxation.
The loss is still computed at the output boundary (compare free output to target).
But now the grid must actually *generate* the output, not just memorize it.

---

## Phase 3: Data Quality & Preprocessing (Week 2, parallel with Phase 2)

### 3.1 MIDI Quality Filtering
- Filter out songs with < 3 instruments (less interesting for accompaniment)
- Filter out MIDI with extreme tempos (< 40 BPM or > 200 BPM)
- Filter out MIDI with very few notes (< 50 notes total)
- Ensure at least one melodic + one rhythmic instrument per song

### 3.2 Piano Roll Preprocessing
- **Quantize to 16th notes**: Current fs=8.0 gives 8 ticks/second.
  At 120 BPM, a 16th note = 0.125s → 1 tick. Good match. Keep fs=8.0.
- **Velocity binarization option**: Instead of continuous [0,1], try binary {0,1}
  (note on/off) — simpler target for the grid to learn initially
- **Pitch range reduction**: Most music uses pitches 36-96 (C2-C7).
  Mask or remove the rarely-used extremes to focus grid capacity.
- **Transposition augmentation**: Transpose each song by ±6 semitones
  (shift piano roll rows) → 13× more training data, key-invariant learning

### 3.3 Input-Target Pairing Strategy
Current: Random instrument selection for input/target.
New: **Structured pairing** based on musical roles:
- Rhythm section (drums, bass) → predict melody (piano, guitar, strings)
- Accompaniment (piano, guitar) → predict melody (vocal proxy, lead)
- Full band minus one → predict the missing instrument
- This teaches musically meaningful relationships

### 3.4 Snippet Context Enhancement
- **Overlap training**: Instead of random non-overlapping snippets,
  use sliding window with 50% overlap → better temporal continuity
- **Longer context, predict last**: Feed 32 ticks of context, only compute
  loss on last 8 ticks → teaches the grid to use history

---

## Phase 4: Evaluation Framework (Week 3)

### 4.1 Quantitative Metrics
| Metric | What it measures | Target |
|--------|-----------------|--------|
| Note F1 | Accuracy of note presence | > 0.5 |
| Onset F1 | Accuracy of note beginnings | > 0.3 |
| Pitch histogram KL | Distribution match to real music | < 2.0 |
| Temporal autocorr | Output stability/change | 0.3-0.7 |
| Polyphony match | Correct number of simultaneous notes | ±2 notes |
| Signal propagation | Input reaches output | > 0.1 ratio |
| Column energy balance | No dead/exploding columns | std < 0.5 |

### 4.2 Qualitative Evaluation
- **MIDI playback**: Export generated MIDI and listen
- **Piano roll comparison**: Side-by-side plot of target vs generated
- **Grid activity video**: Animate grid state across ticks during inference

### 4.3 Ablation Studies
- Hierarchy vs flat (current)
- Temporal prediction vs leaky integrator
- Teacher forcing schedule: 100% vs 50% vs 10%
- Binary vs continuous piano roll
- Transposition augmentation vs none

---

## Phase 5: Jam Mode (Week 4)

### 5.1 Real-Time MIDI Interface
- Accept MIDI input stream (via python-rtmidi or similar)
- Buffer 8 ticks (1 second at fs=8.0)
- Run inference on grid with input clamped
- Output predictions as MIDI note-on/note-off events
- Latency target: < 500ms (4 ticks lookahead)

### 5.2 Interactive Controls
- Select target instrument category via conditioning
- Adjust "creativity" (temperature on output sigmoid)
- Manual pitch range constraint (e.g., bass only plays 28-60)
- Blend multiple generated instruments

### 5.3 Web Interface Integration
- Extend existing Django API with `/api/training/jam/` endpoint
- WebMIDI input in browser → API → grid inference → WebMIDI output
- Real-time grid visualization during jamming

---

## Implementation Priority

```
1. [CRITICAL] Build jam command + signal propagation diagnostic  (Phase 1)
2. [CRITICAL] Hourglass architecture with proper PC layers       (Phase 2.1-2.2)
3. [HIGH]     Teacher forcing → free running schedule            (Phase 2.4)
4. [HIGH]     Temporal prediction model                          (Phase 2.3)
5. [MEDIUM]   Data filtering + transposition augmentation        (Phase 3)
6. [MEDIUM]   Evaluation metrics                                 (Phase 4)
7. [LOW]      Real-time MIDI jam interface                       (Phase 5)
```

## Key Hypothesis

> The grid collapses because it's a flat sheet trying to do what a hierarchy does.
> By reinterpreting columns as hierarchical layers with proper PC dynamics
> (predictions down, errors up, pooling between scales), and by training with
> progressive output unclamping, the grid should learn to generate musically
> coherent accompaniment.

## Risk: What If Hierarchy Alone Isn't Enough?

Fallback approaches:
- **Attention mechanism**: Add cross-column attention (breaks strict locality but might be needed)
- **Hybrid approach**: Use a small transformer for temporal modeling, PC grid for pitch relationships
- **Amortized inference**: Train an encoder network to initialize grid state (faster convergence)
