# Predictive Coding Jam Grid — Design Document

## What Are We Building?

A browser-based visual music tool built on predictive coding (PC), a theory from
neuroscience about how the brain learns by constantly predicting what comes next
and correcting its mistakes.

The core idea: a 128x128 grid of colored squares, where each square is a tiny
"neuron" that tries to predict what its neighbors are doing. You feed music in on
the left side, tell the grid what instrument to play via a few control neurons,
and it learns to produce that instrument's part on the right side. The grid
literally glows with prediction errors as it thinks.

You train it on multitrack MIDI files — songs where each instrument is on a
separate channel. Once trained, you can jam with it in real time: play something
in, pick an instrument, and watch the grid figure out what to play back.

---

## How Predictive Coding Works (The Short Version)

Predictive coding says every neuron does two things:

1. **Predict** — Based on its current state, guess what its neighbors' values
   should be.
2. **Correct** — Compare the prediction to reality. The difference is the
   "prediction error." Use that error to update your state.

When prediction errors are high, the neuron is surprised. When they're low, it
has learned the pattern. Over time, the whole network settles into a state where
predictions match reality — it has learned.

In our grid:
- **Representation (r)**: The neuron's current belief about what value it should
  hold.
- **Error (e)**: How wrong the neuron's neighbors are about predicting it.

Every neuron is actually a *pair*: one value for the representation, one for the
error. That's why the literature calls them "neuron pairs."

---

## The Grid

### Layout

```
         conditioning neurons (one-hot instrument selector)
         vvvvvvvv
    col 0                                              col 127
    +----+----+----+----+----+----+-- ... --+----+----+----+
    |    |    |    | IN | IN | IN |         |    |    |    |  row 0
    +----+----+----+----+----+----+-- ... --+----+----+----+
    |    |    |    |    |    |    |         |    |    |    |  row 1
    |    |    |    |    |    |    |         |    |    |    |
    | I  |    |    |    |    |    |         |    |    | O  |
    | N  |    |    |  INTERIOR   |         |    |    | U  |
    | P  |    |    |  NEURONS    |         |    |    | T  |
    | U  |    |    |  (learn     |         |    |    | P  |
    | T  |    |    |   freely)   |         |    |    | U  |
    |    |    |    |    |    |    |         |    |    | T  |
    |    |    |    |    |    |    |         |    |    |    |
    +----+----+----+----+----+----+-- ... --+----+----+----+
    |    |    |    |    |    |    |         |    |    |    |  row 127
    +----+----+----+----+----+----+-- ... --+----+----+----+
```

- **128 x 128 neurons** — 16,384 neuron pairs total.
- **Left column (col 0)**: Input. Clamped to MIDI data every tick. Row index =
  MIDI pitch (0-127). Value = note velocity (0.0 to 1.0). Multiple instruments
  can be mixed together here.
- **Right column (col 127)**: Output. During training, this is clamped to the
  target instrument's MIDI. During jam mode, it's free — the network predicts
  what should go here.
- **Conditioning block (top-middle of left edge, ~rows 60-67, col 0)**: A small
  group of neurons that tell the network which instrument to generate. One-hot
  encoding — light up one neuron = "play as bass", another = "play as piano",
  etc.
- **Interior neurons (everything else)**: Free to learn whatever representation
  helps minimize prediction errors across the grid.

### What Each Neuron Stores

Every neuron at position (i, j) holds four values:

| Value | What it is | Why it's needed |
|-------|-----------|-----------------|
| **r** | Representation | The neuron's current belief/activation |
| **e** | Prediction error | How wrong neighbors are about this neuron |
| **s** | Leaky integrator state | Slow-decaying memory (passive, like an echo) |
| **h** | Recurrent hidden state | Active temporal memory (learned, like remembering a rhythm) |

The leaky integrator (**s**) gives neurons a fading memory of recent inputs —
think of it like reverb. The recurrent state (**h**) is more deliberate — the
neuron can learn to actively hold onto patterns it needs to remember, like "we're
in the chorus now."

Both are necessary because music is sequential. Without memory, the grid only
sees the current instant and can't learn that a note's meaning depends on what
came before.

### Connections

Each neuron connects to its 4 spatial neighbors (up, down, left, right). Each
connection has a learned weight controlling how strongly one neuron's prediction
influences the other.

---

## The Update Rule (How The Grid Thinks)

Each "tick" of the simulation (one musical time step), the grid runs multiple
**relaxation steps**. Each relaxation step, every neuron does this simultaneously:

### 1. Predict

Each neuron uses its representation to generate a prediction for each neighbor:

```
prediction_for_neighbor = weight_to_neighbor * tanh(r)
```

`tanh` squashes the value to [-1, 1] so things don't explode.

### 2. Compute Error

Compare what neighbors predicted about this neuron vs. its actual representation:

```
e = r - sum(incoming predictions from all neighbors)
```

Positive error = neighbors are under-predicting this neuron.
Negative error = neighbors are over-predicting.

### 3. Update Memory

The two memory mechanisms tick forward:

```
s = alpha * s + (1 - alpha) * r        # leaky integrator (alpha is learned)
h = beta * tanh(h_previous)            # recurrent state (beta is learned)
```

- **alpha** close to 1.0 = long memory (slow decay)
- **alpha** close to 0.0 = short memory (fast decay)
- **beta** controls how strongly past state influences the present

### 4. Update Representation

Adjust the neuron's belief to reduce errors:

```
r += learning_rate * (-e + sum(neighbor_errors * weights) + h + s - r)
```

This pushes the representation toward a state where all prediction errors are
minimized.

### 5. Update Weights (iPC)

Unlike standard PC which waits for the grid to settle before learning, we use
**incremental PC (iPC)** — weights update at every relaxation step:

```
weight_to_neighbor += lr_weight * e * tanh(r_neighbor)
alpha += lr_alpha * (gradient from error w.r.t. alpha)
beta  += lr_beta  * (gradient from error w.r.t. beta)
```

This is more biologically plausible and avoids having to decide "when has the
grid converged?"

### Clamping

- **Training**: Left column, right column, and conditioning neurons are all
  overwritten after each relaxation step. They're held fixed. The interior learns
  to connect them.
- **Jam mode**: Only left column and conditioning are clamped. The right column
  evolves freely — that's the network's musical output.

---

## The Visualization

The grid isn't just a math engine — it's art. The rendering directly shows the
network's internal state.

| What you see | What it means |
|-------------|---------------|
| **Square color (hue)** | Direction of prediction error — e.g., blue for under-predicting, red for over-predicting |
| **Square opacity (alpha)** | Magnitude of error — vivid = large error (surprised), faint = small error (confident) |
| **Gap between squares** | Connection weight — strong connection = colors bleed together; no connection = black gap |

So a region of the grid that has "figured it out" will be faint and smooth. A
region that's actively surprised will be vivid and contrasty. During training,
you'd see waves of color ripple inward from the clamped edges as the grid
processes each musical moment.

The exact color palette will be determined experimentally once the system is
running.

### Rendering Approach: WebGL Ping-Pong

The grid state lives in GPU textures. A fragment shader reads the state textures
and draws each neuron as a colored square with the visual mappings above. The
connection-weight-to-gap-blend is rendered by interpolating colors between
adjacent squares based on their shared weight.

This is the same set of textures used for the compute step (in jam mode), so
there's zero overhead copying data between the simulation and the display — the
simulation state literally *is* the image.

---

## Architecture

The system has two major modes that use different tech stacks:

```
TRAINING MODE (headless, server-side)
======================================

  MIDI Files                Django Backend              JAX (GPU)
  on disk          +-------------------------+    +------------------+
  +---------+      |                         |    |                  |
  | song1/  |----->| Corpus Scanner          |    | PC Update Loop   |
  | song2/  |      |   - find instruments    |    |   - iPC steps    |
  | song3/  |      |   - build vocabulary    |    |   - weight update|
  | ...     |      |                         |    |   - memory update|
  +---------+      | Batch Generator         |--->|                  |
                   |   - curriculum scheduler|    | Checkpointing    |
                   |   - random mixing       |    |   - save weights |
                   |   - snippet cutting     |    |   - save params  |
                   |                         |    +------------------+
                   | REST API                |
                   |   - GET /corpus         |
                   |   - GET /batch          |
                   |   - GET/POST /config    |
                   |   - GET /metrics        |
                   +-------------------------+

JAM MODE (browser, real-time)
======================================

  Trained Model          Browser (WebGL)            MIDI Devices
  (exported)       +-------------------------+    +------------------+
  +----------+     |                         |    |                  |
  | weights  |---->| Load as GPU Textures    |    | Input:           |
  | params   |     |                         |    |  MIDI keyboard   |
  | config   |     | WebGL Ping-Pong Shaders |    |  or MIDI file    |
  +----------+     |   - PC forward pass     |<---|                  |
                   |   - no weight updates   |    | Output:          |
                   |                         |--->|  Virtual MIDI    |
                   | WebGL Render Shaders    |    |  or hardware     |
                   |   - error -> color      |    +------------------+
                   |   - weight -> gap blend |
                   |                         |
                   | Web MIDI API            |
                   |   - live input          |
                   |   - live output         |
                   +-------------------------+
```

### Tech Stack Summary

| Component | Technology | What it does |
|-----------|-----------|-------------|
| Training compute | **JAX** (possibly using PCX library) | Runs the PC update loop on the GPU. Handles iPC weight updates, memory, and all the math. |
| Data pipeline | **Django** + **mido** / **pretty_midi** | Scans MIDI corpus, builds instrument vocabulary, generates randomized training batches according to curriculum. |
| Training API | **Django REST Framework** | Exposes endpoints for training config, batch requests, metrics, and model checkpoints. |
| Jam inference | **WebGL** (GLSL fragment shaders) | Runs the PC forward pass in the browser in real time. Same math as training, minus weight updates. |
| Visualization | **WebGL** (GLSL fragment shaders) | Renders the grid. Reads directly from the compute textures — no data copying. |
| MIDI I/O | **Web MIDI API** + JS MIDI parser | Handles live MIDI input/output in jam mode. Also parses MIDI files for playback-based jamming. |
| Model export | JAX -> flat float32 arrays -> WebGL textures | Bridges training and jam. Weights and parameters exported as binary blobs that load directly into GPU textures. |

---

## Data Pipeline

### Corpus Processing (runs once)

When you point the system at a directory of multitrack MIDI files:

1. **Scan** every file — extract channel/track names, General MIDI program
   numbers, tempo, length.
2. **Build instrument vocabulary** — normalize similar labels into categories
   (e.g., "Electric Bass" + "Fingered Bass" + "Fretless Bass" -> "Bass"). The
   exact clustering is determined by what's in the corpus.
3. **Set one-hot block size** — the number of unique instrument categories
   determines how many conditioning neurons we need.
4. **Store metadata** — per-song index of which instruments are present, tempo,
   duration.

### Batch Generation (runs every training step)

Fully automated. For each item in a training batch:

1. **Curriculum scheduler** decides current snippet length.
2. **Pick a random song** from the corpus.
3. **Pick a random start point** within that song.
4. **Pick a random target instrument** (from those in that song) — this is the
   output the grid should learn to produce.
5. **Pick input instruments** — randomly select 1 to N of the *remaining*
   instruments. Merge their MIDI data together (sum/max velocities per pitch).
6. **Build the one-hot conditioning vector** for the target instrument.
7. **Quantize** the snippet to fixed ticks (e.g., 16th notes).
8. **Yield**: input sequence (128-dim velocity vector per tick), target sequence,
   conditioning vector.

This means every training batch is different: different songs, different
snippets, different instrument combinations, different targets. The network sees
maximum variety.

### Curriculum

Training starts easy and gets harder:

| Phase | Snippet Length | When to advance |
|-------|---------------|-----------------|
| 1 | 1-2 bars (~2-4 sec at 120 BPM) | Average error drops below threshold |
| 2 | 4 bars (~8 sec) | Average error drops below threshold |
| 3 | 8+ bars (~16+ sec) | Ongoing |

Short snippets let the network nail rhythm and harmony first. Longer snippets
require the memory mechanisms (leaky integration, recurrent state) to develop, so
the network learns to track phrase structure and musical context.

---

## GPU Texture Layout

The grid state on the GPU is stored in three RGBA float textures (128x128 each).
Each pixel = one neuron. Each color channel = one value.

| Texture | R channel | G channel | B channel | A channel |
|---------|-----------|-----------|-----------|-----------|
| **State** | r (representation) | e (error) | s (leaky state) | h (recurrent state) |
| **Weights** | w_left | w_right | w_up | w_down |
| **Params** | alpha (decay rate) | beta (recurrent gain) | learning_rate | bias |

During training (JAX), these are JAX arrays on the GPU. For jam mode, these
exact same values are loaded into WebGL textures. The render shader reads from
them directly.

A "ping-pong" setup means we have two copies of the State texture. Each
relaxation step reads from one and writes to the other, then they swap. This is a
standard GPU pattern for simulations where every cell updates simultaneously.

---

## Jam Mode Details

### Input Sources

1. **Live MIDI** — Connect a MIDI keyboard or controller via Web MIDI API. Notes
   are mapped to the left column in real time (row = pitch, value = velocity).
2. **File playback** — Load a MIDI file in the browser. A JS MIDI parser steps
   through it at tempo. Same mapping to the left column.

### Output

The right column's representation values are read each frame, thresholded to
valid MIDI velocities, and emitted as note-on/note-off events via Web MIDI API.
This can go to a virtual MIDI device, a hardware synth, or a software instrument.

### Instrument Selection

Click on the conditioning block in the UI to switch which instrument the grid
jams as. The one-hot pattern updates, and the grid's output shifts to match.

### Performance Budget

- Grid: 128x128 = 16,384 neurons
- Per relaxation step: ~16K texture reads + writes (highly parallel on GPU)
- At 60fps with ~10 relaxation steps per frame = ~160K shader invocations/frame
- Well within WebGL capability on any modern GPU

---

## UI

### Training Mode (browser dashboard, optional)

The actual training runs headless on the server. The browser dashboard is
optional and shows:

- Loss curves (average grid error over time)
- Curriculum phase indicator
- Training parameters with live editing:
  - Learning rates (representation, weights, alpha, beta)
  - Relaxation steps per tick
  - Batch size
  - Curriculum thresholds
  - Nonlinearity choice
- Throughput stats (ticks/sec, batches/sec)
- Start / stop / save controls

### Jam Mode (main experience)

- **Center**: The 128x128 grid visualization (the star of the show)
- **Side panel**:
  - MIDI input source (live device or file)
  - Instrument selector (clickable one-hot block)
  - Playback controls (play/pause/tempo for file mode)
- **Status bar**: Current tick, relaxation steps/sec, average error

The grid is the main attraction. Everything else stays out of the way.

---

## Model Export: JAX to WebGL

After training, the model needs to move from JAX (server, Python) to WebGL
(browser, JavaScript). The pipeline:

1. **JAX side**: Extract the three state arrays (State, Weights, Params) as
   flat float32 NumPy arrays. Save as binary files (`.bin`).
2. **Also export**: Grid dimensions, instrument vocabulary (JSON), any config
   the browser needs.
3. **Browser side**: Fetch the `.bin` files, create WebGL float textures from
   them. The GLSL shaders are written to expect the exact same layout.

This is intentionally simple — no model format conversion, no ONNX, no runtime.
Just raw floats into textures.

---

## Known Stability Issues & Mitigations

Predictive coding networks are prone to several instabilities, especially with
incremental updates (iPC) and 2D grid topologies. This section catalogs the known
failure modes and strategies from the literature. None of these are currently
implemented — they're documented here for future reference.

### 1. Error Explosion

**Problem**: Without clipping, prediction errors can grow without bound, causing
NaN/Inf in representations and weights. This is the most common training failure.

**Mitigations**:
- **Error clipping**: Clamp `e` to `[-clip, clip]` after computing. Simple and
  effective. Common values: clip = 5.0–10.0. (Whittington & Bogacz, 2017)
- **Gradient scaling**: Scale the error signal by `1 / (1 + |e|)` to soft-clip
  rather than hard-clip. Preserves gradient direction.

### 2. Representational Collapse

**Problem**: All neurons converge to the same value (usually 0), losing all
information. The grid "goes dark." This happens when weight updates are too
aggressive relative to representation updates.

**Mitigations**:
- **Weight decay**: `w *= (1 - lambda)` each step, preventing any single weight
  from dominating. Lambda = 1e-4 to 1e-3 is typical.
- **Learning rate separation**: Keep `lr_weights << lr_representations`. Our
  current ratio (0.1x) follows Millidge et al. (2022).
- **Spectral normalization**: Constrain the spectral norm of weight matrices.
  Expensive but theoretically grounded.

### 3. Precision Weighting

**Problem**: Standard PC treats all prediction errors equally, but some neurons'
predictions are inherently noisier than others. The network wastes capacity
trying to minimize noise.

**Mitigations**:
- **Learned precisions**: Each neuron pair learns a precision parameter `pi` that
  scales its error: `e_weighted = pi * e`. High precision = "I trust this
  prediction." (Friston, 2005; Bogacz, 2017)
- **Log-precision parameterization**: Learn `log(pi)` instead of `pi` to avoid
  negative precision. Update rule: `log_pi += lr * (e^2 - 1/pi)`.
- This is the PC equivalent of attention — the network learns where to look.

### 4. Adaptive Step Size

**Problem**: Fixed learning rates are fragile. Too high = divergence, too low =
stagnation. The optimal rate changes during training as the error landscape
shifts.

**Mitigations**:
- **Per-neuron adaptive rates**: Adam-like momentum on the representation
  updates. (Millidge et al., 2021)
- **Line search**: After computing the update direction, search for the step size
  that minimizes error along that direction. Expensive but robust. (Bogacz, 2017)
- **Warm-up schedule**: Start with small lr, ramp up over first N steps.

### 5. Weight Decay vs. Hard Clipping

**Problem**: Weight magnitudes grow monotonically under iPC because updates are
proportional to `e * tanh(r)`, and both can be consistently positive.

**Mitigations**:
- **L2 weight decay**: `w -= lambda * w` each step. Simple, smooth.
- **Hard clipping**: `w = clip(w, -max_w, max_w)`. Abrupt but guarantees bounds.
  Common with `max_w = 2.0–5.0`.
- **Weight normalization**: Normalize each neuron's outgoing weight vector to
  unit norm. Forces competition between connections.

### 6. Temporal Stability

**Problem**: The leaky integrator `s` and recurrent state `h` can accumulate
unbounded values over long sequences, especially during curriculum phase 3.

**Mitigations**:
- **State clipping**: Clamp `s` and `h` to `[-1, 1]` since they're used with
  tanh anyway.
- **Periodic state reset**: Zero out `s` and `h` between training snippets (we
  already do this).
- **Gradient clipping on alpha/beta**: Prevent `alpha` from reaching exactly 1.0
  (infinite memory) by clamping to `[0, 0.999]`.

### Key References

- **Friston (2005)** — "A theory of cortical responses." Foundational PC paper;
  introduces precision weighting.
- **Bogacz (2017)** — "A tutorial on the free-energy framework for modelling
  perception and learning." Practical PC tutorial; adaptive step sizes.
- **Whittington & Bogacz (2017)** — "An approximation of the error
  backpropagation algorithm in a predictive coding network with local Hebbian
  synaptic plasticity." Error clipping, iPC convergence.
- **Millidge, Tschantz & Buckley (2021)** — "Predictive Coding Approximates
  Backprop along Arbitrary Computation Graphs." Theoretical grounding for
  iPC; lr separation analysis.
- **Millidge, Seth & Buckley (2022)** — "Predictive coding: a theoretical and
  experimental review." Comprehensive survey; stability best practices.
- **Salvatori et al. (2023)** — "Brain-Inspired Computational Intelligence via
  Predictive Coding." Modern survey covering 2D/grid topologies.

---

## Open Questions (to resolve during implementation)

- **Exact color palette** — to be determined experimentally once rendering works.
- **Instrument clustering** — how aggressively to merge similar instruments
  depends on what the corpus actually contains.
- **Optimal relaxation steps per tick** — literature says depth to 2x depth
  (~128-256 for our grid), but this is for feedforward networks. Our 2D grid may
  behave differently. Needs experimentation.
- **Whether to use PCX library or custom JAX** — depends on how well PCX fits
  our 2D grid topology (it's designed for layer-stacked networks).
- **Recurrent state architecture** — the simple `h = beta * tanh(h_prev)` may
  need to be richer (e.g., GRU-like gating) if the network struggles with longer
  snippets.
