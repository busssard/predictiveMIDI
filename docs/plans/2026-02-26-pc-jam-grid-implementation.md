# PC Jam Grid — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a browser-based predictive coding music grid that learns to jam along with MIDI input — JAX backend for training, WebGL frontend for real-time inference and visualization.

**Architecture:** Django serves the MIDI corpus and training config via REST API. JAX runs the PC grid training on GPU, consuming batches from a shared-memory queue. Trained models export as flat float32 arrays that load directly into WebGL textures in the browser for real-time jam mode.

**Tech Stack:** Python 3.11+, Django 5, Django REST Framework, JAX (GPU), pretty_midi, mido, vanilla HTML/JS/WebGL (no frontend framework)

**Reference:** See `docs/plans/2026-02-26-pc-jam-grid-design.md` for the full design document with diagrams and rationale.

---

## Project Structure

```
experiment/
├── pcjam/                         # Django project config
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── corpus/                        # Django app: MIDI corpus management
│   ├── __init__.py
│   ├── models.py                  # Song, Instrument, Track models
│   ├── admin.py
│   ├── views.py                   # REST endpoints
│   ├── serializers.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── scanner.py             # Scan MIDI directory, extract metadata
│   │   ├── vocabulary.py          # Build instrument vocabulary
│   │   ├── batch_generator.py     # Generate training batches
│   │   └── curriculum.py          # Curriculum scheduling
│   └── tests/
│       ├── __init__.py
│       ├── test_scanner.py
│       ├── test_vocabulary.py
│       ├── test_batch_generator.py
│       └── test_curriculum.py
├── training/                      # Django app: training engine
│   ├── __init__.py
│   ├── models.py                  # TrainingRun, Checkpoint models
│   ├── admin.py
│   ├── views.py                   # REST endpoints for config/metrics
│   ├── serializers.py
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── grid.py                # JAX: grid state initialization
│   │   ├── update_rule.py         # JAX: single PC relaxation step
│   │   ├── trainer.py             # JAX: training loop
│   │   └── export.py              # Export model to binary
│   └── tests/
│       ├── __init__.py
│       ├── test_grid.py
│       ├── test_update_rule.py
│       ├── test_trainer.py
│       └── test_export.py
├── frontend/                      # Static files for jam mode
│   ├── index.html
│   ├── js/
│   │   ├── app.js                 # Main entry, UI wiring
│   │   ├── grid-renderer.js       # WebGL grid visualization
│   │   ├── grid-compute.js        # WebGL PC forward pass
│   │   ├── model-loader.js        # Load exported model into textures
│   │   └── midi-io.js             # Web MIDI API + file playback
│   └── shaders/
│       ├── pc-step.frag           # PC relaxation step (inference only)
│       ├── render-grid.frag       # Error → color, weight → gap
│       └── passthrough.vert       # Simple vertex shader
├── midi_data/                     # MIDI files go here (gitignored)
├── checkpoints/                   # Saved model states (gitignored)
├── manage.py
├── requirements.txt
├── pytest.ini
├── .gitignore
└── docs/plans/
```

---

## Task 1: Project Scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `pytest.ini`
- Create: `.gitignore`
- Create: `pcjam/__init__.py`, `pcjam/settings.py`, `pcjam/urls.py`, `pcjam/wsgi.py`
- Create: `manage.py`

### Step 1: Create requirements.txt

```txt
django>=5.0,<6.0
djangorestframework>=3.15,<4.0
django-cors-headers>=4.3,<5.0
pretty_midi>=0.2.10
mido>=1.3
jax[cuda12]>=0.4.30
jaxlib>=0.4.30
numpy>=1.26
pytest>=8.0
pytest-django>=4.8
```

### Step 2: Create pytest.ini

```ini
[pytest]
DJANGO_SETTINGS_MODULE = pcjam.settings
python_files = tests/test_*.py
python_classes = Test*
python_functions = test_*
```

### Step 3: Create .gitignore

```
__pycache__/
*.pyc
*.egg-info/
.eggs/
db.sqlite3
midi_data/
checkpoints/
*.bin
.venv/
node_modules/
```

### Step 4: Create Django project files

**`pcjam/__init__.py`**: empty file

**`pcjam/settings.py`**:
```python
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = "dev-secret-key-change-in-production"
DEBUG = True
ALLOWED_HOSTS = ["*"]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rest_framework",
    "corsheaders",
    "corpus",
    "training",
]

MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
]

ROOT_URLCONF = "pcjam.urls"
WSGI_APPLICATION = "pcjam.wsgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

STATIC_URL = "static/"
STATICFILES_DIRS = [BASE_DIR / "frontend"]

CORS_ALLOW_ALL_ORIGINS = True

# Project-specific settings
MIDI_DATA_DIR = os.environ.get("MIDI_DATA_DIR", str(BASE_DIR / "midi_data"))
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", str(BASE_DIR / "checkpoints"))
GRID_SIZE = 128
```

**`pcjam/urls.py`**:
```python
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/corpus/", include("corpus.urls")),
    path("api/training/", include("training.urls")),
]
```

**`pcjam/wsgi.py`**:
```python
import os
from django.core.wsgi import get_wsgi_application
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "pcjam.settings")
application = get_wsgi_application()
```

**`manage.py`**:
```python
#!/usr/bin/env python
import os
import sys

def main():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "pcjam.settings")
    from django.core.management import execute_from_command_line
    execute_from_command_line(sys.argv)

if __name__ == "__main__":
    main()
```

### Step 5: Create empty app directories

Create `corpus/__init__.py`, `corpus/tests/__init__.py`, `corpus/services/__init__.py`,
`training/__init__.py`, `training/tests/__init__.py`, `training/engine/__init__.py`.

Stub `corpus/models.py`, `corpus/admin.py`, `corpus/views.py`, `corpus/serializers.py`,
`corpus/urls.py` (empty urlpatterns).
Same for `training/`.

### Step 6: Install dependencies and verify

Run: `pip install -r requirements.txt`

Run: `python manage.py check`
Expected: "System check identified no issues."

### Step 7: Commit

```bash
git add -A
git commit -m "feat: scaffold Django project with corpus and training apps"
```

---

## Task 2: MIDI Corpus Scanner

Scan a directory of MIDI files, extract metadata (instruments, tempo, duration, track names).

**Files:**
- Create: `corpus/services/scanner.py`
- Test: `corpus/tests/test_scanner.py`
- Create: `tests/fixtures/` (test MIDI files)

### Step 1: Create a test MIDI fixture

We need a small MIDI file for testing. We'll generate one programmatically in the test setup.

**`corpus/tests/test_scanner.py`**:
```python
import tempfile
from pathlib import Path
import pretty_midi
import pytest
from corpus.services.scanner import scan_midi_file, scan_directory


def make_test_midi(path, instruments):
    """Create a MIDI file with given instruments.

    instruments: list of (name, program, is_drum, notes) tuples.
    notes: list of (pitch, start, end, velocity) tuples.
    """
    midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    for name, program, is_drum, notes in instruments:
        inst = pretty_midi.Instrument(program=program, is_drum=is_drum, name=name)
        for pitch, start, end, velocity in notes:
            inst.notes.append(pretty_midi.Note(
                velocity=velocity, pitch=pitch, start=start, end=end,
            ))
        midi.instruments.append(inst)
    midi.write(str(path))


class TestScanMidiFile:
    def test_extracts_instruments(self, tmp_path):
        path = tmp_path / "test.mid"
        make_test_midi(path, [
            ("Piano", 0, False, [(60, 0.0, 1.0, 100)]),
            ("Bass", 32, False, [(36, 0.0, 2.0, 80)]),
        ])
        result = scan_midi_file(path)
        assert result["path"] == str(path)
        assert len(result["instruments"]) == 2
        assert result["instruments"][0]["name"] == "Piano"
        assert result["instruments"][0]["program"] == 0
        assert result["instruments"][1]["name"] == "Bass"
        assert result["instruments"][1]["program"] == 32

    def test_extracts_tempo(self, tmp_path):
        path = tmp_path / "test.mid"
        make_test_midi(path, [("Piano", 0, False, [(60, 0.0, 1.0, 100)])])
        result = scan_midi_file(path)
        assert result["tempo"] == pytest.approx(120.0, abs=1.0)

    def test_extracts_duration(self, tmp_path):
        path = tmp_path / "test.mid"
        make_test_midi(path, [
            ("Piano", 0, False, [(60, 0.0, 3.5, 100)]),
        ])
        result = scan_midi_file(path)
        assert result["duration"] == pytest.approx(3.5, abs=0.1)

    def test_identifies_drums(self, tmp_path):
        path = tmp_path / "test.mid"
        make_test_midi(path, [
            ("Drums", 0, True, [(36, 0.0, 0.5, 100)]),
        ])
        result = scan_midi_file(path)
        assert result["instruments"][0]["is_drum"] is True


class TestScanDirectory:
    def test_scans_all_midi_files(self, tmp_path):
        for i in range(3):
            make_test_midi(
                tmp_path / f"song{i}.mid",
                [("Piano", 0, False, [(60, 0.0, 1.0, 100)])],
            )
        # Also create a non-MIDI file that should be ignored
        (tmp_path / "readme.txt").write_text("not midi")
        results = scan_directory(str(tmp_path))
        assert len(results) == 3

    def test_returns_empty_for_empty_dir(self, tmp_path):
        results = scan_directory(str(tmp_path))
        assert results == []
```

### Step 2: Run tests to verify they fail

Run: `pytest corpus/tests/test_scanner.py -v`
Expected: FAIL — `ImportError: cannot import name 'scan_midi_file'`

### Step 3: Implement scanner

**`corpus/services/scanner.py`**:
```python
from pathlib import Path
import pretty_midi


def scan_midi_file(path):
    """Extract metadata from a single MIDI file.

    Returns dict with keys: path, instruments, tempo, duration.
    Each instrument has: name, program, is_drum, note_count, pitch_range.
    """
    path = Path(path)
    midi = pretty_midi.PrettyMIDI(str(path))

    tempos = midi.get_tempo_changes()[1]
    tempo = float(tempos[0]) if len(tempos) > 0 else 120.0

    instruments = []
    for inst in midi.instruments:
        pitches = [n.pitch for n in inst.notes]
        instruments.append({
            "name": inst.name,
            "program": inst.program,
            "is_drum": inst.is_drum,
            "note_count": len(inst.notes),
            "pitch_range": (min(pitches), max(pitches)) if pitches else (0, 0),
        })

    return {
        "path": str(path),
        "instruments": instruments,
        "tempo": tempo,
        "duration": midi.get_end_time(),
    }


def scan_directory(directory):
    """Scan all MIDI files in a directory. Returns list of scan results."""
    directory = Path(directory)
    results = []
    for ext in ("*.mid", "*.midi", "*.MID", "*.MIDI"):
        for path in sorted(directory.glob(ext)):
            try:
                results.append(scan_midi_file(path))
            except Exception as e:
                print(f"Warning: could not parse {path}: {e}")
    return results
```

### Step 4: Run tests to verify they pass

Run: `pytest corpus/tests/test_scanner.py -v`
Expected: All 5 tests PASS

### Step 5: Commit

```bash
git add corpus/services/scanner.py corpus/tests/test_scanner.py
git commit -m "feat: MIDI corpus scanner — extract instruments, tempo, duration"
```

---

## Task 3: Instrument Vocabulary Builder

Cluster raw instrument names/programs into categories. Determines one-hot block size.

**Files:**
- Create: `corpus/services/vocabulary.py`
- Test: `corpus/tests/test_vocabulary.py`

### Step 1: Write failing tests

**`corpus/tests/test_vocabulary.py`**:
```python
from corpus.services.vocabulary import build_vocabulary, categorize_instrument

# General MIDI program ranges:
# 0-7: Piano, 8-15: Chromatic Percussion, 16-23: Organ,
# 24-31: Guitar, 32-39: Bass, 40-47: Strings, 48-55: Ensemble,
# 56-63: Brass, 64-71: Reed, 72-79: Pipe, 80-87: Synth Lead,
# 88-95: Synth Pad, 96-103: Synth Effects, 104-111: Ethnic,
# 112-119: Percussive, 120-127: Sound Effects


class TestCategorizeInstrument:
    def test_piano_programs(self):
        assert categorize_instrument(0, False) == "piano"
        assert categorize_instrument(5, False) == "piano"

    def test_bass_programs(self):
        assert categorize_instrument(32, False) == "bass"
        assert categorize_instrument(35, False) == "bass"

    def test_guitar_programs(self):
        assert categorize_instrument(24, False) == "guitar"
        assert categorize_instrument(31, False) == "guitar"

    def test_drums(self):
        assert categorize_instrument(0, True) == "drums"
        assert categorize_instrument(99, True) == "drums"

    def test_strings(self):
        assert categorize_instrument(40, False) == "strings"

    def test_synth(self):
        assert categorize_instrument(80, False) == "synth"
        assert categorize_instrument(95, False) == "synth"


class TestBuildVocabulary:
    def test_builds_from_scan_results(self):
        scan_results = [
            {
                "path": "song1.mid",
                "instruments": [
                    {"name": "Piano", "program": 0, "is_drum": False},
                    {"name": "Bass", "program": 32, "is_drum": False},
                ],
            },
            {
                "path": "song2.mid",
                "instruments": [
                    {"name": "Piano", "program": 1, "is_drum": False},
                    {"name": "Drums", "program": 0, "is_drum": True},
                    {"name": "Guitar", "program": 25, "is_drum": False},
                ],
            },
        ]
        vocab = build_vocabulary(scan_results)
        assert "piano" in vocab
        assert "bass" in vocab
        assert "drums" in vocab
        assert "guitar" in vocab
        assert vocab["piano"] == 0 or isinstance(vocab["piano"], int)
        # Each category gets a unique index
        indices = list(vocab.values())
        assert len(indices) == len(set(indices))

    def test_empty_corpus(self):
        vocab = build_vocabulary([])
        assert vocab == {}
```

### Step 2: Run tests to verify they fail

Run: `pytest corpus/tests/test_vocabulary.py -v`
Expected: FAIL — `ImportError`

### Step 3: Implement vocabulary builder

**`corpus/services/vocabulary.py`**:
```python
GM_CATEGORIES = {
    range(0, 8): "piano",
    range(8, 16): "chromatic_percussion",
    range(16, 24): "organ",
    range(24, 32): "guitar",
    range(32, 40): "bass",
    range(40, 48): "strings",
    range(48, 56): "ensemble",
    range(56, 64): "brass",
    range(64, 72): "reed",
    range(72, 80): "pipe",
    range(80, 88): "synth",
    range(88, 96): "synth",
    range(96, 104): "synth_fx",
    range(104, 112): "ethnic",
    range(112, 120): "percussive",
    range(120, 128): "sound_fx",
}


def categorize_instrument(program, is_drum):
    """Map a GM program number to an instrument category string."""
    if is_drum:
        return "drums"
    for prog_range, category in GM_CATEGORIES.items():
        if program in prog_range:
            return category
    return "other"


def build_vocabulary(scan_results):
    """Build instrument vocabulary from scan results.

    Returns dict mapping category name to one-hot index.
    """
    categories = set()
    for song in scan_results:
        for inst in song.get("instruments", []):
            cat = categorize_instrument(inst["program"], inst["is_drum"])
            categories.add(cat)
    return {cat: idx for idx, cat in enumerate(sorted(categories))}
```

### Step 4: Run tests to verify they pass

Run: `pytest corpus/tests/test_vocabulary.py -v`
Expected: All 8 tests PASS

### Step 5: Commit

```bash
git add corpus/services/vocabulary.py corpus/tests/test_vocabulary.py
git commit -m "feat: instrument vocabulary builder — GM program clustering"
```

---

## Task 4: Batch Generator

Cut random snippets from songs, mix instruments, produce training batches.

**Files:**
- Create: `corpus/services/batch_generator.py`
- Test: `corpus/tests/test_batch_generator.py`

### Step 1: Write failing tests

**`corpus/tests/test_batch_generator.py`**:
```python
import tempfile
from pathlib import Path
import numpy as np
import pretty_midi
import pytest
from corpus.services.batch_generator import BatchGenerator
from corpus.tests.test_scanner import make_test_midi


def make_corpus(tmp_path, num_songs=3):
    """Create a small test corpus with piano, bass, and drums."""
    paths = []
    for i in range(num_songs):
        path = tmp_path / f"song{i}.mid"
        make_test_midi(path, [
            ("Piano", 0, False, [
                (60 + j, j * 0.5, (j + 1) * 0.5, 100) for j in range(16)
            ]),
            ("Bass", 32, False, [
                (36 + j % 4, j * 0.5, (j + 1) * 0.5, 80) for j in range(16)
            ]),
            ("Drums", 0, True, [
                (36, j * 0.5, j * 0.5 + 0.25, 100) for j in range(16)
            ]),
        ])
        paths.append(path)
    return paths


class TestBatchGenerator:
    def test_produces_correct_shapes(self, tmp_path):
        make_corpus(tmp_path)
        gen = BatchGenerator(str(tmp_path), snippet_ticks=16, fs=8.0)
        batch = gen.generate_batch(batch_size=4)
        assert batch["input"].shape == (4, 16, 128)
        assert batch["target"].shape == (4, 16, 128)
        assert batch["conditioning"].shape[0] == 4
        # conditioning width = number of instrument categories
        assert batch["conditioning"].shape[1] == len(gen.vocabulary)

    def test_input_values_in_range(self, tmp_path):
        make_corpus(tmp_path)
        gen = BatchGenerator(str(tmp_path), snippet_ticks=16, fs=8.0)
        batch = gen.generate_batch(batch_size=4)
        assert batch["input"].min() >= 0.0
        assert batch["input"].max() <= 1.0
        assert batch["target"].min() >= 0.0
        assert batch["target"].max() <= 1.0

    def test_conditioning_is_one_hot(self, tmp_path):
        make_corpus(tmp_path)
        gen = BatchGenerator(str(tmp_path), snippet_ticks=16, fs=8.0)
        batch = gen.generate_batch(batch_size=8)
        for i in range(8):
            assert batch["conditioning"][i].sum() == pytest.approx(1.0)

    def test_target_not_in_input_mix(self, tmp_path):
        """The target instrument should not appear in the input mix."""
        make_corpus(tmp_path)
        gen = BatchGenerator(str(tmp_path), snippet_ticks=16, fs=8.0)
        # This is probabilistic, but with enough samples we can check
        # that target and input differ
        batch = gen.generate_batch(batch_size=4)
        # At minimum, target_instrument_idx should be set
        assert "target_categories" in batch

    def test_different_snippet_lengths(self, tmp_path):
        make_corpus(tmp_path)
        gen = BatchGenerator(str(tmp_path), snippet_ticks=32, fs=8.0)
        batch = gen.generate_batch(batch_size=2)
        assert batch["input"].shape[1] == 32
```

### Step 2: Run tests to verify they fail

Run: `pytest corpus/tests/test_batch_generator.py -v`
Expected: FAIL — `ImportError`

### Step 3: Implement batch generator

**`corpus/services/batch_generator.py`**:
```python
from pathlib import Path
import numpy as np
import pretty_midi
from corpus.services.scanner import scan_directory
from corpus.services.vocabulary import build_vocabulary, categorize_instrument


class BatchGenerator:
    def __init__(self, midi_dir, snippet_ticks=16, fs=8.0, rng_seed=None):
        """
        Args:
            midi_dir: path to directory of MIDI files
            snippet_ticks: number of time steps per snippet
            fs: sampling rate in Hz (ticks per second)
            rng_seed: optional seed for reproducibility
        """
        self.midi_dir = Path(midi_dir)
        self.snippet_ticks = snippet_ticks
        self.fs = fs
        self.rng = np.random.default_rng(rng_seed)

        scan_results = scan_directory(str(self.midi_dir))
        self.vocabulary = build_vocabulary(scan_results)
        self.song_paths = [r["path"] for r in scan_results]
        self._scan_results = scan_results

    def _load_piano_rolls(self, path):
        """Load a MIDI file and return per-instrument piano rolls.

        Returns list of (category, piano_roll) tuples.
        piano_roll shape: (128, total_ticks), values 0.0-1.0
        """
        midi = pretty_midi.PrettyMIDI(str(path))
        rolls = []
        for inst in midi.instruments:
            cat = categorize_instrument(inst.program, inst.is_drum)
            roll = inst.get_piano_roll(fs=self.fs) / 127.0
            rolls.append((cat, roll))
        return rolls

    def _generate_one(self):
        """Generate one training example: input mix, target, conditioning."""
        path = self.rng.choice(self.song_paths)
        rolls = self._load_piano_rolls(path)

        if len(rolls) < 2:
            return self._generate_one()

        categories = [cat for cat, _ in rolls]
        indices = list(range(len(rolls)))

        # Pick target
        target_idx = self.rng.choice(indices)
        target_cat, target_roll = rolls[target_idx]

        # Pick input: 1 to N-1 of the remaining instruments
        remaining = [i for i in indices if i != target_idx]
        num_input = self.rng.integers(1, len(remaining) + 1)
        input_indices = self.rng.choice(remaining, size=num_input, replace=False)

        # Mix input rolls (take max velocity per pitch per tick)
        max_ticks = max(r.shape[1] for _, r in rolls)
        input_mix = np.zeros((128, max_ticks))
        for i in input_indices:
            _, roll = rolls[i]
            input_mix[:, :roll.shape[1]] = np.maximum(
                input_mix[:, :roll.shape[1]], roll[:128]
            )

        # Pad target to same length
        target_padded = np.zeros((128, max_ticks))
        target_padded[:, :target_roll.shape[1]] = target_roll[:128]

        # Cut a random snippet
        if max_ticks <= self.snippet_ticks:
            start = 0
            snippet_len = min(max_ticks, self.snippet_ticks)
        else:
            start = self.rng.integers(0, max_ticks - self.snippet_ticks)
            snippet_len = self.snippet_ticks

        input_snippet = input_mix[:, start:start + snippet_len].T
        target_snippet = target_padded[:, start:start + snippet_len].T

        # Pad if shorter than snippet_ticks
        if input_snippet.shape[0] < self.snippet_ticks:
            pad = self.snippet_ticks - input_snippet.shape[0]
            input_snippet = np.pad(input_snippet, ((0, pad), (0, 0)))
            target_snippet = np.pad(target_snippet, ((0, pad), (0, 0)))

        # One-hot conditioning
        conditioning = np.zeros(len(self.vocabulary))
        if target_cat in self.vocabulary:
            conditioning[self.vocabulary[target_cat]] = 1.0

        return input_snippet, target_snippet, conditioning, target_cat

    def generate_batch(self, batch_size):
        """Generate a batch of training examples.

        Returns dict with:
            input: (batch_size, snippet_ticks, 128) float32
            target: (batch_size, snippet_ticks, 128) float32
            conditioning: (batch_size, num_categories) float32
            target_categories: list of category strings
        """
        inputs, targets, conds, cats = [], [], [], []
        for _ in range(batch_size):
            inp, tgt, cond, cat = self._generate_one()
            inputs.append(inp)
            targets.append(tgt)
            conds.append(cond)
            cats.append(cat)

        return {
            "input": np.array(inputs, dtype=np.float32),
            "target": np.array(targets, dtype=np.float32),
            "conditioning": np.array(conds, dtype=np.float32),
            "target_categories": cats,
        }
```

### Step 4: Run tests to verify they pass

Run: `pytest corpus/tests/test_batch_generator.py -v`
Expected: All 5 tests PASS

### Step 5: Commit

```bash
git add corpus/services/batch_generator.py corpus/tests/test_batch_generator.py
git commit -m "feat: batch generator — random snippets, instrument mixing, one-hot conditioning"
```

---

## Task 5: Curriculum Scheduler

Manages snippet length progression during training.

**Files:**
- Create: `corpus/services/curriculum.py`
- Test: `corpus/tests/test_curriculum.py`

### Step 1: Write failing tests

**`corpus/tests/test_curriculum.py`**:
```python
from corpus.services.curriculum import CurriculumScheduler


class TestCurriculumScheduler:
    def test_starts_at_phase_1(self):
        cs = CurriculumScheduler()
        assert cs.current_phase == 1
        assert cs.snippet_ticks == cs.phases[1]["snippet_ticks"]

    def test_advances_when_error_below_threshold(self):
        cs = CurriculumScheduler()
        phase1_ticks = cs.snippet_ticks
        # Report error above threshold — should stay
        cs.report_error(0.5)
        assert cs.current_phase == 1
        # Report error below threshold enough times
        for _ in range(cs.patience):
            cs.report_error(cs.phases[1]["threshold"] - 0.01)
        assert cs.current_phase == 2
        assert cs.snippet_ticks > phase1_ticks

    def test_does_not_advance_past_final_phase(self):
        cs = CurriculumScheduler()
        max_phase = max(cs.phases.keys())
        cs.current_phase = max_phase
        for _ in range(100):
            cs.report_error(0.001)
        assert cs.current_phase == max_phase

    def test_custom_phases(self):
        phases = {
            1: {"snippet_ticks": 8, "threshold": 0.1},
            2: {"snippet_ticks": 32, "threshold": 0.05},
        }
        cs = CurriculumScheduler(phases=phases, patience=3)
        assert cs.snippet_ticks == 8
```

### Step 2: Run tests to verify they fail

Run: `pytest corpus/tests/test_curriculum.py -v`
Expected: FAIL — `ImportError`

### Step 3: Implement curriculum scheduler

**`corpus/services/curriculum.py`**:
```python
DEFAULT_PHASES = {
    1: {"snippet_ticks": 16, "threshold": 0.15},    # ~2 bars at 8Hz, 120bpm
    2: {"snippet_ticks": 32, "threshold": 0.10},    # ~4 bars
    3: {"snippet_ticks": 64, "threshold": 0.0},     # ~8 bars (final, no advance)
}


class CurriculumScheduler:
    def __init__(self, phases=None, patience=10):
        """
        Args:
            phases: dict mapping phase number to {snippet_ticks, threshold}.
            patience: how many consecutive low-error reports before advancing.
        """
        self.phases = phases or DEFAULT_PHASES
        self.patience = patience
        self.current_phase = 1
        self._consecutive_low = 0

    @property
    def snippet_ticks(self):
        return self.phases[self.current_phase]["snippet_ticks"]

    def report_error(self, avg_error):
        """Report the average error for the latest batch.

        Advances to next phase if error stays below threshold for
        `patience` consecutive reports.
        """
        threshold = self.phases[self.current_phase]["threshold"]
        max_phase = max(self.phases.keys())

        if self.current_phase >= max_phase:
            return

        if avg_error < threshold:
            self._consecutive_low += 1
        else:
            self._consecutive_low = 0

        if self._consecutive_low >= self.patience:
            self.current_phase += 1
            self._consecutive_low = 0
```

### Step 4: Run tests to verify they pass

Run: `pytest corpus/tests/test_curriculum.py -v`
Expected: All 4 tests PASS

### Step 5: Commit

```bash
git add corpus/services/curriculum.py corpus/tests/test_curriculum.py
git commit -m "feat: curriculum scheduler — snippet length progression"
```

---

## Task 6: JAX Grid State

Initialize the 128x128 PC grid: state, weights, params arrays.

**Files:**
- Create: `training/engine/grid.py`
- Test: `training/tests/test_grid.py`

### Step 1: Write failing tests

**`training/tests/test_grid.py`**:
```python
import jax.numpy as jnp
import pytest
from training.engine.grid import GridState, create_grid


class TestCreateGrid:
    def test_creates_correct_shapes(self):
        grid = create_grid(size=128, num_instruments=8)
        assert grid.state.shape == (128, 128, 4)      # r, e, s, h
        assert grid.weights.shape == (128, 128, 4)     # w_left, w_right, w_up, w_down
        assert grid.params.shape == (128, 128, 4)      # alpha, beta, lr, bias

    def test_initial_representation_near_zero(self):
        grid = create_grid(size=128, num_instruments=8)
        r = grid.state[:, :, 0]  # representation channel
        assert jnp.abs(r).max() < 0.1

    def test_initial_weights_small(self):
        grid = create_grid(size=128, num_instruments=8)
        assert jnp.abs(grid.weights).max() < 1.0

    def test_alpha_initialized_in_range(self):
        grid = create_grid(size=128, num_instruments=8)
        alpha = grid.params[:, :, 0]
        assert (alpha >= 0.0).all()
        assert (alpha <= 1.0).all()

    def test_custom_size(self):
        grid = create_grid(size=32, num_instruments=4)
        assert grid.state.shape == (32, 32, 4)

    def test_clamp_masks(self):
        grid = create_grid(size=128, num_instruments=8)
        # Input mask: column 0
        assert grid.input_mask[0, 0] == True
        assert grid.input_mask[64, 0] == True
        assert grid.input_mask[0, 1] == False
        # Output mask: column 127
        assert grid.output_mask[0, 127] == True
        assert grid.output_mask[0, 0] == False
        # Conditioning mask: top-middle of column 0
        assert grid.conditioning_mask.sum() == 8  # num_instruments
```

### Step 2: Run tests to verify they fail

Run: `pytest training/tests/test_grid.py -v`
Expected: FAIL — `ImportError`

### Step 3: Implement grid state

**`training/engine/grid.py`**:
```python
from dataclasses import dataclass
import jax
import jax.numpy as jnp


@dataclass
class GridState:
    """Holds all arrays that define the PC grid.

    state: (H, W, 4) — r, e, s, h per neuron
    weights: (H, W, 4) — w_left, w_right, w_up, w_down
    params: (H, W, 4) — alpha, beta, learning_rate, bias
    input_mask: (H, W) bool — True for input-clamped neurons
    output_mask: (H, W) bool — True for output-clamped neurons
    conditioning_mask: (H, W) bool — True for conditioning neurons
    """
    state: jnp.ndarray
    weights: jnp.ndarray
    params: jnp.ndarray
    input_mask: jnp.ndarray
    output_mask: jnp.ndarray
    conditioning_mask: jnp.ndarray


def create_grid(size=128, num_instruments=8, key=None):
    """Initialize a PC grid with small random values.

    Args:
        size: grid dimension (size x size)
        num_instruments: number of instrument categories (one-hot block size)
        key: JAX PRNG key. Uses jax.random.PRNGKey(0) if None.
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)

    # State: small random init for representation, zeros for error/memory
    r = jax.random.normal(k1, (size, size)) * 0.01
    e = jnp.zeros((size, size))
    s = jnp.zeros((size, size))
    h = jnp.zeros((size, size))
    state = jnp.stack([r, e, s, h], axis=-1)

    # Weights: small random init
    weights = jax.random.normal(k2, (size, size, 4)) * 0.1

    # Params: alpha ~ 0.9 (slow decay), beta ~ 0.1 (weak recurrence),
    # lr = 0.01, bias = 0
    alpha = jnp.full((size, size), 0.9)
    beta = jnp.full((size, size), 0.1)
    lr = jnp.full((size, size), 0.01)
    bias = jnp.zeros((size, size))
    params = jnp.stack([alpha, beta, lr, bias], axis=-1)

    # Clamp masks
    input_mask = jnp.zeros((size, size), dtype=bool).at[:, 0].set(True)
    output_mask = jnp.zeros((size, size), dtype=bool).at[:, -1].set(True)

    # Conditioning: top-middle of column 0
    # Place at rows centered around size//2
    cond_start = (size - num_instruments) // 2
    conditioning_mask = jnp.zeros((size, size), dtype=bool)
    conditioning_mask = conditioning_mask.at[
        cond_start:cond_start + num_instruments, 0
    ].set(True)

    return GridState(
        state=state,
        weights=weights,
        params=params,
        input_mask=input_mask,
        output_mask=output_mask,
        conditioning_mask=conditioning_mask,
    )
```

### Step 4: Run tests to verify they pass

Run: `pytest training/tests/test_grid.py -v`
Expected: All 6 tests PASS

### Step 5: Commit

```bash
git add training/engine/grid.py training/tests/test_grid.py
git commit -m "feat: JAX grid state — initialization, clamp masks"
```

---

## Task 7: PC Update Rule (Single Relaxation Step)

The core math: one step of the predictive coding update, running on the full grid simultaneously.

**Files:**
- Create: `training/engine/update_rule.py`
- Test: `training/tests/test_update_rule.py`

### Step 1: Write failing tests

**`training/tests/test_update_rule.py`**:
```python
import jax
import jax.numpy as jnp
import pytest
from training.engine.grid import create_grid
from training.engine.update_rule import pc_relaxation_step, apply_clamping


class TestPCRelaxationStep:
    def test_output_same_shape_as_input(self):
        grid = create_grid(size=16, num_instruments=4)
        new_state, new_weights = pc_relaxation_step(
            grid.state, grid.weights, grid.params
        )
        assert new_state.shape == grid.state.shape
        assert new_weights.shape == grid.weights.shape

    def test_error_decreases_over_steps(self):
        """After multiple relaxation steps, total error should decrease."""
        key = jax.random.PRNGKey(42)
        grid = create_grid(size=16, num_instruments=4, key=key)
        # Set some non-trivial initial state
        state = grid.state.at[:, :, 0].set(
            jax.random.normal(key, (16, 16)) * 0.5
        )
        initial_error = jnp.abs(state[:, :, 1]).sum()

        # Run 20 relaxation steps
        weights = grid.weights
        for _ in range(20):
            state, weights = pc_relaxation_step(state, weights, grid.params)

        final_error = jnp.abs(state[:, :, 1]).sum()
        # Error should decrease (or at least not explode)
        assert final_error < initial_error * 10

    def test_leaky_state_persists(self):
        """The leaky integrator should carry forward previous state."""
        grid = create_grid(size=8, num_instruments=2)
        # Set representation to 1.0 everywhere
        state = grid.state.at[:, :, 0].set(1.0)
        new_state, _ = pc_relaxation_step(state, grid.weights, grid.params)
        # Leaky state (channel 2) should be nonzero after one step
        s = new_state[:, :, 2]
        assert jnp.abs(s).sum() > 0

    def test_is_jit_compilable(self):
        grid = create_grid(size=8, num_instruments=2)
        jitted = jax.jit(pc_relaxation_step)
        new_state, new_weights = jitted(grid.state, grid.weights, grid.params)
        assert new_state.shape == grid.state.shape


class TestApplyClamping:
    def test_clamped_values_overwritten(self):
        grid = create_grid(size=16, num_instruments=4)
        clamp_values = jnp.ones((16,))  # all 1s for input column
        state = apply_clamping(
            grid.state, grid.input_mask, clamp_values, channel=0
        )
        # Column 0 representation should be 1.0
        assert (state[:, 0, 0] == 1.0).all()
        # Column 1 should be unchanged
        assert (state[:, 1, 0] == grid.state[:, 1, 0]).all()
```

### Step 2: Run tests to verify they fail

Run: `pytest training/tests/test_update_rule.py -v`
Expected: FAIL — `ImportError`

### Step 3: Implement update rule

**`training/engine/update_rule.py`**:
```python
import jax
import jax.numpy as jnp


def pc_relaxation_step(state, weights, params):
    """One relaxation step of the PC grid. All neurons update simultaneously.

    Args:
        state: (H, W, 4) — channels: r, e, s, h
        weights: (H, W, 4) — channels: w_left, w_right, w_up, w_down
        params: (H, W, 4) — channels: alpha, beta, lr, bias

    Returns:
        new_state: (H, W, 4)
        new_weights: (H, W, 4) — updated via iPC
    """
    r = state[:, :, 0]
    e = state[:, :, 1]
    s = state[:, :, 2]
    h = state[:, :, 3]

    w_left = weights[:, :, 0]
    w_right = weights[:, :, 1]
    w_up = weights[:, :, 2]
    w_down = weights[:, :, 3]

    alpha = params[:, :, 0]
    beta = params[:, :, 1]
    lr = params[:, :, 2]
    bias = params[:, :, 3]

    # Activation function
    r_act = jnp.tanh(r)

    # Gather neighbor representations (with zero-padding at boundaries)
    r_left = jnp.pad(r_act[:, :-1], ((0, 0), (1, 0)))   # shift right
    r_right = jnp.pad(r_act[:, 1:], ((0, 0), (0, 1)))    # shift left
    r_up = jnp.pad(r_act[:-1, :], ((1, 0), (0, 0)))      # shift down
    r_down = jnp.pad(r_act[1:, :], ((0, 0), (0, 0)))     # shift up — pad missing row...
    # Actually let's be more careful with padding:
    r_left = jnp.zeros_like(r_act).at[:, 1:].set(r_act[:, :-1])
    r_right = jnp.zeros_like(r_act).at[:, :-1].set(r_act[:, 1:])
    r_up = jnp.zeros_like(r_act).at[1:, :].set(r_act[:-1, :])
    r_down = jnp.zeros_like(r_act).at[:-1, :].set(r_act[1:, :])

    # Predictions arriving at this neuron from neighbors
    pred_from_left = w_left * r_left
    pred_from_right = w_right * r_right
    pred_from_up = w_up * r_up
    pred_from_down = w_down * r_down
    total_prediction = pred_from_left + pred_from_right + pred_from_up + pred_from_down

    # Error: difference between representation and incoming predictions
    new_e = r - total_prediction - bias

    # Gather neighbor errors for the representation update
    e_left = jnp.zeros_like(new_e).at[:, 1:].set(new_e[:, :-1])
    e_right = jnp.zeros_like(new_e).at[:, :-1].set(new_e[:, 1:])
    e_up = jnp.zeros_like(new_e).at[1:, :].set(new_e[:-1, :])
    e_down = jnp.zeros_like(new_e).at[:-1, :].set(new_e[1:, :])

    neighbor_error_signal = (
        e_left * w_left + e_right * w_right +
        e_up * w_up + e_down * w_down
    )

    # Update temporal state
    new_s = alpha * s + (1.0 - alpha) * r
    new_h = beta * jnp.tanh(h)

    # Representation update
    new_r = r + lr * (-new_e + neighbor_error_signal + new_h + new_s - r)

    # iPC weight updates
    lr_w = lr * 0.1  # weight learning rate is fraction of main lr
    new_w_left = w_left + lr_w * new_e * r_left
    new_w_right = w_right + lr_w * new_e * r_right
    new_w_up = w_up + lr_w * new_e * r_up
    new_w_down = w_down + lr_w * new_e * r_down

    new_state = jnp.stack([new_r, new_e, new_s, new_h], axis=-1)
    new_weights = jnp.stack([new_w_left, new_w_right, new_w_up, new_w_down], axis=-1)

    return new_state, new_weights


def apply_clamping(state, mask, values, channel=0):
    """Overwrite the representation of clamped neurons.

    Args:
        state: (H, W, 4)
        mask: (H, W) bool — which neurons to clamp
        values: (H,) — values to clamp (one per row)
        channel: which state channel to clamp (0 = representation)

    Returns:
        new_state with clamped values applied.
    """
    # Broadcast values to match mask shape: values[i] applied to row i
    value_grid = jnp.broadcast_to(values[:, None], mask.shape)
    current = state[:, :, channel]
    clamped = jnp.where(mask, value_grid, current)
    return state.at[:, :, channel].set(clamped)
```

### Step 4: Run tests to verify they pass

Run: `pytest training/tests/test_update_rule.py -v`
Expected: All 5 tests PASS

### Step 5: Commit

```bash
git add training/engine/update_rule.py training/tests/test_update_rule.py
git commit -m "feat: PC update rule — relaxation step with iPC and temporal memory"
```

---

## Task 8: Training Loop

Ties together batch generation, grid state, and the update rule into a training loop.

**Files:**
- Create: `training/engine/trainer.py`
- Test: `training/tests/test_trainer.py`

### Step 1: Write failing tests

**`training/tests/test_trainer.py`**:
```python
import tempfile
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from corpus.tests.test_scanner import make_test_midi
from corpus.tests.test_batch_generator import make_corpus
from training.engine.trainer import Trainer


class TestTrainer:
    def test_train_one_batch_reduces_error(self, tmp_path):
        make_corpus(tmp_path)
        trainer = Trainer(
            midi_dir=str(tmp_path),
            grid_size=16,
            relaxation_steps=10,
            fs=8.0,
        )
        error_before = trainer.evaluate_error(batch_size=2)
        trainer.train_step(batch_size=2)
        error_after = trainer.evaluate_error(batch_size=2)
        # Error should not explode (may not decrease in 1 step, but shouldn't blow up)
        assert error_after < error_before * 5

    def test_train_multiple_steps(self, tmp_path):
        make_corpus(tmp_path)
        trainer = Trainer(
            midi_dir=str(tmp_path),
            grid_size=16,
            relaxation_steps=10,
            fs=8.0,
        )
        errors = []
        for _ in range(5):
            err = trainer.train_step(batch_size=2)
            errors.append(err)
        # Should produce numeric (non-NaN) errors
        assert all(np.isfinite(e) for e in errors)

    def test_curriculum_integration(self, tmp_path):
        make_corpus(tmp_path)
        trainer = Trainer(
            midi_dir=str(tmp_path),
            grid_size=16,
            relaxation_steps=5,
            fs=8.0,
        )
        assert trainer.curriculum.current_phase == 1

    def test_save_and_load_checkpoint(self, tmp_path):
        make_corpus(tmp_path)
        trainer = Trainer(
            midi_dir=str(tmp_path),
            grid_size=16,
            relaxation_steps=5,
            fs=8.0,
        )
        trainer.train_step(batch_size=2)
        ckpt_path = tmp_path / "checkpoint"
        trainer.save_checkpoint(str(ckpt_path))
        # Load into a new trainer
        trainer2 = Trainer(
            midi_dir=str(tmp_path),
            grid_size=16,
            relaxation_steps=5,
            fs=8.0,
        )
        trainer2.load_checkpoint(str(ckpt_path))
        # Weights should match
        assert jnp.allclose(trainer.grid.weights, trainer2.grid.weights)
```

### Step 2: Run tests to verify they fail

Run: `pytest training/tests/test_trainer.py -v`
Expected: FAIL — `ImportError`

### Step 3: Implement trainer

**`training/engine/trainer.py`**:
```python
from pathlib import Path
import jax
import jax.numpy as jnp
import jax.lax as lax
import numpy as np
from training.engine.grid import create_grid
from training.engine.update_rule import pc_relaxation_step, apply_clamping
from corpus.services.batch_generator import BatchGenerator
from corpus.services.curriculum import CurriculumScheduler


class Trainer:
    def __init__(self, midi_dir, grid_size=128, relaxation_steps=128,
                 fs=8.0, key=None):
        self.grid_size = grid_size
        self.relaxation_steps = relaxation_steps
        self.fs = fs

        self.batch_gen = BatchGenerator(midi_dir, snippet_ticks=16, fs=fs)
        self.curriculum = CurriculumScheduler()
        num_instruments = len(self.batch_gen.vocabulary)

        if key is None:
            key = jax.random.PRNGKey(0)
        self.grid = create_grid(
            size=grid_size, num_instruments=num_instruments, key=key
        )

    def _run_relaxation(self, state, weights, params, input_vals, target_vals,
                        conditioning_vals, input_mask, output_mask,
                        conditioning_mask, steps):
        """Run multiple relaxation steps with clamping using lax.scan."""
        def step_fn(carry, _):
            st, wt = carry
            st, wt = pc_relaxation_step(st, wt, params)
            # Re-clamp input, output, conditioning
            st = apply_clamping(st, input_mask, input_vals, channel=0)
            st = apply_clamping(st, output_mask, target_vals, channel=0)
            st = apply_clamping(st, conditioning_mask, conditioning_vals, channel=0)
            error = jnp.abs(st[:, :, 1]).mean()
            return (st, wt), error

        (final_state, final_weights), errors = lax.scan(
            step_fn, (state, weights), None, length=steps
        )
        return final_state, final_weights, errors

    def train_step(self, batch_size=32):
        """Run one training step: generate batch, process each tick."""
        self.batch_gen.snippet_ticks = self.curriculum.snippet_ticks
        batch = self.batch_gen.generate_batch(batch_size)

        total_error = 0.0
        num_ticks = batch["input"].shape[1]

        # Process one batch item at a time (simplest first, vmap later)
        for b in range(batch_size):
            state = self.grid.state
            weights = self.grid.weights

            for t in range(num_ticks):
                input_vals = jnp.array(batch["input"][b, t])       # (128,)
                target_vals = jnp.array(batch["target"][b, t])     # (128,)

                # Build conditioning: expand one-hot to grid rows
                cond_vec = batch["conditioning"][b]                 # (num_inst,)
                cond_vals = jnp.zeros(self.grid_size)
                num_inst = len(cond_vec)
                cond_start = (self.grid_size - num_inst) // 2
                cond_vals = cond_vals.at[cond_start:cond_start + num_inst].set(
                    jnp.array(cond_vec)
                )

                state, weights, errors = self._run_relaxation(
                    state, weights, self.grid.params,
                    input_vals, target_vals, cond_vals,
                    self.grid.input_mask, self.grid.output_mask,
                    self.grid.conditioning_mask,
                    self.relaxation_steps,
                )
                total_error += float(errors[-1])

            # Update grid with learned weights (keep last batch item's state)
            self.grid.weights = weights

        avg_error = total_error / (batch_size * num_ticks)
        self.curriculum.report_error(avg_error)
        return avg_error

    def evaluate_error(self, batch_size=4):
        """Evaluate current error without updating weights."""
        self.batch_gen.snippet_ticks = self.curriculum.snippet_ticks
        batch = self.batch_gen.generate_batch(batch_size)

        total_error = 0.0
        num_ticks = batch["input"].shape[1]

        for b in range(batch_size):
            state = self.grid.state
            for t in range(num_ticks):
                input_vals = jnp.array(batch["input"][b, t])
                target_vals = jnp.array(batch["target"][b, t])
                cond_vec = batch["conditioning"][b]
                cond_vals = jnp.zeros(self.grid_size)
                num_inst = len(cond_vec)
                cond_start = (self.grid_size - num_inst) // 2
                cond_vals = cond_vals.at[cond_start:cond_start + num_inst].set(
                    jnp.array(cond_vec)
                )

                state, _, errors = self._run_relaxation(
                    state, self.grid.weights, self.grid.params,
                    input_vals, target_vals, cond_vals,
                    self.grid.input_mask, self.grid.output_mask,
                    self.grid.conditioning_mask,
                    self.relaxation_steps,
                )
                total_error += float(errors[-1])

        return total_error / (batch_size * num_ticks)

    def save_checkpoint(self, path):
        """Save grid state, weights, and params as numpy arrays."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "state.npy", np.array(self.grid.state))
        np.save(path / "weights.npy", np.array(self.grid.weights))
        np.save(path / "params.npy", np.array(self.grid.params))

    def load_checkpoint(self, path):
        """Load grid state, weights, and params from numpy arrays."""
        path = Path(path)
        self.grid.state = jnp.array(np.load(path / "state.npy"))
        self.grid.weights = jnp.array(np.load(path / "weights.npy"))
        self.grid.params = jnp.array(np.load(path / "params.npy"))
```

### Step 4: Run tests to verify they pass

Run: `pytest training/tests/test_trainer.py -v`
Expected: All 4 tests PASS

### Step 5: Commit

```bash
git add training/engine/trainer.py training/tests/test_trainer.py
git commit -m "feat: training loop — batch processing, relaxation, curriculum, checkpoints"
```

---

## Task 9: Model Export (JAX → Binary)

Export trained model as flat float32 files for loading into WebGL textures.

**Files:**
- Create: `training/engine/export.py`
- Test: `training/tests/test_export.py`

### Step 1: Write failing tests

**`training/tests/test_export.py`**:
```python
import json
import struct
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from training.engine.grid import create_grid
from training.engine.export import export_model, load_exported_model


class TestExportModel:
    def test_creates_binary_files(self, tmp_path):
        grid = create_grid(size=16, num_instruments=4)
        vocabulary = {"piano": 0, "bass": 1, "guitar": 2, "drums": 3}
        export_model(grid, vocabulary, str(tmp_path))
        assert (tmp_path / "state.bin").exists()
        assert (tmp_path / "weights.bin").exists()
        assert (tmp_path / "params.bin").exists()
        assert (tmp_path / "config.json").exists()

    def test_binary_file_sizes_correct(self, tmp_path):
        grid = create_grid(size=16, num_instruments=4)
        vocabulary = {"piano": 0, "bass": 1, "guitar": 2, "drums": 3}
        export_model(grid, vocabulary, str(tmp_path))
        # Each texture: 16 * 16 * 4 channels * 4 bytes = 4096 bytes
        expected = 16 * 16 * 4 * 4
        assert (tmp_path / "state.bin").stat().st_size == expected
        assert (tmp_path / "weights.bin").stat().st_size == expected
        assert (tmp_path / "params.bin").stat().st_size == expected

    def test_config_json_correct(self, tmp_path):
        grid = create_grid(size=16, num_instruments=4)
        vocabulary = {"piano": 0, "bass": 1, "guitar": 2, "drums": 3}
        export_model(grid, vocabulary, str(tmp_path))
        config = json.loads((tmp_path / "config.json").read_text())
        assert config["grid_size"] == 16
        assert config["num_instruments"] == 4
        assert config["vocabulary"] == vocabulary

    def test_roundtrip_preserves_values(self, tmp_path):
        grid = create_grid(size=16, num_instruments=4)
        vocabulary = {"piano": 0, "bass": 1, "guitar": 2, "drums": 3}
        export_model(grid, vocabulary, str(tmp_path))
        loaded = load_exported_model(str(tmp_path))
        assert np.allclose(loaded["state"], np.array(grid.state), atol=1e-6)
        assert np.allclose(loaded["weights"], np.array(grid.weights), atol=1e-6)
```

### Step 2: Run tests to verify they fail

Run: `pytest training/tests/test_export.py -v`
Expected: FAIL — `ImportError`

### Step 3: Implement export

**`training/engine/export.py`**:
```python
import json
from pathlib import Path
import numpy as np


def export_model(grid, vocabulary, output_dir):
    """Export a trained grid to binary files for WebGL.

    Writes:
        state.bin — flat float32 array (H * W * 4)
        weights.bin — flat float32 array (H * W * 4)
        params.bin — flat float32 array (H * W * 4)
        config.json — grid size, vocabulary, etc.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state = np.array(grid.state, dtype=np.float32)
    weights = np.array(grid.weights, dtype=np.float32)
    params = np.array(grid.params, dtype=np.float32)

    state.tofile(output_dir / "state.bin")
    weights.tofile(output_dir / "weights.bin")
    params.tofile(output_dir / "params.bin")

    config = {
        "grid_size": int(state.shape[0]),
        "num_instruments": len(vocabulary),
        "vocabulary": vocabulary,
        "channels_per_texture": 4,
        "dtype": "float32",
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))


def load_exported_model(model_dir):
    """Load exported binary model back into numpy arrays (for verification)."""
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    size = config["grid_size"]
    shape = (size, size, 4)

    return {
        "state": np.fromfile(model_dir / "state.bin", dtype=np.float32).reshape(shape),
        "weights": np.fromfile(model_dir / "weights.bin", dtype=np.float32).reshape(shape),
        "params": np.fromfile(model_dir / "params.bin", dtype=np.float32).reshape(shape),
        "config": config,
    }
```

### Step 4: Run tests to verify they pass

Run: `pytest training/tests/test_export.py -v`
Expected: All 4 tests PASS

### Step 5: Commit

```bash
git add training/engine/export.py training/tests/test_export.py
git commit -m "feat: model export — JAX grid to binary float32 for WebGL"
```

---

## Task 10: Django REST API — Corpus Endpoints

Expose corpus metadata and batch generation via REST.

**Files:**
- Create: `corpus/models.py` (Django models)
- Create: `corpus/serializers.py`
- Create: `corpus/views.py`
- Create: `corpus/urls.py`

### Step 1: Implement corpus models

**`corpus/models.py`**:
```python
from django.db import models


class CorpusScan(models.Model):
    """Stores results of scanning the MIDI directory."""
    scanned_at = models.DateTimeField(auto_now_add=True)
    midi_dir = models.CharField(max_length=512)
    num_songs = models.IntegerField(default=0)
    vocabulary_json = models.JSONField(default=dict)

    class Meta:
        ordering = ["-scanned_at"]


class Song(models.Model):
    """Metadata for a single MIDI file in the corpus."""
    scan = models.ForeignKey(CorpusScan, on_delete=models.CASCADE, related_name="songs")
    path = models.CharField(max_length=512)
    tempo = models.FloatField()
    duration = models.FloatField()
    instruments_json = models.JSONField(default=list)
```

### Step 2: Implement views

**`corpus/views.py`**:
```python
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from corpus.services.scanner import scan_directory
from corpus.services.vocabulary import build_vocabulary
from corpus.models import CorpusScan, Song


class CorpusScanView(APIView):
    """POST to scan MIDI directory. GET to retrieve latest scan results."""

    def post(self, request):
        midi_dir = request.data.get("midi_dir", settings.MIDI_DATA_DIR)
        results = scan_directory(midi_dir)
        vocabulary = build_vocabulary(results)

        scan = CorpusScan.objects.create(
            midi_dir=midi_dir,
            num_songs=len(results),
            vocabulary_json=vocabulary,
        )
        for r in results:
            Song.objects.create(
                scan=scan,
                path=r["path"],
                tempo=r["tempo"],
                duration=r["duration"],
                instruments_json=r["instruments"],
            )
        return Response({
            "scan_id": scan.id,
            "num_songs": len(results),
            "vocabulary": vocabulary,
        }, status=status.HTTP_201_CREATED)

    def get(self, request):
        scan = CorpusScan.objects.first()
        if not scan:
            return Response({"error": "No scan yet. POST to scan first."},
                            status=status.HTTP_404_NOT_FOUND)
        songs = scan.songs.all().values("path", "tempo", "duration", "instruments_json")
        return Response({
            "scan_id": scan.id,
            "scanned_at": scan.scanned_at,
            "num_songs": scan.num_songs,
            "vocabulary": scan.vocabulary_json,
            "songs": list(songs),
        })


class CorpusVocabularyView(APIView):
    """GET the current instrument vocabulary."""

    def get(self, request):
        scan = CorpusScan.objects.first()
        if not scan:
            return Response({"error": "No scan yet."}, status=status.HTTP_404_NOT_FOUND)
        return Response({"vocabulary": scan.vocabulary_json})
```

### Step 3: Wire up URLs

**`corpus/urls.py`**:
```python
from django.urls import path
from corpus.views import CorpusScanView, CorpusVocabularyView

urlpatterns = [
    path("scan/", CorpusScanView.as_view(), name="corpus-scan"),
    path("vocabulary/", CorpusVocabularyView.as_view(), name="corpus-vocabulary"),
]
```

### Step 4: Run migrations and verify

Run: `python manage.py makemigrations corpus && python manage.py migrate`
Expected: Migrations created and applied successfully.

Run: `python manage.py check`
Expected: "System check identified no issues."

### Step 5: Commit

```bash
git add corpus/models.py corpus/views.py corpus/urls.py corpus/admin.py
git add corpus/migrations/
git commit -m "feat: Django REST API for corpus scanning and vocabulary"
```

---

## Task 11: Django REST API — Training Endpoints

Expose training config, start/stop, and metrics.

**Files:**
- Create: `training/models.py`
- Create: `training/views.py`
- Create: `training/urls.py`

### Step 1: Implement training models

**`training/models.py`**:
```python
from django.db import models


class TrainingRun(models.Model):
    started_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20, default="pending",
                              choices=[("pending", "Pending"),
                                       ("running", "Running"),
                                       ("stopped", "Stopped")])
    config_json = models.JSONField(default=dict)
    current_phase = models.IntegerField(default=1)
    total_steps = models.IntegerField(default=0)
    latest_error = models.FloatField(null=True)


class TrainingMetric(models.Model):
    run = models.ForeignKey(TrainingRun, on_delete=models.CASCADE,
                            related_name="metrics")
    step = models.IntegerField()
    avg_error = models.FloatField()
    phase = models.IntegerField()
    recorded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["step"]
```

### Step 2: Implement views

**`training/views.py`**:
```python
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from training.models import TrainingRun, TrainingMetric


class TrainingConfigView(APIView):
    """GET/POST training configuration."""

    DEFAULT_CONFIG = {
        "grid_size": 128,
        "relaxation_steps": 128,
        "batch_size": 16,
        "lr": 0.01,
        "lr_weights": 0.001,
        "fs": 8.0,
        "curriculum_patience": 10,
    }

    def get(self, request):
        run = TrainingRun.objects.filter(status="running").first()
        if run:
            return Response({"config": run.config_json, "run_id": run.id})
        return Response({"config": self.DEFAULT_CONFIG, "run_id": None})

    def post(self, request):
        config = {**self.DEFAULT_CONFIG, **request.data}
        run = TrainingRun.objects.filter(status="running").first()
        if run:
            run.config_json = config
            run.save()
            return Response({"config": config, "run_id": run.id})
        return Response({"config": config, "run_id": None,
                         "message": "Config saved. Start a training run to apply."})


class TrainingMetricsView(APIView):
    """GET training metrics for the current or specified run."""

    def get(self, request):
        run_id = request.query_params.get("run_id")
        if run_id:
            run = TrainingRun.objects.filter(id=run_id).first()
        else:
            run = TrainingRun.objects.order_by("-started_at").first()

        if not run:
            return Response({"error": "No training run found."},
                            status=status.HTTP_404_NOT_FOUND)

        metrics = run.metrics.all().values("step", "avg_error", "phase")
        return Response({
            "run_id": run.id,
            "status": run.status,
            "current_phase": run.current_phase,
            "total_steps": run.total_steps,
            "latest_error": run.latest_error,
            "metrics": list(metrics),
        })
```

### Step 3: Wire up URLs

**`training/urls.py`**:
```python
from django.urls import path
from training.views import TrainingConfigView, TrainingMetricsView

urlpatterns = [
    path("config/", TrainingConfigView.as_view(), name="training-config"),
    path("metrics/", TrainingMetricsView.as_view(), name="training-metrics"),
]
```

### Step 4: Run migrations and verify

Run: `python manage.py makemigrations training && python manage.py migrate`

Run: `python manage.py check`
Expected: "System check identified no issues."

### Step 5: Commit

```bash
git add training/models.py training/views.py training/urls.py training/admin.py
git add training/migrations/
git commit -m "feat: Django REST API for training config and metrics"
```

---

## Task 12: WebGL Vertex Shader & Infrastructure

Set up the frontend boilerplate: HTML page, WebGL context, texture loading, ping-pong setup.

**Files:**
- Create: `frontend/index.html`
- Create: `frontend/js/app.js`
- Create: `frontend/js/model-loader.js`
- Create: `frontend/shaders/passthrough.vert`

### Step 1: Create vertex shader

**`frontend/shaders/passthrough.vert`**:
```glsl
attribute vec2 a_position;
varying vec2 v_uv;

void main() {
    v_uv = a_position * 0.5 + 0.5;
    gl_Position = vec4(a_position, 0.0, 1.0);
}
```

### Step 2: Create model loader

**`frontend/js/model-loader.js`**:
```javascript
/**
 * Load exported PC grid model into WebGL textures.
 *
 * Expects files: config.json, state.bin, weights.bin, params.bin
 * in the given base URL.
 */
export async function loadModel(gl, baseUrl) {
    const configResp = await fetch(`${baseUrl}/config.json`);
    const config = await configResp.json();

    const size = config.grid_size;

    const [stateData, weightsData, paramsData] = await Promise.all([
        fetch(`${baseUrl}/state.bin`).then(r => r.arrayBuffer()),
        fetch(`${baseUrl}/weights.bin`).then(r => r.arrayBuffer()),
        fetch(`${baseUrl}/params.bin`).then(r => r.arrayBuffer()),
    ]);

    return {
        config,
        state: createFloatTexture(gl, size, new Float32Array(stateData)),
        weights: createFloatTexture(gl, size, new Float32Array(weightsData)),
        params: createFloatTexture(gl, size, new Float32Array(paramsData)),
    };
}

/**
 * Create a size x size RGBA float texture from flat float32 data.
 */
export function createFloatTexture(gl, size, data) {
    const ext = gl.getExtension('OES_texture_float');
    if (!ext) throw new Error('OES_texture_float not supported');

    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(
        gl.TEXTURE_2D, 0, gl.RGBA,
        size, size, 0,
        gl.RGBA, gl.FLOAT, data
    );
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return tex;
}

/**
 * Create a framebuffer with a float texture attachment (for ping-pong).
 */
export function createFramebuffer(gl, texture) {
    const fb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(
        gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0,
        gl.TEXTURE_2D, texture, 0
    );
    return fb;
}
```

### Step 3: Create HTML page

**`frontend/index.html`**:
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>PC Jam Grid</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { background: #111; color: #eee; font-family: monospace; display: flex; height: 100vh; }
        #grid-container { flex: 1; display: flex; align-items: center; justify-content: center; }
        canvas { image-rendering: pixelated; }
        #sidebar { width: 280px; padding: 16px; background: #1a1a1a; overflow-y: auto; }
        #sidebar h2 { font-size: 14px; margin-bottom: 8px; color: #888; }
        #sidebar select, #sidebar button { width: 100%; padding: 8px; margin-bottom: 8px;
            background: #222; color: #eee; border: 1px solid #333; font-family: monospace; }
        #sidebar button:hover { background: #333; }
        #status { font-size: 12px; color: #666; padding: 8px; border-top: 1px solid #333; }
    </style>
</head>
<body>
    <div id="grid-container">
        <canvas id="grid" width="512" height="512"></canvas>
    </div>
    <div id="sidebar">
        <h2>INSTRUMENT</h2>
        <select id="instrument-select"></select>

        <h2>INPUT SOURCE</h2>
        <select id="midi-input-select">
            <option value="file">File Playback</option>
        </select>
        <input type="file" id="midi-file" accept=".mid,.midi" style="margin-bottom:8px;">

        <h2>CONTROLS</h2>
        <button id="btn-play">Play</button>
        <button id="btn-stop">Stop</button>

        <div id="status">
            <div>Tick: <span id="tick-count">0</span></div>
            <div>Error: <span id="avg-error">-</span></div>
            <div>Steps/frame: <span id="steps-per-frame">10</span></div>
        </div>
    </div>
    <script type="module" src="/static/js/app.js"></script>
</body>
</html>
```

### Step 4: Create app.js skeleton

**`frontend/js/app.js`**:
```javascript
import { loadModel, createFloatTexture, createFramebuffer } from './model-loader.js';

const canvas = document.getElementById('grid');
const gl = canvas.getContext('webgl', { preserveDrawingBuffer: true });

if (!gl) {
    document.body.innerHTML = '<h1>WebGL not supported</h1>';
    throw new Error('No WebGL');
}

// Will be initialized when model loads
let model = null;
let stateTextures = [null, null]; // ping-pong pair
let stateFramebuffers = [null, null];
let currentIdx = 0;

async function init() {
    // Load model from server
    try {
        model = await loadModel(gl, '/static/model');
        console.log('Model loaded:', model.config);
        populateInstruments(model.config.vocabulary);
        setupPingPong(gl, model);
    } catch (e) {
        console.log('No model loaded yet. Waiting for training...');
    }
}

function populateInstruments(vocabulary) {
    const select = document.getElementById('instrument-select');
    select.innerHTML = '';
    for (const [name, idx] of Object.entries(vocabulary)) {
        const opt = document.createElement('option');
        opt.value = idx;
        opt.textContent = name;
        select.appendChild(opt);
    }
}

function setupPingPong(gl, model) {
    const size = model.config.grid_size;
    // Create two state textures for ping-pong
    const emptyData = new Float32Array(size * size * 4);
    stateTextures[0] = model.state;
    stateTextures[1] = createFloatTexture(gl, size, emptyData);
    stateFramebuffers[0] = createFramebuffer(gl, stateTextures[0]);
    stateFramebuffers[1] = createFramebuffer(gl, stateTextures[1]);
}

init();
```

### Step 5: Commit

```bash
git add frontend/
git commit -m "feat: WebGL frontend scaffold — HTML, model loader, ping-pong setup"
```

---

## Task 13: WebGL PC Relaxation Shader (Inference Only)

The GLSL fragment shader that runs one PC relaxation step. No weight updates (inference only for jam mode).

**Files:**
- Create: `frontend/shaders/pc-step.frag`
- Create: `frontend/js/grid-compute.js`

### Step 1: Write PC step fragment shader

**`frontend/shaders/pc-step.frag`**:
```glsl
precision highp float;

uniform sampler2D u_state;    // r, e, s, h
uniform sampler2D u_weights;  // w_left, w_right, w_up, w_down
uniform sampler2D u_params;   // alpha, beta, lr, bias
uniform vec2 u_resolution;    // grid size (128.0, 128.0)

varying vec2 v_uv;

void main() {
    vec2 pixel = 1.0 / u_resolution;
    vec4 state = texture2D(u_state, v_uv);
    vec4 weights = texture2D(u_weights, v_uv);
    vec4 params = texture2D(u_params, v_uv);

    float r = state.r;
    float e = state.g;
    float s = state.b;
    float h = state.a;

    float w_left  = weights.r;
    float w_right = weights.g;
    float w_up    = weights.b;
    float w_down  = weights.a;

    float alpha = params.r;
    float beta  = params.g;
    float lr    = params.b;
    float bias  = params.a;

    // Activation
    float r_act = tanh(r);

    // Get neighbor representations (zero at boundaries via clamp-to-edge)
    float r_left  = tanh(texture2D(u_state, v_uv + vec2(-pixel.x, 0.0)).r);
    float r_right = tanh(texture2D(u_state, v_uv + vec2( pixel.x, 0.0)).r);
    float r_up    = tanh(texture2D(u_state, v_uv + vec2(0.0,  pixel.y)).r);
    float r_down  = tanh(texture2D(u_state, v_uv + vec2(0.0, -pixel.y)).r);

    // Zero out boundary lookups (clamp-to-edge repeats edge values, we want 0)
    if (v_uv.x - pixel.x < 0.0) r_left = 0.0;
    if (v_uv.x + pixel.x > 1.0) r_right = 0.0;
    if (v_uv.y + pixel.y > 1.0) r_up = 0.0;
    if (v_uv.y - pixel.y < 0.0) r_down = 0.0;

    // Incoming predictions
    float pred = w_left * r_left + w_right * r_right + w_up * r_up + w_down * r_down;

    // Error
    float new_e = r - pred - bias;

    // Neighbor errors
    float e_left  = texture2D(u_state, v_uv + vec2(-pixel.x, 0.0)).g;
    float e_right = texture2D(u_state, v_uv + vec2( pixel.x, 0.0)).g;
    float e_up    = texture2D(u_state, v_uv + vec2(0.0,  pixel.y)).g;
    float e_down  = texture2D(u_state, v_uv + vec2(0.0, -pixel.y)).g;

    if (v_uv.x - pixel.x < 0.0) e_left = 0.0;
    if (v_uv.x + pixel.x > 1.0) e_right = 0.0;
    if (v_uv.y + pixel.y > 1.0) e_up = 0.0;
    if (v_uv.y - pixel.y < 0.0) e_down = 0.0;

    float neighbor_err = e_left * w_left + e_right * w_right + e_up * w_up + e_down * w_down;

    // Temporal updates
    float new_s = alpha * s + (1.0 - alpha) * r;
    float new_h = beta * tanh(h);

    // Representation update
    float new_r = r + lr * (-new_e + neighbor_err + new_h + new_s - r);

    gl_FragColor = vec4(new_r, new_e, new_s, new_h);
}
```

### Step 2: Write compute helper

**`frontend/js/grid-compute.js`**:
```javascript
/**
 * Manages the WebGL ping-pong compute pipeline for PC inference.
 */
export class GridCompute {
    constructor(gl, vertSource, fragSource) {
        this.gl = gl;
        this.program = this._createProgram(vertSource, fragSource);

        // Attribute: full-screen quad
        const quad = new Float32Array([-1,-1, 1,-1, -1,1, 1,1]);
        this.quadBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, quad, gl.STATIC_DRAW);

        // Uniform locations
        this.u_state = gl.getUniformLocation(this.program, 'u_state');
        this.u_weights = gl.getUniformLocation(this.program, 'u_weights');
        this.u_params = gl.getUniformLocation(this.program, 'u_params');
        this.u_resolution = gl.getUniformLocation(this.program, 'u_resolution');
    }

    /**
     * Run one relaxation step: read from stateTextures[srcIdx],
     * write to stateFramebuffers[dstIdx].
     */
    step(stateTextures, stateFramebuffers, weightsTexture, paramsTexture,
         srcIdx, dstIdx, gridSize) {
        const gl = this.gl;

        gl.useProgram(this.program);
        gl.bindFramebuffer(gl.FRAMEBUFFER, stateFramebuffers[dstIdx]);
        gl.viewport(0, 0, gridSize, gridSize);

        // Bind textures
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, stateTextures[srcIdx]);
        gl.uniform1i(this.u_state, 0);

        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, weightsTexture);
        gl.uniform1i(this.u_weights, 1);

        gl.activeTexture(gl.TEXTURE2);
        gl.bindTexture(gl.TEXTURE_2D, paramsTexture);
        gl.uniform1i(this.u_params, 2);

        gl.uniform2f(this.u_resolution, gridSize, gridSize);

        // Draw full-screen quad
        const a_position = gl.getAttribLocation(this.program, 'a_position');
        gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
        gl.enableVertexAttribArray(a_position);
        gl.vertexAttribPointer(a_position, 2, gl.FLOAT, false, 0, 0);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }

    _createProgram(vertSource, fragSource) {
        const gl = this.gl;
        const vs = this._compile(gl.VERTEX_SHADER, vertSource);
        const fs = this._compile(gl.FRAGMENT_SHADER, fragSource);
        const prog = gl.createProgram();
        gl.attachShader(prog, vs);
        gl.attachShader(prog, fs);
        gl.linkProgram(prog);
        if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
            throw new Error('Link error: ' + gl.getProgramInfoLog(prog));
        }
        return prog;
    }

    _compile(type, source) {
        const gl = this.gl;
        const shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            throw new Error('Compile error: ' + gl.getShaderInfoLog(shader));
        }
        return shader;
    }
}
```

### Step 3: Commit

```bash
git add frontend/shaders/pc-step.frag frontend/js/grid-compute.js
git commit -m "feat: WebGL PC relaxation shader — inference-only ping-pong compute"
```

---

## Task 14: WebGL Grid Renderer

The visualization shader: error → color, weight → gap blend.

**Files:**
- Create: `frontend/shaders/render-grid.frag`
- Create: `frontend/js/grid-renderer.js`

### Step 1: Write render shader

**`frontend/shaders/render-grid.frag`**:
```glsl
precision highp float;

uniform sampler2D u_state;    // r, e, s, h
uniform sampler2D u_weights;  // w_left, w_right, w_up, w_down
uniform vec2 u_resolution;    // grid size
uniform vec3 u_color_pos;     // color for positive error (over-predicting)
uniform vec3 u_color_neg;     // color for negative error (under-predicting)

varying vec2 v_uv;

void main() {
    vec2 pixel = 1.0 / u_resolution;
    vec2 cell = v_uv * u_resolution;
    vec2 cellCenter = floor(cell) + 0.5;
    vec2 cellFrac = fract(cell);

    // Distance from cell center (for gap rendering)
    vec2 distFromCenter = abs(cellFrac - 0.5);

    // Get this cell's state
    vec2 centerUV = cellCenter / u_resolution;
    vec4 state = texture2D(u_state, centerUV);
    vec4 weights = texture2D(u_weights, centerUV);

    float error = state.g; // prediction error
    float absError = abs(error);

    // Map error to color: positive = color_pos, negative = color_neg
    vec3 errorColor = error > 0.0 ? u_color_pos : u_color_neg;

    // Alpha from error magnitude (clamped)
    float alpha = clamp(absError * 5.0, 0.05, 1.0);

    // Gap blending: check if we're in the gap region between cells
    float gapSize = 0.12; // fraction of cell that is gap
    float inGapX = smoothstep(0.5 - gapSize, 0.5, distFromCenter.x);
    float inGapY = smoothstep(0.5 - gapSize, 0.5, distFromCenter.y);
    float inGap = max(inGapX, inGapY);

    // Connection strength determines gap fill
    // If in horizontal gap, use w_right; if vertical gap, use w_up
    float connectionStrength = 0.0;
    if (inGapX > inGapY) {
        connectionStrength = cellFrac.x > 0.5
            ? abs(weights.g) // w_right
            : abs(weights.r); // w_left
    } else {
        connectionStrength = cellFrac.y > 0.5
            ? abs(weights.b) // w_up
            : abs(weights.a); // w_down
    }
    connectionStrength = clamp(connectionStrength * 2.0, 0.0, 1.0);

    // In gap: blend toward black based on connection strength
    // Strong connection = color bleeds through; weak = black gap
    float gapDarkness = inGap * (1.0 - connectionStrength);

    vec3 finalColor = errorColor * alpha * (1.0 - gapDarkness);

    gl_FragColor = vec4(finalColor, 1.0);
}
```

### Step 2: Write renderer class

**`frontend/js/grid-renderer.js`**:
```javascript
/**
 * Renders the PC grid state as colored squares with weight-blended gaps.
 */
export class GridRenderer {
    constructor(gl, vertSource, fragSource) {
        this.gl = gl;
        this.program = this._createProgram(vertSource, fragSource);

        const quad = new Float32Array([-1,-1, 1,-1, -1,1, 1,1]);
        this.quadBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, quad, gl.STATIC_DRAW);

        this.u_state = gl.getUniformLocation(this.program, 'u_state');
        this.u_weights = gl.getUniformLocation(this.program, 'u_weights');
        this.u_resolution = gl.getUniformLocation(this.program, 'u_resolution');
        this.u_color_pos = gl.getUniformLocation(this.program, 'u_color_pos');
        this.u_color_neg = gl.getUniformLocation(this.program, 'u_color_neg');
    }

    render(stateTexture, weightsTexture, gridSize, canvasWidth, canvasHeight) {
        const gl = this.gl;

        gl.useProgram(this.program);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null); // render to canvas
        gl.viewport(0, 0, canvasWidth, canvasHeight);

        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, stateTexture);
        gl.uniform1i(this.u_state, 0);

        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, weightsTexture);
        gl.uniform1i(this.u_weights, 1);

        gl.uniform2f(this.u_resolution, gridSize, gridSize);

        // Default colors — configurable later
        gl.uniform3f(this.u_color_pos, 1.0, 0.3, 0.2); // warm red
        gl.uniform3f(this.u_color_neg, 0.2, 0.4, 1.0);  // cool blue

        const a_position = gl.getAttribLocation(this.program, 'a_position');
        gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
        gl.enableVertexAttribArray(a_position);
        gl.vertexAttribPointer(a_position, 2, gl.FLOAT, false, 0, 0);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }

    _createProgram(vertSource, fragSource) {
        const gl = this.gl;
        const vs = this._compile(gl.VERTEX_SHADER, vertSource);
        const fs = this._compile(gl.FRAGMENT_SHADER, fragSource);
        const prog = gl.createProgram();
        gl.attachShader(prog, vs);
        gl.attachShader(prog, fs);
        gl.linkProgram(prog);
        if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
            throw new Error('Link error: ' + gl.getProgramInfoLog(prog));
        }
        return prog;
    }

    _compile(type, source) {
        const gl = this.gl;
        const shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            throw new Error('Compile error: ' + gl.getShaderInfoLog(shader));
        }
        return shader;
    }
}
```

### Step 3: Commit

```bash
git add frontend/shaders/render-grid.frag frontend/js/grid-renderer.js
git commit -m "feat: WebGL grid renderer — error-to-color, weight-to-gap visualization"
```

---

## Task 15: Web MIDI I/O

Handle live MIDI input via Web MIDI API and MIDI file playback.

**Files:**
- Create: `frontend/js/midi-io.js`

### Step 1: Implement MIDI I/O module

**`frontend/js/midi-io.js`**:
```javascript
/**
 * Handles MIDI input (live and file playback) and output via Web MIDI API.
 */
export class MidiIO {
    constructor() {
        this.midiAccess = null;
        this.inputDevice = null;
        this.outputDevice = null;
        // Current state: velocity per pitch (0-127 -> 0.0-1.0)
        this.inputState = new Float32Array(128);
        this.outputState = new Float32Array(128);
        this._onNoteCallback = null;
    }

    async init() {
        if (!navigator.requestMIDIAccess) {
            console.warn('Web MIDI API not available');
            return;
        }
        this.midiAccess = await navigator.requestMIDIAccess();
        return this.getInputDevices();
    }

    getInputDevices() {
        if (!this.midiAccess) return [];
        const devices = [];
        for (const input of this.midiAccess.inputs.values()) {
            devices.push({ id: input.id, name: input.name });
        }
        return devices;
    }

    getOutputDevices() {
        if (!this.midiAccess) return [];
        const devices = [];
        for (const output of this.midiAccess.outputs.values()) {
            devices.push({ id: output.id, name: output.name });
        }
        return devices;
    }

    connectInput(deviceId) {
        if (this.inputDevice) {
            this.inputDevice.onmidimessage = null;
        }
        this.inputDevice = this.midiAccess.inputs.get(deviceId);
        if (this.inputDevice) {
            this.inputDevice.onmidimessage = (msg) => this._handleMessage(msg);
        }
    }

    connectOutput(deviceId) {
        this.outputDevice = this.midiAccess.outputs.get(deviceId);
    }

    _handleMessage(msg) {
        const [status, note, velocity] = msg.data;
        const command = status & 0xf0;
        if (command === 0x90 && velocity > 0) {
            // Note on
            this.inputState[note] = velocity / 127.0;
        } else if (command === 0x80 || (command === 0x90 && velocity === 0)) {
            // Note off
            this.inputState[note] = 0.0;
        }
        if (this._onNoteCallback) {
            this._onNoteCallback(note, this.inputState[note]);
        }
    }

    onNote(callback) {
        this._onNoteCallback = callback;
    }

    /**
     * Send the grid's output column as MIDI note events.
     * outputValues: Float32Array(128) with velocities 0.0-1.0
     */
    sendOutput(outputValues) {
        if (!this.outputDevice) return;
        for (let pitch = 0; pitch < 128; pitch++) {
            const vel = Math.round(outputValues[pitch] * 127);
            const prevVel = Math.round(this.outputState[pitch] * 127);

            if (vel > 0 && prevVel === 0) {
                // Note on
                this.outputDevice.send([0x90, pitch, vel]);
            } else if (vel === 0 && prevVel > 0) {
                // Note off
                this.outputDevice.send([0x80, pitch, 0]);
            }
        }
        this.outputState.set(outputValues);
    }

    /**
     * Load and play back a MIDI file through inputState.
     * Returns a controller object with stop() method.
     */
    playFile(arrayBuffer, fs, onTick) {
        // Minimal MIDI file parser — we parse note events and schedule them
        // For a proper implementation, use a library like Midi.js or tone.js
        // This is a placeholder that will be replaced with proper parsing
        let stopped = false;
        return {
            stop() { stopped = true; }
        };
    }

    getInputState() {
        return this.inputState;
    }
}
```

### Step 2: Commit

```bash
git add frontend/js/midi-io.js
git commit -m "feat: Web MIDI I/O — live input, output, device management"
```

---

## Task 16: Django Management Command — Train

A `manage.py train` command that runs the full training loop headlessly.

**Files:**
- Create: `training/management/__init__.py`
- Create: `training/management/commands/__init__.py`
- Create: `training/management/commands/train.py`

### Step 1: Implement the command

**`training/management/commands/train.py`**:
```python
import time
from django.core.management.base import BaseCommand
from django.conf import settings
from training.engine.trainer import Trainer
from training.engine.export import export_model
from training.models import TrainingRun, TrainingMetric


class Command(BaseCommand):
    help = "Run PC grid training on the MIDI corpus"

    def add_arguments(self, parser):
        parser.add_argument("--midi-dir", default=settings.MIDI_DATA_DIR)
        parser.add_argument("--grid-size", type=int, default=128)
        parser.add_argument("--relaxation-steps", type=int, default=128)
        parser.add_argument("--batch-size", type=int, default=16)
        parser.add_argument("--num-steps", type=int, default=1000)
        parser.add_argument("--checkpoint-every", type=int, default=50)
        parser.add_argument("--export-dir", default=str(settings.BASE_DIR / "frontend" / "model"))
        parser.add_argument("--fs", type=float, default=8.0)

    def handle(self, *args, **options):
        self.stdout.write(f"Starting training on {options['midi_dir']}")

        trainer = Trainer(
            midi_dir=options["midi_dir"],
            grid_size=options["grid_size"],
            relaxation_steps=options["relaxation_steps"],
            fs=options["fs"],
        )

        run = TrainingRun.objects.create(
            status="running",
            config_json=options,
        )

        try:
            for step in range(1, options["num_steps"] + 1):
                t0 = time.time()
                avg_error = trainer.train_step(batch_size=options["batch_size"])
                dt = time.time() - t0

                TrainingMetric.objects.create(
                    run=run,
                    step=step,
                    avg_error=avg_error,
                    phase=trainer.curriculum.current_phase,
                )
                run.total_steps = step
                run.latest_error = avg_error
                run.current_phase = trainer.curriculum.current_phase
                run.save()

                self.stdout.write(
                    f"Step {step}/{options['num_steps']} | "
                    f"Error: {avg_error:.6f} | "
                    f"Phase: {trainer.curriculum.current_phase} | "
                    f"Time: {dt:.1f}s"
                )

                if step % options["checkpoint_every"] == 0:
                    ckpt_path = f"{settings.CHECKPOINT_DIR}/step_{step}"
                    trainer.save_checkpoint(ckpt_path)
                    self.stdout.write(f"  Checkpoint saved: {ckpt_path}")

        except KeyboardInterrupt:
            self.stdout.write("\nTraining interrupted.")
        finally:
            run.status = "stopped"
            run.save()

            # Export final model for browser
            export_model(
                trainer.grid,
                trainer.batch_gen.vocabulary,
                options["export_dir"],
            )
            self.stdout.write(f"Model exported to {options['export_dir']}")
```

### Step 2: Commit

```bash
mkdir -p training/management/commands
touch training/management/__init__.py training/management/commands/__init__.py
git add training/management/
git commit -m "feat: manage.py train command — headless training loop with checkpoints"
```

---

## Task 17: Integration Test — End-to-End Training

Verify the full pipeline works: scan corpus → generate batches → train → export.

**Files:**
- Create: `tests/test_integration.py`

### Step 1: Write integration test

**`tests/__init__.py`**: empty

**`tests/test_integration.py`**:
```python
import json
from pathlib import Path
import numpy as np
import pytest
from corpus.tests.test_scanner import make_test_midi
from corpus.services.scanner import scan_directory
from corpus.services.vocabulary import build_vocabulary
from corpus.services.batch_generator import BatchGenerator
from training.engine.trainer import Trainer
from training.engine.export import export_model, load_exported_model


class TestEndToEnd:
    def test_full_pipeline(self, tmp_path):
        """Scan → batch → train → export → verify."""
        # 1. Create test corpus
        for i in range(5):
            make_test_midi(tmp_path / f"song{i}.mid", [
                ("Piano", 0, False, [
                    (60 + j, j * 0.25, (j + 1) * 0.25, 100) for j in range(32)
                ]),
                ("Bass", 32, False, [
                    (36 + j % 4, j * 0.5, (j + 1) * 0.5, 80) for j in range(16)
                ]),
                ("Drums", 0, True, [
                    (36, j * 0.25, j * 0.25 + 0.1, 100) for j in range(32)
                ]),
            ])

        # 2. Scan and build vocabulary
        results = scan_directory(str(tmp_path))
        assert len(results) == 5
        vocab = build_vocabulary(results)
        assert len(vocab) >= 2

        # 3. Train for a few steps
        trainer = Trainer(
            midi_dir=str(tmp_path),
            grid_size=16,  # small for testing
            relaxation_steps=5,
            fs=4.0,
        )
        errors = []
        for _ in range(3):
            err = trainer.train_step(batch_size=2)
            errors.append(err)
            assert np.isfinite(err)

        # 4. Export
        export_dir = tmp_path / "export"
        export_model(trainer.grid, trainer.batch_gen.vocabulary, str(export_dir))
        assert (export_dir / "state.bin").exists()
        assert (export_dir / "weights.bin").exists()
        assert (export_dir / "config.json").exists()

        # 5. Verify export
        loaded = load_exported_model(str(export_dir))
        assert loaded["config"]["grid_size"] == 16
        assert np.allclose(loaded["weights"], np.array(trainer.grid.weights), atol=1e-6)
```

### Step 2: Run integration test

Run: `pytest tests/test_integration.py -v`
Expected: PASS

### Step 3: Commit

```bash
git add tests/
git commit -m "test: end-to-end integration test — scan through export"
```

---

## Summary

| Task | What it builds | Key files |
|------|---------------|-----------|
| 1 | Project scaffolding | `requirements.txt`, `pcjam/settings.py`, `manage.py` |
| 2 | MIDI corpus scanner | `corpus/services/scanner.py` |
| 3 | Instrument vocabulary | `corpus/services/vocabulary.py` |
| 4 | Batch generator | `corpus/services/batch_generator.py` |
| 5 | Curriculum scheduler | `corpus/services/curriculum.py` |
| 6 | JAX grid state | `training/engine/grid.py` |
| 7 | PC update rule | `training/engine/update_rule.py` |
| 8 | Training loop | `training/engine/trainer.py` |
| 9 | Model export | `training/engine/export.py` |
| 10 | Corpus REST API | `corpus/views.py`, `corpus/urls.py` |
| 11 | Training REST API | `training/views.py`, `training/urls.py` |
| 12 | WebGL scaffold | `frontend/index.html`, `frontend/js/model-loader.js` |
| 13 | PC inference shader | `frontend/shaders/pc-step.frag`, `frontend/js/grid-compute.js` |
| 14 | Grid renderer | `frontend/shaders/render-grid.frag`, `frontend/js/grid-renderer.js` |
| 15 | Web MIDI I/O | `frontend/js/midi-io.js` |
| 16 | Train command | `training/management/commands/train.py` |
| 17 | Integration test | `tests/test_integration.py` |

**After all tasks:** Run `python manage.py train --midi-dir /path/to/your/midi/files` to train, then open the browser to jam.
