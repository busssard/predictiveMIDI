import json
from pathlib import Path
import numpy as np
import pretty_midi
from corpus.services.scanner import scan_directory
from corpus.services.vocabulary import build_vocabulary, categorize_instrument


class BatchGenerator:
    def __init__(self, midi_dir, snippet_ticks=16, fs=8.0, rng_seed=None,
                 scan_results=None, index_path=None):
        """
        Args:
            midi_dir: path to directory of MIDI files
            snippet_ticks: number of time steps per snippet
            fs: sampling rate in Hz (ticks per second)
            rng_seed: optional seed for reproducibility
            scan_results: optional pre-computed scan results (skips scan_directory)
            index_path: optional path to corpus_index.json (skips scanning)
        """
        self.midi_dir = Path(midi_dir)
        self.snippet_ticks = snippet_ticks
        self.fs = fs
        self.rng = np.random.default_rng(rng_seed)

        if scan_results is not None:
            songs = scan_results
            self.vocabulary = build_vocabulary(songs)
        elif index_path is not None:
            with open(index_path) as f:
                index_data = json.load(f)
            songs = index_data["songs"]
            self.vocabulary = index_data["vocabulary"]
        else:
            songs = scan_directory(str(self.midi_dir))
            self.vocabulary = build_vocabulary(songs)

        def _song_key(s):
            return s["path"] if "path" in s else s["source_paths"][0]

        self.song_paths = [_song_key(s) for s in songs]
        self._scan_results = songs
        self._scan_by_path = {_song_key(s): s for s in songs}

    def _load_piano_rolls(self, path):
        """Load a MIDI file and return per-instrument piano rolls.

        If the scan result has source_paths, loads from all listed files.

        Returns list of (category, piano_roll) tuples.
        piano_roll shape: (128, total_ticks), values 0.0-1.0
        """
        scan = self._scan_by_path.get(path)
        source_paths = scan.get("source_paths") if scan else None
        if not source_paths:
            source_paths = [path]

        rolls = []
        for src in source_paths:
            midi = pretty_midi.PrettyMIDI(str(src))
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
