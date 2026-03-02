import json
import logging
from collections import defaultdict
from pathlib import Path
import numpy as np
import pretty_midi
from corpus.services.scanner import scan_directory
from corpus.services.vocabulary import build_vocabulary, categorize_instrument

logger = logging.getLogger(__name__)


def _song_key(s):
    return s["path"] if "path" in s else s["source_paths"][0]


class BatchGenerator:
    def __init__(self, midi_dir, snippet_ticks=16, fs=8.0, rng_seed=None,
                 scan_results=None, index_path=None, test_fraction=0.1):
        """
        Args:
            midi_dir: path to directory of MIDI files
            snippet_ticks: number of time steps per snippet
            fs: sampling rate in Hz (ticks per second)
            rng_seed: optional seed for reproducibility
            scan_results: optional pre-computed scan results (skips scan_directory)
            index_path: optional path to corpus_index.json (skips scanning)
            test_fraction: fraction of each dataset to hold out for testing
        """
        self.midi_dir = Path(midi_dir)
        self.snippet_ticks = snippet_ticks
        self.fs = fs
        self.rng = np.random.default_rng(rng_seed)

        if scan_results is not None:
            songs = scan_results
            self.vocabulary = build_vocabulary(songs)
        elif index_path is not None:
            index_data = self._load_index(index_path)
            songs = index_data["songs"]
            self.vocabulary = index_data["vocabulary"]
        else:
            songs = scan_directory(str(self.midi_dir))
            self.vocabulary = build_vocabulary(songs)

        train_songs, test_songs = self._split_train_test(songs, test_fraction)

        self.song_paths = [_song_key(s) for s in train_songs]
        self._scan_results = train_songs
        self._scan_by_path = {_song_key(s): s for s in songs}  # keep all for loading
        self._path_to_dataset = {_song_key(s): s.get("dataset", "unknown") for s in songs}

        self.test_song_paths = [_song_key(s) for s in test_songs]
        self._test_scan_results = test_songs

    @staticmethod
    def _split_train_test(songs, test_fraction):
        """Split songs into train/test, taking test_fraction from each dataset."""
        if test_fraction <= 0:
            return songs, []

        by_dataset = defaultdict(list)
        for s in songs:
            by_dataset[s.get("dataset", "unknown")].append(s)

        # Deterministic split: use sorted order, take last N% as test
        train, test = [], []
        for ds in sorted(by_dataset):
            ds_songs = by_dataset[ds]
            n_test = max(1, int(len(ds_songs) * test_fraction))
            train.extend(ds_songs[:-n_test])
            test.extend(ds_songs[-n_test:])
        return train, test

    @staticmethod
    def _load_index(index_path):
        """Load corpus index, merging split files if present.

        Accepts a single JSON file or a path like 'data/corpus_index.json'
        where split parts (corpus_index_1.json, corpus_index_2.json, ...)
        exist alongside or instead of the single file.
        """
        p = Path(index_path)
        parts = sorted(p.parent.glob(p.stem.replace("_1", "").replace("_2", "") + "_*.json"))
        if parts:
            merged = None
            for part_path in parts:
                with open(part_path) as f:
                    data = json.load(f)
                if merged is None:
                    merged = data
                else:
                    merged["songs"].extend(data["songs"])
            return merged
        with open(index_path) as f:
            return json.load(f)

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
            try:
                midi = pretty_midi.PrettyMIDI(str(src))
            except Exception as e:
                logger.warning("Failed to load MIDI %s: %s", src, e)
                continue
            for inst in midi.instruments:
                cat = categorize_instrument(inst.program, inst.is_drum)
                roll = inst.get_piano_roll(fs=self.fs) / 127.0
                rolls.append((cat, roll))
        return rolls

    def _generate_one(self, snippet_ticks=None, song_paths=None, _attempt=0):
        """Generate one training example: input mix, target, conditioning."""
        if snippet_ticks is None:
            snippet_ticks = self.snippet_ticks
        if song_paths is None:
            song_paths = self.song_paths

        if _attempt >= 50:
            raise RuntimeError("Failed to generate a valid sample after 50 attempts")

        path = self.rng.choice(song_paths)
        dataset = self._path_to_dataset.get(path, "unknown")
        rolls = self._load_piano_rolls(path)

        if len(rolls) < 2:
            if _attempt < 50:
                logger.debug("Retrying: %s had < 2 instruments (attempt %d)", path, _attempt)
            return self._generate_one(snippet_ticks, song_paths, _attempt=_attempt + 1)

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
        if max_ticks <= snippet_ticks:
            start = 0
            snippet_len = min(max_ticks, snippet_ticks)
        else:
            start = self.rng.integers(0, max_ticks - snippet_ticks)
            snippet_len = snippet_ticks

        input_snippet = input_mix[:, start:start + snippet_len].T
        target_snippet = target_padded[:, start:start + snippet_len].T

        # Pad if shorter than snippet_ticks
        if input_snippet.shape[0] < snippet_ticks:
            pad = snippet_ticks - input_snippet.shape[0]
            input_snippet = np.pad(input_snippet, ((0, pad), (0, 0)))
            target_snippet = np.pad(target_snippet, ((0, pad), (0, 0)))

        # One-hot conditioning
        conditioning = np.zeros(len(self.vocabulary))
        if target_cat in self.vocabulary:
            conditioning[self.vocabulary[target_cat]] = 1.0

        return input_snippet, target_snippet, conditioning, target_cat, dataset

    def generate_batch(self, batch_size):
        """Generate a batch of training examples.

        Returns dict with:
            input: (batch_size, snippet_ticks, 128) float32
            target: (batch_size, snippet_ticks, 128) float32
            conditioning: (batch_size, num_categories) float32
            target_categories: list of category strings
        """
        # Snapshot snippet_ticks so all items in the batch use the same length,
        # even if the curriculum advances mid-batch in another thread.
        ticks = self.snippet_ticks

        inputs, targets, conds, cats, datasets = [], [], [], [], []
        for _ in range(batch_size):
            inp, tgt, cond, cat, ds = self._generate_one(ticks)
            inputs.append(inp)
            targets.append(tgt)
            conds.append(cond)
            cats.append(cat)
            datasets.append(ds)

        return {
            "input": np.array(inputs, dtype=np.float32),
            "target": np.array(targets, dtype=np.float32),
            "conditioning": np.array(conds, dtype=np.float32),
            "target_categories": cats,
            "datasets": datasets,
        }
