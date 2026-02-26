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
