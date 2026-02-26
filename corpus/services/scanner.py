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
