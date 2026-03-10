"""Jam command: load a checkpoint, feed MIDI input, generate accompaniment output.

Usage:
    python manage.py jam --checkpoint checkpoints/step_250 --midi input.mid
    python manage.py jam --checkpoint checkpoints/step_250 --midi input.mid --target piano
    python manage.py jam --checkpoint checkpoints/step_250 --midi input.mid --output output.mid
"""
from pathlib import Path

import numpy as np
import pretty_midi
from django.core.management.base import BaseCommand

from corpus.services.vocabulary import categorize_instrument
from training.engine.inference import load_checkpoint, run_inference


class Command(BaseCommand):
    help = "Generate accompaniment from a trained PC grid given MIDI input"

    def add_arguments(self, parser):
        parser.add_argument("--checkpoint", required=True,
                            help="Path to checkpoint directory")
        parser.add_argument("--midi", required=True,
                            help="Path to input MIDI file")
        parser.add_argument("--output", default=None,
                            help="Path to output MIDI file (default: <input>_jam.mid)")
        parser.add_argument("--target", default="piano",
                            help="Target instrument category (default: piano)")
        parser.add_argument("--threshold", type=float, default=0.5,
                            help="Note activation threshold (default: 0.5)")
        parser.add_argument("--relaxation-steps", type=int, default=64,
                            help="Relaxation steps per tick (default: 64)")
        parser.add_argument("--fs", type=float, default=8.0,
                            help="Sampling rate in Hz (default: 8.0)")
        parser.add_argument("--velocity", type=int, default=80,
                            help="Output MIDI velocity (default: 80)")
        parser.add_argument("--no-plot", action="store_true",
                            help="Skip piano roll visualization")

    def handle(self, *args, **options):
        checkpoint_path = options["checkpoint"]
        midi_path = options["midi"]
        target_cat = options["target"]
        threshold = options["threshold"]
        relax_steps = options["relaxation_steps"]
        fs = options["fs"]
        velocity = options["velocity"]

        # Load checkpoint
        self.stdout.write(f"Loading checkpoint from {checkpoint_path}...")
        grid, metadata = load_checkpoint(checkpoint_path)
        vocabulary = metadata.get("vocabulary", {})
        self.stdout.write(f"  Grid: {metadata['grid_height']}x{metadata['grid_width']}, "
                          f"activation={metadata.get('activation')}, "
                          f"connectivity={metadata.get('connectivity')}")
        self.stdout.write(f"  Vocabulary: {list(vocabulary.keys())}")

        # Load input MIDI
        self.stdout.write(f"Loading input MIDI from {midi_path}...")
        midi = pretty_midi.PrettyMIDI(str(midi_path))

        # Mix all instruments into a single input piano roll
        input_rolls = []
        for inst in midi.instruments:
            cat = categorize_instrument(inst.program, inst.is_drum)
            roll = inst.get_piano_roll(fs=fs) / 127.0
            self.stdout.write(f"  Input: {inst.name or cat} ({cat}), "
                              f"{roll.shape[1]} ticks, "
                              f"{np.count_nonzero(roll.max(axis=0))} active ticks")
            input_rolls.append(roll)

        if not input_rolls:
            self.stderr.write("No instruments found in input MIDI!")
            return

        max_ticks = max(r.shape[1] for r in input_rolls)
        input_mix = np.zeros((128, max_ticks), dtype=np.float32)
        for roll in input_rolls:
            input_mix[:, :roll.shape[1]] = np.maximum(
                input_mix[:, :roll.shape[1]], roll[:128])

        # Transpose to (T, 128)
        input_seq = input_mix.T
        self.stdout.write(f"  Mixed input: {input_seq.shape[0]} ticks, "
                          f"{np.count_nonzero(input_seq.max(axis=1))}/{input_seq.shape[0]} active ticks")

        # Build conditioning vector
        if target_cat not in vocabulary:
            self.stderr.write(f"Target '{target_cat}' not in vocabulary. "
                              f"Available: {list(vocabulary.keys())}")
            return

        cond = np.zeros(len(vocabulary), dtype=np.float32)
        cond[vocabulary[target_cat]] = 1.0
        self.stdout.write(f"  Target instrument: {target_cat}")

        # Run inference
        self.stdout.write(f"Running inference ({input_seq.shape[0]} ticks, "
                          f"{relax_steps} relaxation steps)...")
        output_seq, all_states = run_inference(
            grid, metadata, input_seq, cond,
            relaxation_steps=relax_steps,
        )
        self.stdout.write(f"  Output shape: {output_seq.shape}")

        # Analyze output
        output_binary = (output_seq > threshold).astype(np.float32)
        active_ticks = np.count_nonzero(output_binary.max(axis=1))
        notes_per_tick = output_binary.sum(axis=1)
        total_notes = int(output_binary.sum())

        self.stdout.write(f"\n--- Output Analysis ---")
        self.stdout.write(f"  Active ticks: {active_ticks}/{output_seq.shape[0]} "
                          f"({100*active_ticks/max(output_seq.shape[0],1):.1f}%)")
        self.stdout.write(f"  Total note activations: {total_notes}")
        self.stdout.write(f"  Notes per tick: mean={notes_per_tick.mean():.1f}, "
                          f"max={notes_per_tick.max():.0f}")

        # Pitch distribution
        pitch_activity = output_binary.sum(axis=0)
        active_pitches = np.nonzero(pitch_activity)[0]
        if len(active_pitches) > 0:
            self.stdout.write(f"  Active pitch range: {active_pitches[0]}-{active_pitches[-1]} "
                              f"({len(active_pitches)} unique pitches)")
            top5 = np.argsort(pitch_activity)[-5:][::-1]
            self.stdout.write(f"  Top 5 pitches: {list(top5)} "
                              f"(counts: {[int(pitch_activity[p]) for p in top5]})")
        else:
            self.stdout.write("  No active pitches in output!")

        # Signal propagation analysis
        self.stdout.write(f"\n--- Signal Propagation ---")
        mean_act_per_col = np.abs(all_states[:, :, :, 0]).mean(axis=(0, 1))
        mean_err_per_col = np.abs(all_states[:, :, :, 1]).mean(axis=(0, 1))
        self.stdout.write(f"  Mean |activation| per column: "
                          f"{', '.join(f'c{i}={v:.4f}' for i, v in enumerate(mean_act_per_col))}")
        self.stdout.write(f"  Mean |error| per column: "
                          f"{', '.join(f'c{i}={v:.4f}' for i, v in enumerate(mean_err_per_col))}")

        prop_ratio = mean_act_per_col[-1] / max(mean_act_per_col[0], 1e-8)
        self.stdout.write(f"  Signal propagation ratio (col_last/col_0): {prop_ratio:.6f}")

        # Convert output to MIDI
        output_midi = _piano_roll_to_midi(output_binary, fs=fs, velocity=velocity,
                                          instrument_name=target_cat)

        output_path = options["output"]
        if output_path is None:
            output_path = str(Path(midi_path).with_suffix("")) + f"_jam_{target_cat}.mid"

        output_midi.write(str(output_path))
        self.stdout.write(f"\nOutput MIDI written to: {output_path}")

        # Optional visualization
        if not options["no_plot"]:
            plot_path = str(Path(output_path).with_suffix(".png"))
            _plot_comparison(input_seq, output_seq, output_binary,
                             threshold, plot_path, target_cat)
            self.stdout.write(f"Piano roll comparison: {plot_path}")


def _piano_roll_to_midi(binary_roll, fs=8.0, velocity=80, instrument_name="piano"):
    """Convert binary piano roll (T, 128) to a PrettyMIDI object."""
    midi = pretty_midi.PrettyMIDI()

    # Map category to GM program
    program_map = {
        "piano": 0, "chromatic_percussion": 8, "organ": 16,
        "guitar": 24, "bass": 32, "strings": 40, "ensemble": 48,
        "brass": 56, "reed": 64, "pipe": 73, "synth": 80,
        "synth_fx": 96, "ethnic": 104, "percussive": 112,
        "sound_fx": 120,
    }
    program = program_map.get(instrument_name, 0)
    is_drum = (instrument_name == "drums")

    inst = pretty_midi.Instrument(program=program, is_drum=is_drum,
                                  name=instrument_name)

    T, num_pitches = binary_roll.shape
    tick_duration = 1.0 / fs

    for pitch in range(num_pitches):
        note_on = None
        for t in range(T):
            if binary_roll[t, pitch] > 0 and note_on is None:
                note_on = t
            elif binary_roll[t, pitch] == 0 and note_on is not None:
                start = note_on * tick_duration
                end = t * tick_duration
                if end - start >= tick_duration * 0.5:  # min note length
                    inst.notes.append(pretty_midi.Note(
                        velocity=velocity, pitch=pitch,
                        start=start, end=end))
                note_on = None
        # Close any open note
        if note_on is not None:
            start = note_on * tick_duration
            end = T * tick_duration
            if end - start >= tick_duration * 0.5:
                inst.notes.append(pretty_midi.Note(
                    velocity=velocity, pitch=pitch,
                    start=start, end=end))

    midi.instruments.append(inst)
    return midi


def _plot_comparison(input_seq, output_seq, output_binary, threshold,
                     save_path, target_cat):
    """Plot input vs output piano rolls side by side."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

    # Input
    axes[0].imshow(input_seq.T, aspect="auto", origin="lower",
                   cmap="Blues", interpolation="nearest")
    axes[0].set_ylabel("Pitch")
    axes[0].set_title("Input (mixed instruments)")

    # Raw output (probabilities)
    im = axes[1].imshow(output_seq.T, aspect="auto", origin="lower",
                        cmap="Reds", interpolation="nearest", vmin=0, vmax=1)
    axes[1].set_ylabel("Pitch")
    axes[1].set_title(f"Output probabilities (target: {target_cat})")
    plt.colorbar(im, ax=axes[1], shrink=0.6)

    # Binary output
    axes[2].imshow(output_binary.T, aspect="auto", origin="lower",
                   cmap="Greens", interpolation="nearest")
    axes[2].set_ylabel("Pitch")
    axes[2].set_xlabel("Tick")
    axes[2].set_title(f"Binary output (threshold={threshold})")

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
