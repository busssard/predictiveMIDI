"""Train hierarchical PC network on MIDI corpus.

Usage:
    python manage.py train_hierarchical --num-steps 250 --batch-size 16
    python manage.py train_hierarchical --layer-sizes 128 64 32 64 128 --teacher-forcing 0.8
"""
import json
import time
from pathlib import Path
from django.core.management.base import BaseCommand
from django.conf import settings
from training.engine.hierarchical_trainer import HierarchicalTrainer
from training.models import TrainingRun, TrainingMetric


class Command(BaseCommand):
    help = "Train hierarchical PC network on the MIDI corpus"

    def add_arguments(self, parser):
        parser.add_argument("--midi-dir", default=settings.MIDI_DATA_DIR)
        parser.add_argument("--layer-sizes", type=int, nargs="+",
                            default=[128, 64, 32, 64, 128],
                            help="Layer sizes for hourglass (default: 128 64 32 64 128)")
        parser.add_argument("--relaxation-steps", type=int, default=32)
        parser.add_argument("--batch-size", type=int, default=16)
        parser.add_argument("--num-steps", type=int, default=1000)
        parser.add_argument("--checkpoint-every", type=int, default=50)
        parser.add_argument("--fs", type=float, default=8.0)
        parser.add_argument("--index-path", default=None,
                            help="Path to corpus_index.json (auto-detects if omitted)")
        parser.add_argument("--lr", type=float, default=0.01,
                            help="Representation learning rate")
        parser.add_argument("--lr-w", type=float, default=0.001,
                            help="Weight learning rate")
        parser.add_argument("--alpha", type=float, default=0.8,
                            help="Temporal state decay (0=fast, 1=slow)")
        parser.add_argument("--lambda-sparse", type=float, default=0.005,
                            help="L1 sparsity penalty")
        parser.add_argument("--teacher-forcing", type=float, default=1.0,
                            help="Fraction of ticks with output clamped (1.0=always)")
        parser.add_argument("--tf-min", type=float, default=0.0,
                            help="Minimum teacher forcing after annealing (default: 0.0)")
        parser.add_argument("--tf-anneal-steps", type=int, default=0,
                            help="Steps to anneal TF from initial to min (0=no annealing)")
        parser.add_argument("--output-supervision", type=float, default=0.0,
                            help="Soft output target strength (0=none, 0.5=moderate)")
        parser.add_argument("--sup-min", type=float, default=0.0,
                            help="Minimum supervision after annealing")
        parser.add_argument("--sup-anneal-steps", type=int, default=0,
                            help="Steps to anneal supervision (0=no annealing)")
        parser.add_argument("--curriculum-patience", type=int, default=10)
        parser.add_argument("--phase-1-ticks", type=int, default=16)
        parser.add_argument("--phase-1-threshold", type=float, default=0.15)
        parser.add_argument("--phase-2-ticks", type=int, default=32)
        parser.add_argument("--phase-2-threshold", type=float, default=0.10)
        parser.add_argument("--phase-3-ticks", type=int, default=64)
        parser.add_argument("--phase-3-threshold", type=float, default=0.0)
        parser.add_argument("--checkpoint-dir", default=None,
                            help="Checkpoint directory (default: checkpoints_hierarchical/)")
        parser.add_argument("--resume-from", default=None,
                            help="Load weights from this checkpoint before training")
        parser.add_argument("--weight-update", default="hebbian",
                            choices=["hebbian", "autodiff"],
                            help="Weight update method: hebbian (outer-product) or autodiff (jax.grad)")
        parser.add_argument("--metrics-dir", default=None,
                            help="Directory for JSONL metrics logging (default: same as checkpoint-dir)")

    def handle(self, *args, **options):
        index_path = options.get("index_path")
        if index_path is None:
            default_index = Path(settings.BASE_DIR) / "data" / "corpus_index.json"
            split_parts = sorted(default_index.parent.glob("corpus_index_*.json"))
            if default_index.exists():
                index_path = str(default_index)
            elif split_parts:
                index_path = str(split_parts[0])

        ckpt_dir = options.get("checkpoint_dir")
        if ckpt_dir is None:
            ckpt_dir = str(Path(settings.BASE_DIR) / "checkpoints_hierarchical")

        curriculum_phases = {
            1: {"snippet_ticks": options["phase_1_ticks"], "threshold": options["phase_1_threshold"]},
            2: {"snippet_ticks": options["phase_2_ticks"], "threshold": options["phase_2_threshold"]},
            3: {"snippet_ticks": options["phase_3_ticks"], "threshold": options["phase_3_threshold"]},
        }

        trainer_kwargs = dict(
            midi_dir=options["midi_dir"],
            layer_sizes=options["layer_sizes"],
            relaxation_steps=options["relaxation_steps"],
            fs=options["fs"],
            lr=options["lr"],
            lr_w=options["lr_w"],
            alpha=options["alpha"],
            lambda_sparse=options["lambda_sparse"],
            teacher_forcing_ratio=options["teacher_forcing"],
            tf_min=options["tf_min"],
            tf_anneal_steps=options["tf_anneal_steps"],
            output_supervision=options["output_supervision"],
            sup_min=options["sup_min"],
            sup_anneal_steps=options["sup_anneal_steps"],
            curriculum_phases=curriculum_phases,
            curriculum_patience=options["curriculum_patience"],
            prefetch=True,
            weight_update=options["weight_update"],
            metrics_dir=options.get("metrics_dir") or ckpt_dir,
        )

        if index_path:
            self.stdout.write(f"Loading corpus index from {index_path}")
            trainer = HierarchicalTrainer(index_path=index_path, **trainer_kwargs)
            self.stdout.write(f"Loaded {len(trainer.batch_gen.song_paths)} songs")
        else:
            self.stdout.write(f"Scanning {options['midi_dir']}...")
            trainer = HierarchicalTrainer(**trainer_kwargs)

        # Resume from checkpoint if specified
        if options.get("resume_from"):
            resume_path = options["resume_from"]
            self.stdout.write(f"Resuming from checkpoint: {resume_path}")
            trainer.load_checkpoint(resume_path)

        self.stdout.write(f"Architecture: {options['layer_sizes']}")
        self.stdout.write(f"Teacher forcing: {options['teacher_forcing']}")
        self.stdout.write(f"Relaxation steps: {options['relaxation_steps']}")
        self.stdout.write(f"Weight update: {options['weight_update']}")

        run = TrainingRun.objects.create(
            status="running",
            config_json={
                "architecture": "hierarchical",
                "layer_sizes": options["layer_sizes"],
                **{k: v for k, v in options.items()
                   if k not in ("verbosity", "settings", "pythonpath",
                                "traceback", "no_color", "force_color",
                                "skip_checks", "checkpoint_dir")},
            },
        )

        heartbeat_path = Path(ckpt_dir) / "heartbeat.json"
        heartbeat_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            for step in range(1, options["num_steps"] + 1):
                t0 = time.time()
                avg_error, meta = trainer.train_step(
                    batch_size=options["batch_size"])
                dt = time.time() - t0

                f1 = meta.get("f1", 0.0)
                prec = meta.get("precision", 0.0)
                rec = meta.get("recall", 0.0)
                active_err = meta.get("active_error", 0.0)

                TrainingMetric.objects.create(
                    run=run, step=step, avg_error=avg_error,
                    phase=trainer.curriculum.current_phase,
                )
                run.total_steps = step
                run.latest_error = avg_error
                run.current_phase = trainer.curriculum.current_phase
                run.save()

                tf_now = meta.get("teacher_forcing", options["teacher_forcing"])
                sup_now = trainer.output_supervision
                status_line = (
                    f"Step {step}/{options['num_steps']} | "
                    f"Error: {avg_error:.6f} | "
                    f"Active: {active_err:.6f} | "
                    f"F1: {f1:.3f} (P:{prec:.3f} R:{rec:.3f}) | "
                    f"Phase: {trainer.curriculum.current_phase} | "
                    f"TF: {tf_now:.2f} | "
                    f"Sup: {sup_now:.2f} | "
                    f"Time: {dt:.1f}s"
                )
                self.stdout.write(status_line)

                # Write heartbeat for external monitoring
                heartbeat_path.write_text(json.dumps({
                    "step": step,
                    "total_steps": options["num_steps"],
                    "error": round(avg_error, 6),
                    "f1": round(f1, 3),
                    "phase": trainer.curriculum.current_phase,
                    "time_per_step": round(dt, 1),
                    "timestamp": time.time(),
                }))

                if step % options["checkpoint_every"] == 0:
                    ckpt_path = f"{ckpt_dir}/step_{step}"
                    trainer.save_checkpoint(ckpt_path)
                    self.stdout.write(f"  Checkpoint: {ckpt_path}")

        except KeyboardInterrupt:
            self.stdout.write("\nTraining interrupted.")
        finally:
            run.status = "stopped"
            run.save()

            final_path = f"{ckpt_dir}/final"
            trainer.save_checkpoint(final_path)
            self.stdout.write(f"Final checkpoint: {final_path}")
