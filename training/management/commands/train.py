import time
from pathlib import Path
from django.core.management.base import BaseCommand
from django.conf import settings
from corpus.services.dataset_scanner import scan_datasets
from training.engine.trainer import Trainer
from training.engine.export import export_model
from training.models import TrainingRun, TrainingMetric


class Command(BaseCommand):
    help = "Run PC grid training on the MIDI corpus"

    def add_arguments(self, parser):
        parser.add_argument("--midi-dir", default=settings.MIDI_DATA_DIR)
        parser.add_argument("--grid-width", type=int, default=16)
        parser.add_argument("--grid-height", type=int, default=128)
        parser.add_argument("--relaxation-steps", type=int, default=64)
        parser.add_argument("--batch-size", type=int, default=16)
        parser.add_argument("--num-steps", type=int, default=1000)
        parser.add_argument("--checkpoint-every", type=int, default=50)
        parser.add_argument("--export-dir", default=str(settings.BASE_DIR / "frontend" / "model"))
        parser.add_argument("--fs", type=float, default=8.0)
        parser.add_argument("--datasets", nargs="*", default=None,
                            help="Which datasets to use (lakh, aam, slakh). Default: all detected.")
        parser.add_argument("--index-path", default=None,
                            help="Path to corpus_index.json. Auto-detects data/corpus_index.json.")
        parser.add_argument("--lr", type=float, default=0.005,
                            help="Representation learning rate")
        parser.add_argument("--lr-w", type=float, default=0.001,
                            help="Weight learning rate (independent of lr)")
        parser.add_argument("--alpha", type=float, default=0.9,
                            help="Leaky integrator decay (0=fast, 1=slow)")
        parser.add_argument("--beta", type=float, default=0.1,
                            help="Recurrent state gain")
        parser.add_argument("--curriculum-patience", type=int, default=10,
                            help="Consecutive low-error steps before phase advance")
        parser.add_argument("--phase-1-ticks", type=int, default=16)
        parser.add_argument("--phase-1-threshold", type=float, default=0.15)
        parser.add_argument("--phase-2-ticks", type=int, default=32)
        parser.add_argument("--phase-2-threshold", type=float, default=0.10)
        parser.add_argument("--phase-3-ticks", type=int, default=64)
        parser.add_argument("--phase-3-threshold", type=float, default=0.0)
        parser.add_argument("--pos-weight", type=float, default=20.0,
                            help="Upweight active (non-zero) notes at boundaries (default: 20)")
        parser.add_argument("--lambda-sparse", type=float, default=0.01,
                            help="L1 sparsity penalty on representations (default: 0.01)")
        parser.add_argument("--connectivity", choices=["neighbor", "fc", "fc_double"],
                            default="neighbor",
                            help="Inter-column connectivity mode")
        parser.add_argument("--state-momentum", type=float, default=0.9,
                            help="Temporal amortization momentum (0.0=disabled)")
        parser.add_argument("--spike-boost", type=float, default=5.0,
                            help="Column wavefront spiking precision boost")
        parser.add_argument("--asl-gamma-neg", type=float, default=4.0,
                            help="ASL negative-class focusing parameter")
        parser.add_argument("--asl-margin", type=float, default=0.05,
                            help="ASL probability shift margin")
        parser.add_argument("--activation", default="leaky_relu",
                            choices=["tanh", "sigmoid", "relu", "leaky_relu", "linear"],
                            help="Activation function (default: leaky_relu)")
        parser.add_argument("--lr-amplification", type=float, default=0.0,
                            help="Per-column lr scaling amplification (0.0=disabled)")
        parser.add_argument("--no-forward-init", action="store_true",
                            help="Disable forward initialization")
        parser.add_argument("--metrics-dir", default=None,
                            help="Directory for JSONL metrics log (default: same as checkpoint dir)")

    def handle(self, *args, **options):
        # Auto-detect corpus index (single file or split parts)
        index_path = options.get("index_path")
        if index_path is None:
            default_index = Path(settings.BASE_DIR) / "data" / "corpus_index.json"
            split_parts = sorted(default_index.parent.glob("corpus_index_*.json"))
            if default_index.exists():
                index_path = str(default_index)
            elif split_parts:
                index_path = str(split_parts[0])

        curriculum_phases = {
            1: {"snippet_ticks": options["phase_1_ticks"], "threshold": options["phase_1_threshold"]},
            2: {"snippet_ticks": options["phase_2_ticks"], "threshold": options["phase_2_threshold"]},
            3: {"snippet_ticks": options["phase_3_ticks"], "threshold": options["phase_3_threshold"]},
        }

        metrics_dir = options.get("metrics_dir") or str(settings.CHECKPOINT_DIR)

        trainer_kwargs = dict(
            midi_dir=options["midi_dir"],
            grid_width=options["grid_width"],
            grid_height=options["grid_height"],
            relaxation_steps=options["relaxation_steps"],
            fs=options["fs"],
            lr=options["lr"],
            lr_w=options["lr_w"],
            alpha=options["alpha"],
            beta=options["beta"],
            prefetch=True,
            curriculum_phases=curriculum_phases,
            curriculum_patience=options["curriculum_patience"],
            connectivity=options["connectivity"],
            state_momentum=options["state_momentum"],
            spike_boost=options["spike_boost"],
            asl_gamma_neg=options["asl_gamma_neg"],
            asl_margin=options["asl_margin"],
            activation=options["activation"],
            lr_amplification=options["lr_amplification"],
            forward_init=not options["no_forward_init"],
            metrics_dir=metrics_dir,
        )

        if index_path:
            self.stdout.write(f"Loading corpus index from {index_path}")
            trainer = Trainer(index_path=index_path, **trainer_kwargs)
            self.stdout.write(f"Loaded {len(trainer.batch_gen.song_paths)} songs from index")
        else:
            self.stdout.write(f"Scanning datasets in {options['midi_dir']}...")
            scan_results = scan_datasets(options["midi_dir"], datasets=options["datasets"])
            self.stdout.write(f"Found {len(scan_results)} songs")

            if not scan_results:
                self.stderr.write("No MIDI files found. Check --midi-dir path.")
                return

            trainer = Trainer(scan_results=scan_results, **trainer_kwargs)

        run = TrainingRun.objects.create(
            status="running",
            config_json={k: v for k, v in options.items() if k not in ("verbosity", "settings", "pythonpath", "traceback", "no_color", "force_color", "skip_checks")},
        )

        try:
            for step in range(1, options["num_steps"] + 1):
                t0 = time.time()
                avg_error, meta = trainer.train_step(batch_size=options["batch_size"], step=step)
                dt = time.time() - t0

                active_err = meta.get("active_error", 0.0) if isinstance(meta, dict) else 0.0
                col_energy = meta.get("col_energy", {}) if isinstance(meta, dict) else {}
                f1 = meta.get("f1", 0.0) if isinstance(meta, dict) else 0.0
                prec = meta.get("precision", 0.0) if isinstance(meta, dict) else 0.0
                rec = meta.get("recall", 0.0) if isinstance(meta, dict) else 0.0

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

                col_str = ""
                if col_energy and step % 10 == 0:
                    parts = [f"c{k}={v:.4f}" for k, v in sorted(col_energy.items())]
                    col_str = f" | ColE: {' '.join(parts)}"

                self.stdout.write(
                    f"Step {step}/{options['num_steps']} | "
                    f"Error: {avg_error:.6f} | "
                    f"Active: {active_err:.6f} | "
                    f"F1: {f1:.3f} (P:{prec:.3f} R:{rec:.3f}) | "
                    f"Phase: {trainer.curriculum.current_phase} | "
                    f"Time: {dt:.1f}s{col_str}"
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

            export_model(
                trainer.grid,
                trainer.batch_gen.vocabulary,
                options["export_dir"],
            )
            self.stdout.write(f"Model exported to {options['export_dir']}")
