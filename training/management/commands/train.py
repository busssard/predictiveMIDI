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
        parser.add_argument("--grid-size", type=int, default=128)
        parser.add_argument("--relaxation-steps", type=int, default=128)
        parser.add_argument("--batch-size", type=int, default=16)
        parser.add_argument("--num-steps", type=int, default=1000)
        parser.add_argument("--checkpoint-every", type=int, default=50)
        parser.add_argument("--export-dir", default=str(settings.BASE_DIR / "frontend" / "model"))
        parser.add_argument("--fs", type=float, default=8.0)
        parser.add_argument("--datasets", nargs="*", default=None,
                            help="Which datasets to use (lakh, aam, slakh). Default: all detected.")
        parser.add_argument("--index-path", default=None,
                            help="Path to corpus_index.json. Auto-detects data/corpus_index.json.")

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

        if index_path:
            self.stdout.write(f"Loading corpus index from {index_path}")
            trainer = Trainer(
                midi_dir=options["midi_dir"],
                grid_size=options["grid_size"],
                relaxation_steps=options["relaxation_steps"],
                fs=options["fs"],
                index_path=index_path,
                prefetch=True,
            )
            self.stdout.write(f"Loaded {len(trainer.batch_gen.song_paths)} songs from index")
        else:
            self.stdout.write(f"Scanning datasets in {options['midi_dir']}...")
            scan_results = scan_datasets(options["midi_dir"], datasets=options["datasets"])
            self.stdout.write(f"Found {len(scan_results)} songs")

            if not scan_results:
                self.stderr.write("No MIDI files found. Check --midi-dir path.")
                return

            trainer = Trainer(
                midi_dir=options["midi_dir"],
                grid_size=options["grid_size"],
                relaxation_steps=options["relaxation_steps"],
                fs=options["fs"],
                scan_results=scan_results,
                prefetch=True,
            )

        run = TrainingRun.objects.create(
            status="running",
            config_json={k: v for k, v in options.items() if k not in ("verbosity", "settings", "pythonpath", "traceback", "no_color", "force_color", "skip_checks")},
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

            export_model(
                trainer.grid,
                trainer.batch_gen.vocabulary,
                options["export_dir"],
            )
            self.stdout.write(f"Model exported to {options['export_dir']}")
