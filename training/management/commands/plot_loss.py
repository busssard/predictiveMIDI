import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from django.core.management.base import BaseCommand
from training.models import TrainingRun, TrainingMetric


class Command(BaseCommand):
    help = "Plot training loss curve for a run"

    def add_arguments(self, parser):
        parser.add_argument("--run-id", type=int, default=None,
                            help="TrainingRun ID. Default: latest run.")
        parser.add_argument("--output", default="training_loss.png")

    def handle(self, *args, **options):
        if options["run_id"]:
            run = TrainingRun.objects.get(pk=options["run_id"])
        else:
            run = TrainingRun.objects.order_by("-started_at").first()
            if run is None:
                self.stderr.write("No training runs found.")
                return

        metrics = run.metrics.order_by("step").values_list(
            "step", "avg_error", "phase"
        )
        if not metrics:
            self.stderr.write(f"No metrics for run {run.pk}.")
            return

        steps, errors, phases = zip(*metrics)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(steps, errors, linewidth=0.8, color="steelblue")
        ax.set_xlabel("Step")
        ax.set_ylabel("Avg Error")
        ax.set_title(f"Training Loss — Run {run.pk} ({run.started_at:%Y-%m-%d %H:%M})")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

        # Mark phase transitions
        prev_phase = phases[0]
        for step, phase in zip(steps, phases):
            if phase != prev_phase:
                ax.axvline(step, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
                ax.text(step, max(errors) * 0.9, f"Phase {phase}",
                        fontsize=8, color="red", ha="left")
                prev_phase = phase

        fig.tight_layout()
        fig.savefig(options["output"], dpi=150)
        self.stdout.write(f"Saved to {options['output']} ({len(steps)} steps)")
