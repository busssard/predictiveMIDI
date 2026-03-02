import atexit
import math
import signal
import threading
import time
import logging
from collections import defaultdict
from pathlib import Path

from django.conf import settings

from training.engine.trainer import Trainer
from training.engine.export import export_model
from training.models import TrainingRun, TrainingMetric

logger = logging.getLogger(__name__)


def _cleanup_jax():
    """Release JAX GPU resources on process exit."""
    try:
        import jax
        jax.clear_caches()
        # Block until all pending GPU work completes
        for dev in jax.devices():
            dev.synchronize()
    except Exception:
        pass

atexit.register(_cleanup_jax)


def _shutdown_handler(signum, frame):
    """Stop training gracefully on SIGINT/SIGTERM, then re-raise."""
    runner = TrainingRunner._instance
    if runner and runner.is_running:
        logger.info("Signal %d received — stopping training...", signum)
        runner.stop(wait=True, timeout=5)
    # Re-raise so Django/Python handles shutdown normally
    signal.default_int_handler(signum, frame)


# Only register in main thread (Django autoreloader spawns threads)
if threading.current_thread() is threading.main_thread():
    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)


class TrainingRunner:
    """Singleton that runs training in a background daemon thread.

    Only one training run can be active at a time.
    The dashboard checks status after start/stop — no polling during training.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    obj = super().__new__(cls)
                    obj._thread = None
                    obj._stop_event = threading.Event()
                    obj._run_id = None
                    obj._error_msg = None
                    obj._stats = {}
                    cls._instance = obj
                    # Mark any orphaned "running" entries from previous server sessions
                    obj._cleanup_stale_runs()
        return cls._instance

    @staticmethod
    def _cleanup_stale_runs():
        try:
            stale = TrainingRun.objects.filter(status="running")
            count = stale.update(status="stopped")
            if count:
                logger.info("Marked %d stale training run(s) as stopped", count)
        except Exception:
            pass  # DB might not be ready yet

    @property
    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    def get_status(self):
        return {
            "running": self.is_running,
            "run_id": self._run_id,
            "error": self._error_msg,
            **self._stats,
        }

    def start(self, config):
        if self.is_running:
            raise RuntimeError("Training already running")

        self._stop_event.clear()
        self._error_msg = None
        self._thread = threading.Thread(
            target=self._train_loop,
            args=(config,),
            daemon=True,
        )
        self._thread.start()

    def stop(self, wait=False, timeout=10):
        self._stop_event.set()
        if wait and self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def _train_loop(self, config):
        run = None
        try:
            # Auto-detect corpus index (single file or split parts)
            index_path = None
            default_index = Path(settings.BASE_DIR) / "data" / "corpus_index.json"
            split_parts = sorted(default_index.parent.glob("corpus_index_*.json"))
            if default_index.exists():
                index_path = str(default_index)
            elif split_parts:
                index_path = str(split_parts[0])

            midi_dir = getattr(settings, "MIDI_DATA_DIR", str(Path(settings.BASE_DIR) / "midi_data"))

            activation = config.get("activation", "leaky_relu")
            lr = float(config.get("lr", 0.005))
            lr_w = float(config.get("lr_w", 0.001))
            alpha = float(config.get("alpha", 0.9))
            beta = float(config.get("beta", 0.1))
            connectivity = config.get("connectivity", "neighbor")
            state_momentum = float(config.get("state_momentum", 0.9))
            spike_boost = float(config.get("spike_boost", 5.0))
            asl_gamma_neg = float(config.get("asl_gamma_neg", 4.0))
            asl_margin = float(config.get("asl_margin", 0.05))
            lr_amplification = float(config.get("lr_amplification", 0.0))

            curriculum_config = config.get("curriculum", {})
            curriculum_phases = curriculum_config.get("phases")
            curriculum_patience = curriculum_config.get("patience", 10)

            # Convert phase keys from string (JSON) to int if needed
            if curriculum_phases:
                curriculum_phases = {
                    int(k): v for k, v in curriculum_phases.items()
                }

            pos_weight = float(config.get("pos_weight", 20.0))
            lambda_sparse = float(config.get("lambda_sparse", 0.01))

            trainer_kwargs = dict(
                midi_dir=midi_dir,
                grid_width=config.get("grid_width", config.get("grid_size", 16)),
                grid_height=config.get("grid_height", config.get("grid_size", 128)),
                relaxation_steps=config.get("relaxation_steps", 64),
                fs=config.get("fs", 8.0),
                prefetch=True,
                activation=activation,
                lr=lr,
                lr_w=lr_w,
                alpha=alpha,
                beta=beta,
                curriculum_phases=curriculum_phases,
                curriculum_patience=curriculum_patience,
                pos_weight=pos_weight,
                lambda_sparse=lambda_sparse,
                connectivity=connectivity,
                state_momentum=state_momentum,
                spike_boost=spike_boost,
                asl_gamma_neg=asl_gamma_neg,
                asl_margin=asl_margin,
                lr_amplification=lr_amplification,
            )

            if index_path:
                trainer = Trainer(index_path=index_path, **trainer_kwargs)
            else:
                from corpus.services.dataset_scanner import scan_datasets
                scan_results = scan_datasets(midi_dir)
                if not scan_results:
                    self._error_msg = "No MIDI files found"
                    return
                trainer = Trainer(scan_results=scan_results, **trainer_kwargs)

            run = TrainingRun.objects.create(
                status="running",
                config_json=config,
            )
            self._run_id = run.id

            num_steps = config.get("num_steps", 1000)
            batch_size = config.get("batch_size", 16)
            checkpoint_every = config.get("checkpoint_every", 50)
            fs = config.get("fs", 8.0)

            total_samples = 0
            total_ticks = 0
            dataset_counts = defaultdict(int)
            self._stats = {
                "total_samples": 0,
                "total_ticks": 0,
                "total_seconds": 0.0,
                "snippet_ticks": trainer.curriculum.snippet_ticks,
                "snippet_seconds": trainer.curriculum.snippet_ticks / fs,
                "dataset_pct": {},
            }

            consecutive_nan = 0
            prev_errors = []
            last_good_weights = trainer.grid.weights

            for step in range(1, num_steps + 1):
                if self._stop_event.is_set():
                    break

                snippet_ticks = trainer.curriculum.snippet_ticks
                step_t0 = time.time()
                avg_error, batch_meta = trainer.train_step(batch_size=batch_size)
                step_dt = time.time() - step_t0

                if math.isnan(avg_error) or math.isinf(avg_error):
                    consecutive_nan += 1
                    logger.warning(
                        "NaN at step %d (consecutive: %d) — skipping batch",
                        step, consecutive_nan,
                    )
                    # Rollback to last good weights
                    trainer.grid.weights = last_good_weights
                    if consecutive_nan >= 3:
                        self._error_msg = f"3 consecutive NaN batches at step {step}"
                        break
                    continue

                consecutive_nan = 0
                last_good_weights = trainer.grid.weights

                # Check for dramatic error increase (divergence)
                if len(prev_errors) >= 5:
                    recent_avg = sum(prev_errors[-5:]) / 5
                    if avg_error > recent_avg * 10:
                        self._error_msg = (
                            f"Error spiked at step {step}: "
                            f"{avg_error:.6f} vs avg {recent_avg:.6f}"
                        )
                        break
                prev_errors.append(avg_error)
                if len(prev_errors) > 10:
                    prev_errors.pop(0)

                total_samples += batch_size
                total_ticks += batch_size * snippet_ticks
                for ds in batch_meta.get("datasets", []):
                    dataset_counts[ds] += 1
                dataset_pct = {
                    ds: round(100 * n / total_samples, 1)
                    for ds, n in dataset_counts.items()
                }
                active_error = batch_meta.get("active_error", 0.0)
                col_energy = batch_meta.get("col_energy", {})
                self._stats.update({
                    "step": step,
                    "num_steps": num_steps,
                    "step_time": round(step_dt, 2),
                    "total_samples": total_samples,
                    "total_ticks": total_ticks,
                    "total_seconds": total_ticks / fs,
                    "snippet_ticks": snippet_ticks,
                    "snippet_seconds": snippet_ticks / fs,
                    "dataset_pct": dataset_pct,
                    "active_error": active_error,
                    "col_energy": col_energy,
                    "f1": batch_meta.get("f1", 0.0),
                    "precision": batch_meta.get("precision", 0.0),
                    "recall": batch_meta.get("recall", 0.0),
                })

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

                if checkpoint_every and step % checkpoint_every == 0:
                    ckpt_path = Path(settings.CHECKPOINT_DIR) / f"step_{step}"
                    trainer.save_checkpoint(str(ckpt_path))

        except Exception as e:
            logger.exception("Training failed")
            self._error_msg = str(e)
        finally:
            if run is not None:
                run.status = "stopped"
                run.save()
                try:
                    export_dir = str(Path(settings.BASE_DIR) / "frontend" / "model")
                    export_model(trainer.grid, trainer.batch_gen.vocabulary, export_dir)
                except Exception:
                    logger.exception("Model export failed")
