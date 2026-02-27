import threading
import time
import logging
from pathlib import Path

from django.conf import settings

from training.engine.trainer import Trainer
from training.engine.export import export_model
from training.models import TrainingRun, TrainingMetric

logger = logging.getLogger(__name__)


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
                    cls._instance = obj
        return cls._instance

    @property
    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    def get_status(self):
        return {
            "running": self.is_running,
            "run_id": self._run_id,
            "error": self._error_msg,
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

    def stop(self):
        self._stop_event.set()

    def _train_loop(self, config):
        run = None
        try:
            # Auto-detect corpus index
            index_path = None
            default_index = Path(settings.BASE_DIR) / "data" / "corpus_index.json"
            if default_index.exists():
                index_path = str(default_index)

            midi_dir = getattr(settings, "MIDI_DATA_DIR", str(Path(settings.BASE_DIR) / "midi_data"))

            activation = config.get("activation", "tanh")

            if index_path:
                trainer = Trainer(
                    midi_dir=midi_dir,
                    grid_size=config.get("grid_size", 128),
                    relaxation_steps=config.get("relaxation_steps", 128),
                    fs=config.get("fs", 8.0),
                    index_path=index_path,
                    prefetch=True,
                    activation=activation,
                )
            else:
                from corpus.services.dataset_scanner import scan_datasets
                scan_results = scan_datasets(midi_dir)
                if not scan_results:
                    self._error_msg = "No MIDI files found"
                    return
                trainer = Trainer(
                    midi_dir=midi_dir,
                    grid_size=config.get("grid_size", 128),
                    relaxation_steps=config.get("relaxation_steps", 128),
                    fs=config.get("fs", 8.0),
                    scan_results=scan_results,
                    prefetch=True,
                    activation=activation,
                )

            run = TrainingRun.objects.create(
                status="running",
                config_json=config,
            )
            self._run_id = run.id

            num_steps = config.get("num_steps", 1000)
            batch_size = config.get("batch_size", 16)
            checkpoint_every = config.get("checkpoint_every", 50)

            for step in range(1, num_steps + 1):
                if self._stop_event.is_set():
                    break

                avg_error = trainer.train_step(batch_size=batch_size)

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
