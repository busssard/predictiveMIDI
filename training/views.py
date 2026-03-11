import json
import logging
from pathlib import Path

from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from training.models import TrainingRun, TrainingMetric, NetworkLayout
from training.services.runner import TrainingRunner

logger = logging.getLogger(__name__)


class TrainingConfigView(APIView):
    """GET/POST training configuration."""

    DEFAULT_CONFIG = {
        "grid_width": 16,
        "grid_height": 128,
        "relaxation_steps": 64,
        "batch_size": 16,
        "lr": 0.005,
        "lr_w": 0.001,
        "fs": 8.0,
        "curriculum_patience": 10,
        "connectivity": "neighbor",
        "spike_boost": 5.0,
        "asl_gamma_neg": 4.0,
        "asl_margin": 0.05,
        "activation": "leaky_relu",
        "state_momentum": 0.9,
        "lr_amplification": 0.0,
    }

    def get(self, request):
        run = TrainingRun.objects.filter(status="running").first()
        if run:
            return Response({"config": run.config_json, "run_id": run.id})
        return Response({"config": self.DEFAULT_CONFIG, "run_id": None})

    def post(self, request):
        config = {**self.DEFAULT_CONFIG, **request.data}
        run = TrainingRun.objects.filter(status="running").first()
        if run:
            run.config_json = config
            run.save()
            return Response({"config": config, "run_id": run.id})
        return Response({"config": config, "run_id": None,
                         "message": "Config saved. Start a training run to apply."})


class TrainingMetricsView(APIView):
    """GET training metrics for the current or specified run."""

    def get(self, request):
        run_id = request.query_params.get("run_id")
        if run_id:
            run = TrainingRun.objects.filter(id=run_id).first()
        else:
            run = TrainingRun.objects.order_by("-started_at").first()

        if not run:
            return Response({"error": "No training run found."},
                            status=status.HTTP_404_NOT_FOUND)

        after_step = request.query_params.get("after_step")
        metrics = run.metrics.all()
        if after_step:
            metrics = metrics.filter(step__gt=int(after_step))
        metrics = metrics.values("step", "avg_error", "phase")
        return Response({
            "run_id": run.id,
            "status": run.status,
            "current_phase": run.current_phase,
            "total_steps": run.total_steps,
            "latest_error": run.latest_error,
            "metrics": list(metrics),
        })


class TrainingStartView(APIView):
    """POST to start training in a background thread."""

    def post(self, request):
        runner = TrainingRunner()
        if runner.is_running:
            return Response(
                {"error": "Training already running"},
                status=status.HTTP_409_CONFLICT,
            )
        config = {
            "grid_width": 16,
            "grid_height": 128,
            "relaxation_steps": 64,
            "batch_size": 16,
            "num_steps": 1000,
            "checkpoint_every": 50,
            "fs": 8.0,
            "connectivity": "neighbor",
        }
        config.update(request.data)
        # Ensure numeric types
        for key in ("grid_width", "grid_height", "relaxation_steps", "batch_size",
                     "num_steps", "checkpoint_every"):
            if key in config:
                config[key] = int(config[key])
        for key in ("fs", "lr", "lr_w", "alpha", "beta", "pos_weight",
                     "lambda_sparse", "spike_boost", "asl_gamma_neg",
                     "asl_margin", "state_momentum", "lr_amplification"):
            if key in config:
                config[key] = float(config[key])

        runner.start(config)
        return Response({"status": "started", "config": config})


class TrainingStopView(APIView):
    """POST to gracefully stop training."""

    def post(self, request):
        runner = TrainingRunner()
        if not runner.is_running:
            return Response(
                {"error": "No training running"},
                status=status.HTTP_409_CONFLICT,
            )
        runner.stop()
        return Response({"status": "stopping"})


class TrainingStatusView(APIView):
    """GET current training status — lightweight, no DB query."""

    def get(self, request):
        runner = TrainingRunner()
        return Response(runner.get_status())


class TrainingRunsListView(APIView):
    """GET list of recent training runs."""

    def get(self, request):
        runs = TrainingRun.objects.order_by("-started_at")[:20]
        return Response({
            "runs": [
                {
                    "id": r.id,
                    "status": r.status,
                    "started_at": r.started_at.isoformat(),
                    "total_steps": r.total_steps,
                    "latest_error": r.latest_error,
                    "current_phase": r.current_phase,
                }
                for r in runs
            ]
        })


class CheckpointListView(APIView):
    """GET list of auto-saved checkpoints (flat and hierarchical)."""

    def get(self, request):
        checkpoints = []

        # 1) Scan flat checkpoints from CHECKPOINT_DIR
        ckpt_dir = Path(settings.CHECKPOINT_DIR)
        if ckpt_dir.exists():
            for d in sorted(ckpt_dir.iterdir()):
                if d.is_dir() and d.name.startswith("step_"):
                    state_file = d / "state.npy"
                    if state_file.exists():
                        step_str = d.name.replace("step_", "")
                        try:
                            step = int(step_str)
                        except ValueError:
                            continue
                        checkpoints.append({
                            "name": d.name,
                            "step": step,
                            "timestamp": state_file.stat().st_mtime,
                            "architecture": "flat",
                        })

        # 2) Scan hierarchical checkpoint directories
        base_dir = Path(settings.BASE_DIR)
        for hier_dir in sorted(base_dir.glob("checkpoints_hierarchical*")):
            if not hier_dir.is_dir():
                continue
            for sub in sorted(hier_dir.iterdir()):
                if not sub.is_dir():
                    continue
                meta_file = sub / "metadata.json"
                if not meta_file.exists():
                    continue
                try:
                    metadata = json.loads(meta_file.read_text())
                except (json.JSONDecodeError, OSError):
                    continue
                if metadata.get("architecture") != "hierarchical":
                    continue
                # Parse step number from name, or use -1 for 'final'
                if sub.name.startswith("step_"):
                    step_str = sub.name.replace("step_", "")
                    try:
                        step = int(step_str)
                    except ValueError:
                        step = -1
                elif sub.name == "final":
                    step = -1
                else:
                    continue
                # Use relative path from BASE_DIR as the name
                rel_name = f"{hier_dir.name}/{sub.name}"
                checkpoints.append({
                    "name": rel_name,
                    "step": step,
                    "timestamp": meta_file.stat().st_mtime,
                    "architecture": "hierarchical",
                    "layer_sizes": metadata.get("layer_sizes", []),
                })

        return Response({"checkpoints": checkpoints})


class ExportCheckpointView(APIView):
    """POST to export a checkpoint as a WebGL model."""

    @staticmethod
    def _resolve_checkpoint_path(checkpoint):
        """Resolve checkpoint name to absolute path.

        Flat checkpoints live under CHECKPOINT_DIR (e.g. "step_10").
        Hierarchical checkpoints use a relative path from BASE_DIR
        (e.g. "checkpoints_hierarchical/step_50").
        """
        # Try as flat checkpoint first
        flat_path = Path(settings.CHECKPOINT_DIR) / checkpoint
        if flat_path.exists():
            return flat_path
        # Try as relative path from BASE_DIR (hierarchical)
        hier_path = Path(settings.BASE_DIR) / checkpoint
        if hier_path.exists():
            return hier_path
        return None

    @staticmethod
    def _detect_architecture(ckpt_path):
        """Detect if a checkpoint is flat or hierarchical."""
        meta_file = ckpt_path / "metadata.json"
        if meta_file.exists():
            try:
                metadata = json.loads(meta_file.read_text())
                if metadata.get("architecture") == "hierarchical":
                    return "hierarchical"
            except (json.JSONDecodeError, OSError):
                pass
        if (ckpt_path / "state.npy").exists():
            return "flat"
        return "unknown"

    def post(self, request):
        checkpoint = request.data.get("checkpoint")
        name = request.data.get("name")

        if not checkpoint or not name:
            return Response(
                {"error": "checkpoint and name are required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        ckpt_path = self._resolve_checkpoint_path(checkpoint)
        if ckpt_path is None:
            return Response(
                {"error": f"Checkpoint {checkpoint} not found"},
                status=status.HTTP_404_NOT_FOUND,
            )

        architecture = self._detect_architecture(ckpt_path)
        export_dir = Path(settings.BASE_DIR) / "frontend" / "model" / name

        try:
            if architecture == "hierarchical":
                from training.engine.export import export_hierarchical_model
                export_hierarchical_model(str(ckpt_path), str(export_dir))
            else:
                self._export_flat(ckpt_path, export_dir)

            return Response({"status": "exported", "name": name})
        except Exception as e:
            logger.exception("Export failed")
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    @staticmethod
    def _export_flat(ckpt_path, export_dir):
        """Export a flat grid checkpoint."""
        import numpy as np
        from training.engine.export import export_model
        from training.engine.grid import GridState
        import jax.numpy as jnp

        state = jnp.array(np.load(ckpt_path / "state.npy"))
        weights = jnp.array(np.load(ckpt_path / "weights.npy"))
        params = jnp.array(np.load(ckpt_path / "params.npy"))

        height, width = int(state.shape[0]), int(state.shape[1])
        dummy_mask = jnp.zeros((height, width), dtype=bool)

        lp_path = ckpt_path / "log_precision.npy"
        if lp_path.exists():
            log_precision = jnp.array(np.load(lp_path))
        else:
            log_precision = jnp.zeros((height, width))

        grid = GridState(
            state=state, weights=weights, params=params,
            log_precision=log_precision,
            input_mask=dummy_mask, output_mask=dummy_mask,
            conditioning_mask=dummy_mask,
        )

        # Load vocabulary from checkpoint metadata first, fall back to latest
        vocab = {}
        metadata_file = ckpt_path / "metadata.json"
        if metadata_file.exists():
            metadata = json.loads(metadata_file.read_text())
            vocab = metadata.get("vocabulary", {})
        if not vocab:
            default_model = Path(settings.BASE_DIR) / "frontend" / "model" / "config.json"
            if default_model.exists():
                vocab = json.loads(default_model.read_text()).get("vocabulary", {})

        export_model(grid, vocab, str(export_dir))


class ModelListView(APIView):
    """GET list of exported models available for the jam page.

    Auto-exports the latest checkpoint as the default model if the
    checkpoint is newer than the current export (or no export exists).
    """

    def get(self, request):
        model_dir = Path(settings.BASE_DIR) / "frontend" / "model"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Auto-export latest checkpoint if needed
        self._auto_export_latest(model_dir)

        models = []

        # Check for default model
        default_config = model_dir / "config.json"
        if default_config.exists():
            try:
                config = json.loads(default_config.read_text())
                models.append({"name": "latest", "config": config})
            except Exception:
                pass

        # Check subdirectories
        for d in sorted(model_dir.iterdir()):
            if d.is_dir() and (d / "config.json").exists():
                try:
                    config = json.loads((d / "config.json").read_text())
                    models.append({"name": d.name, "config": config})
                except Exception:
                    continue

        return Response({"models": models})

    @staticmethod
    def _auto_export_latest(model_dir):
        """Export the latest checkpoint as default model if it's newer."""
        ckpt_dir = Path(settings.CHECKPOINT_DIR)
        if not ckpt_dir.exists():
            return

        # Find the latest checkpoint by step number
        ckpts = sorted(ckpt_dir.glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
        if not ckpts:
            return
        latest_ckpt = ckpts[-1]

        # Check if export is up to date
        export_config = model_dir / "config.json"
        ckpt_state = latest_ckpt / "state.npy"
        if not ckpt_state.exists():
            return
        if export_config.exists() and export_config.stat().st_mtime >= ckpt_state.stat().st_mtime:
            return  # Export is already up to date

        try:
            import numpy as np
            import jax.numpy as jnp
            from training.engine.export import export_model
            from training.engine.grid import GridState

            state = jnp.array(np.load(latest_ckpt / "state.npy"))
            weights = jnp.array(np.load(latest_ckpt / "weights.npy"))
            params = jnp.array(np.load(latest_ckpt / "params.npy"))

            height, width = int(state.shape[0]), int(state.shape[1])
            dummy_mask = jnp.zeros((height, width), dtype=bool)

            lp = jnp.zeros((height, width))
            lp_path = latest_ckpt / "log_precision.npy"
            if lp_path.exists():
                lp = jnp.array(np.load(lp_path))

            grid = GridState(
                state=state, weights=weights, params=params,
                log_precision=lp,
                input_mask=dummy_mask, output_mask=dummy_mask,
                conditioning_mask=dummy_mask,
            )

            # Load optional fields
            fc_path = latest_ckpt / "fc_weights.npy"
            if fc_path.exists():
                grid.fc_weights = jnp.array(np.load(fc_path))
            wt_path = latest_ckpt / "w_temporal.npy"
            if wt_path.exists():
                grid.w_temporal = jnp.array(np.load(wt_path))

            metadata_file = latest_ckpt / "metadata.json"
            vocab = {}
            if metadata_file.exists():
                vocab = json.loads(metadata_file.read_text()).get("vocabulary", {})
                connectivity = json.loads(metadata_file.read_text()).get("connectivity", "neighbor")
                grid.connectivity = connectivity
            if not vocab and export_config.exists():
                vocab = json.loads(export_config.read_text()).get("vocabulary", {})

            export_model(grid, vocab, str(model_dir))
            logger.info("Auto-exported checkpoint %s as default model", latest_ckpt.name)
        except Exception:
            logger.exception("Auto-export of latest checkpoint failed")


class LayoutListView(APIView):
    """GET list / POST create network layouts."""

    def get(self, request):
        layouts = NetworkLayout.objects.order_by("-updated_at")
        return Response({
            "layouts": [
                {
                    "id": l.id,
                    "name": l.name,
                    "created_at": l.created_at.isoformat(),
                    "updated_at": l.updated_at.isoformat(),
                }
                for l in layouts
            ]
        })

    def post(self, request):
        name = request.data.get("name")
        layout_json = request.data.get("layout_json", {})
        if not name:
            return Response(
                {"error": "name is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        layout, created = NetworkLayout.objects.update_or_create(
            name=name,
            defaults={"layout_json": layout_json},
        )
        return Response({
            "id": layout.id,
            "name": layout.name,
            "layout_json": layout.layout_json,
            "created": created,
        }, status=status.HTTP_201_CREATED if created else status.HTTP_200_OK)


class LayoutDetailView(APIView):
    """GET / PUT / DELETE a single network layout."""

    def get(self, request, pk):
        try:
            layout = NetworkLayout.objects.get(pk=pk)
        except NetworkLayout.DoesNotExist:
            return Response({"error": "Not found"}, status=status.HTTP_404_NOT_FOUND)
        return Response({
            "id": layout.id,
            "name": layout.name,
            "layout_json": layout.layout_json,
            "created_at": layout.created_at.isoformat(),
            "updated_at": layout.updated_at.isoformat(),
        })

    def put(self, request, pk):
        try:
            layout = NetworkLayout.objects.get(pk=pk)
        except NetworkLayout.DoesNotExist:
            return Response({"error": "Not found"}, status=status.HTTP_404_NOT_FOUND)
        if "name" in request.data:
            layout.name = request.data["name"]
        if "layout_json" in request.data:
            layout.layout_json = request.data["layout_json"]
        layout.save()
        return Response({
            "id": layout.id,
            "name": layout.name,
            "layout_json": layout.layout_json,
        })

    def delete(self, request, pk):
        try:
            layout = NetworkLayout.objects.get(pk=pk)
        except NetworkLayout.DoesNotExist:
            return Response({"error": "Not found"}, status=status.HTTP_404_NOT_FOUND)
        layout.delete()
        return Response({"status": "deleted"})
