from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from training.models import TrainingRun, TrainingMetric
from training.services.runner import TrainingRunner


class TrainingConfigView(APIView):
    """GET/POST training configuration."""

    DEFAULT_CONFIG = {
        "grid_size": 128,
        "relaxation_steps": 128,
        "batch_size": 16,
        "lr": 0.01,
        "lr_weights": 0.001,
        "fs": 8.0,
        "curriculum_patience": 10,
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
            "grid_size": 128,
            "relaxation_steps": 128,
            "batch_size": 16,
            "num_steps": 1000,
            "checkpoint_every": 50,
            "fs": 8.0,
        }
        config.update(request.data)
        # Ensure numeric types
        for key in ("grid_size", "relaxation_steps", "batch_size",
                     "num_steps", "checkpoint_every"):
            if key in config:
                config[key] = int(config[key])
        if "fs" in config:
            config["fs"] = float(config["fs"])

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
