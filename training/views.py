from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from training.models import TrainingRun, TrainingMetric


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

        metrics = run.metrics.all().values("step", "avg_error", "phase")
        return Response({
            "run_id": run.id,
            "status": run.status,
            "current_phase": run.current_phase,
            "total_steps": run.total_steps,
            "latest_error": run.latest_error,
            "metrics": list(metrics),
        })
