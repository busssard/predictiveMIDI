from django.urls import path
from training.views import (
    TrainingConfigView, TrainingMetricsView,
    TrainingStartView, TrainingStopView, TrainingStatusView,
    TrainingRunsListView, CheckpointListView, ExportCheckpointView,
    ModelListView, LayoutListView, LayoutDetailView,
)

urlpatterns = [
    path("config/", TrainingConfigView.as_view(), name="training-config"),
    path("metrics/", TrainingMetricsView.as_view(), name="training-metrics"),
    path("start/", TrainingStartView.as_view(), name="training-start"),
    path("stop/", TrainingStopView.as_view(), name="training-stop"),
    path("status/", TrainingStatusView.as_view(), name="training-status"),
    path("runs/", TrainingRunsListView.as_view(), name="training-runs"),
    path("checkpoints/", CheckpointListView.as_view(), name="training-checkpoints"),
    path("export-checkpoint/", ExportCheckpointView.as_view(), name="training-export-checkpoint"),
    path("models/", ModelListView.as_view(), name="training-models"),
    path("layouts/", LayoutListView.as_view(), name="layout-list"),
    path("layouts/<int:pk>/", LayoutDetailView.as_view(), name="layout-detail"),
]
