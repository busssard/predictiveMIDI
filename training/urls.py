from django.urls import path
from training.views import (
    TrainingConfigView, TrainingMetricsView,
    TrainingStartView, TrainingStopView, TrainingStatusView,
    TrainingRunsListView,
)

urlpatterns = [
    path("config/", TrainingConfigView.as_view(), name="training-config"),
    path("metrics/", TrainingMetricsView.as_view(), name="training-metrics"),
    path("start/", TrainingStartView.as_view(), name="training-start"),
    path("stop/", TrainingStopView.as_view(), name="training-stop"),
    path("status/", TrainingStatusView.as_view(), name="training-status"),
    path("runs/", TrainingRunsListView.as_view(), name="training-runs"),
]
