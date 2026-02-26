from django.urls import path
from training.views import TrainingConfigView, TrainingMetricsView

urlpatterns = [
    path("config/", TrainingConfigView.as_view(), name="training-config"),
    path("metrics/", TrainingMetricsView.as_view(), name="training-metrics"),
]
