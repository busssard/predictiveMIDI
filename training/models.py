from django.db import models


class TrainingRun(models.Model):
    started_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20, default="pending",
                              choices=[("pending", "Pending"),
                                       ("running", "Running"),
                                       ("stopped", "Stopped")])
    config_json = models.JSONField(default=dict)
    current_phase = models.IntegerField(default=1)
    total_steps = models.IntegerField(default=0)
    latest_error = models.FloatField(null=True)


class TrainingMetric(models.Model):
    run = models.ForeignKey(TrainingRun, on_delete=models.CASCADE,
                            related_name="metrics")
    step = models.IntegerField()
    avg_error = models.FloatField()
    phase = models.IntegerField()
    recorded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["step"]
