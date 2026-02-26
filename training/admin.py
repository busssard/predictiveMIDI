from django.contrib import admin
from training.models import TrainingRun, TrainingMetric

admin.site.register(TrainingRun)
admin.site.register(TrainingMetric)
