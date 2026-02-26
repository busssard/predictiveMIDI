from django.db import models


class CorpusScan(models.Model):
    scanned_at = models.DateTimeField(auto_now_add=True)
    midi_dir = models.CharField(max_length=512)
    num_songs = models.IntegerField(default=0)
    vocabulary_json = models.JSONField(default=dict)

    class Meta:
        ordering = ["-scanned_at"]


class Song(models.Model):
    scan = models.ForeignKey(CorpusScan, on_delete=models.CASCADE, related_name="songs")
    path = models.CharField(max_length=512)
    tempo = models.FloatField()
    duration = models.FloatField()
    instruments_json = models.JSONField(default=list)
