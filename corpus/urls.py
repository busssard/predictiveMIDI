from django.urls import path
from corpus.views import CorpusScanView, CorpusVocabularyView, CorpusStatsView, MidiToRollView

urlpatterns = [
    path("scan/", CorpusScanView.as_view(), name="corpus-scan"),
    path("vocabulary/", CorpusVocabularyView.as_view(), name="corpus-vocabulary"),
    path("stats/", CorpusStatsView.as_view(), name="corpus-stats"),
    path("midi-to-roll/", MidiToRollView.as_view(), name="midi-to-roll"),
]
