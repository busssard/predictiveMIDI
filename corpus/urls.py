from django.urls import path
from corpus.views import CorpusScanView, CorpusVocabularyView

urlpatterns = [
    path("scan/", CorpusScanView.as_view(), name="corpus-scan"),
    path("vocabulary/", CorpusVocabularyView.as_view(), name="corpus-vocabulary"),
]
