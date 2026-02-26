from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from corpus.services.scanner import scan_directory
from corpus.services.vocabulary import build_vocabulary
from corpus.models import CorpusScan, Song


class CorpusScanView(APIView):
    """POST to scan MIDI directory. GET to retrieve latest scan results."""

    def post(self, request):
        midi_dir = request.data.get("midi_dir", settings.MIDI_DATA_DIR)
        results = scan_directory(midi_dir)
        vocabulary = build_vocabulary(results)

        scan = CorpusScan.objects.create(
            midi_dir=midi_dir,
            num_songs=len(results),
            vocabulary_json=vocabulary,
        )
        for r in results:
            Song.objects.create(
                scan=scan,
                path=r["path"],
                tempo=r["tempo"],
                duration=r["duration"],
                instruments_json=r["instruments"],
            )
        return Response({
            "scan_id": scan.id,
            "num_songs": len(results),
            "vocabulary": vocabulary,
        }, status=status.HTTP_201_CREATED)

    def get(self, request):
        scan = CorpusScan.objects.first()
        if not scan:
            return Response({"error": "No scan yet. POST to scan first."},
                            status=status.HTTP_404_NOT_FOUND)
        songs = scan.songs.all().values("path", "tempo", "duration", "instruments_json")
        return Response({
            "scan_id": scan.id,
            "scanned_at": scan.scanned_at,
            "num_songs": scan.num_songs,
            "vocabulary": scan.vocabulary_json,
            "songs": list(songs),
        })


class CorpusVocabularyView(APIView):
    """GET the current instrument vocabulary."""

    def get(self, request):
        scan = CorpusScan.objects.first()
        if not scan:
            return Response({"error": "No scan yet."}, status=status.HTTP_404_NOT_FOUND)
        return Response({"vocabulary": scan.vocabulary_json})
