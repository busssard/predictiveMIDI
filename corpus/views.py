import json
import logging
from pathlib import Path

import numpy as np
import pretty_midi
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from rest_framework import status
from corpus.services.scanner import scan_directory
from corpus.services.vocabulary import build_vocabulary, categorize_instrument
from corpus.models import CorpusScan, Song

logger = logging.getLogger(__name__)


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


class CorpusStatsView(APIView):
    """GET corpus statistics from the index file."""

    def get(self, request):
        from corpus.services.batch_generator import BatchGenerator
        index_path = Path(settings.BASE_DIR) / "data" / "corpus_index.json"
        split_parts = sorted(index_path.parent.glob("corpus_index_*.json"))
        if index_path.exists():
            with open(index_path) as f:
                index_data = json.load(f)
        elif split_parts:
            index_data = BatchGenerator._load_index(str(split_parts[0]))
        else:
            return Response(
                {"error": "No corpus index found. Run: python manage.py build_corpus_index"},
                status=status.HTTP_404_NOT_FOUND,
            )

        by_dataset = {}
        for song in index_data.get("songs", []):
            ds = song.get("dataset", "unknown")
            by_dataset[ds] = by_dataset.get(ds, 0) + 1

        return Response({
            "total_songs": len(index_data.get("songs", [])),
            "by_dataset": by_dataset,
            "vocabulary": index_data.get("vocabulary", {}),
            "num_instruments": len(index_data.get("vocabulary", {})),
        })


class MidiToRollView(APIView):
    """POST a MIDI file, get per-instrument piano rolls as JSON."""

    parser_classes = [MultiPartParser]

    def post(self, request):
        midi_file = request.FILES.get("file")
        if not midi_file:
            return Response({"error": "No file uploaded"}, status=status.HTTP_400_BAD_REQUEST)

        fs = float(request.data.get("fs", 8.0))

        try:
            midi = pretty_midi.PrettyMIDI(midi_file)
        except Exception as e:
            logger.warning("Failed to parse MIDI: %s", e)
            return Response({"error": f"Invalid MIDI file: {e}"}, status=status.HTTP_400_BAD_REQUEST)

        instruments = []
        max_ticks = 0
        for inst in midi.instruments:
            roll = inst.get_piano_roll(fs=fs) / 127.0  # (128, T)
            ticks = roll.shape[1]
            max_ticks = max(max_ticks, ticks)
            cat = categorize_instrument(inst.program, inst.is_drum)
            instruments.append({
                "name": inst.name or cat,
                "category": cat,
                "program": inst.program,
                "is_drum": inst.is_drum,
                "roll": roll[:128, :].T.tolist(),  # (T, 128)
            })

        return Response({
            "instruments": instruments,
            "total_ticks": max_ticks,
            "fs": fs,
        })
