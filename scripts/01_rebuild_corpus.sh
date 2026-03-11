#!/bin/bash
# Step 1: Rebuild corpus index with quality filters
# Uses --refilter to apply filters to existing entries without re-scanning.
# Resumable: safe to Ctrl+C and re-run.
set -e
cd "$(dirname "$0")/.."

echo "=== Rebuilding corpus index with quality filters ==="
uv run python manage.py build_corpus_index \
    --midi-dir data/midi \
    --refilter \
    --max-instruments 12 \
    --min-notes-per-instrument 10 \
    --max-duration 600 \
    --min-tempo 30 \
    --max-tempo 300

echo ""
echo "Done. Index at data/corpus_index.json"
