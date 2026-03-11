#!/bin/bash
RUN_DIR=${1:-$(ls -td runs/*/ 2>/dev/null | head -1)}
if [ -z "$RUN_DIR" ]; then echo "No runs found"; exit 1; fi

echo "Monitoring: $RUN_DIR"

# Check if process is running
if [ -f "$RUN_DIR/pid" ]; then
    PID=$(cat "$RUN_DIR/pid")
    if kill -0 "$PID" 2>/dev/null; then
        echo "Status: RUNNING (PID $PID)"
    else
        echo "Status: STOPPED (PID $PID no longer running)"
    fi
fi

echo ""
echo "--- Last 5 metrics ---"
tail -5 "$RUN_DIR/metrics.jsonl" 2>/dev/null || echo "No metrics yet"

echo ""
echo "--- Last 10 log lines ---"
tail -10 "$RUN_DIR/stdout.log" 2>/dev/null || echo "No logs yet"
