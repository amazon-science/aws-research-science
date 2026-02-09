#!/bin/bash
# Stop the queue watcher daemon

PID_FILE="experiments/queue_watcher.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "⚠️  Queue watcher not running (no PID file)"
    exit 0
fi

PID=$(cat "$PID_FILE")

if ! ps -p "$PID" > /dev/null 2>&1; then
    echo "⚠️  Queue watcher not running (stale PID)"
    rm "$PID_FILE"
    exit 0
fi

# Kill the watcher
kill "$PID"
rm "$PID_FILE"

echo "✅ Queue watcher stopped (PID: $PID)"
