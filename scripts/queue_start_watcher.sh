#!/bin/bash
# Start the queue watcher daemon

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="experiments/queue_watcher.pid"

# Check if already running
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "⚠️  Queue watcher already running (PID: $PID)"
        exit 0
    fi
fi

# Start watcher in background
nohup "$SCRIPT_DIR/queue_watcher.sh" > experiments/queue_watcher.log 2>&1 &
PID=$!

# Save PID
mkdir -p experiments
echo "$PID" > "$PID_FILE"

echo "✅ Queue watcher started (PID: $PID)"
echo "📝 Logs: experiments/queue_watcher.log"
