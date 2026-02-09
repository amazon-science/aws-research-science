#!/bin/bash
# Show queue status

QUEUE_FILE="experiments/queue.json"

if [ ! -f "$QUEUE_FILE" ]; then
    echo "📋 Queue is empty"
    exit 0
fi

echo "📋 Queue Status"
echo ""

# Running jobs
RUNNING_COUNT=$(jq '.running | length' "$QUEUE_FILE")
if [ "$RUNNING_COUNT" -gt 0 ]; then
    echo "✓ Running ($RUNNING_COUNT):"
    jq -r '.running[] | "  \(.name) - GPU \(.gpu) (PID: \(.pid))"' "$QUEUE_FILE"
    echo ""
fi

# Queued jobs
QUEUED_COUNT=$(jq '.queued | length' "$QUEUE_FILE")
if [ "$QUEUED_COUNT" -gt 0 ]; then
    echo "⏳ Queued ($QUEUED_COUNT):"
    jq -r '.queued[] | "  \(.name)\(if .retry_count > 0 then " (retry \(.retry_count)/3)" else "" end)\(if .last_error then " - \(.last_error)" else "" end)"' "$QUEUE_FILE"
else
    echo "⏳ No jobs queued"
fi

# Check watcher status
PID_FILE="experiments/queue_watcher.pid"
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo ""
        echo "🔄 Queue watcher: Running (PID: $PID)"
    else
        echo ""
        echo "⚠️  Queue watcher: Not running (start with ./scripts/queue_start_watcher.sh)"
    fi
else
    echo ""
    echo "⚠️  Queue watcher: Not running (start with ./scripts/queue_start_watcher.sh)"
fi
