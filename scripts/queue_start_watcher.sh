#!/bin/bash
# Start the queue watcher daemon.
# Checks heartbeat recency — not just PID — to detect dead/zombie watchers.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="experiments/queue_watcher.pid"
HEARTBEAT_FILE="experiments/queue_watcher.heartbeat"
HEARTBEAT_MAX_AGE=30  # watcher writes every 10s; 30s = 3 missed cycles

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE" 2>/dev/null)
    if ps -p "$PID" > /dev/null 2>&1; then
        # PID alive — check heartbeat to confirm it's actually our watcher
        # (a recycled PID from a training process would pass the ps check)
        if [ -f "$HEARTBEAT_FILE" ]; then
            LAST=$(cat "$HEARTBEAT_FILE" 2>/dev/null || echo 0)
            AGE=$(( $(date +%s) - LAST ))
            if [ "$AGE" -le "$HEARTBEAT_MAX_AGE" ]; then
                echo "⚠️  Queue watcher already running (PID: $PID, heartbeat ${AGE}s ago)"
                exit 0
            fi
            echo "⚠️  PID $PID alive but heartbeat is ${AGE}s old — watcher stuck, restarting"
        else
            UPTIME=$(ps -p "$PID" -o etimes= 2>/dev/null | tr -d ' ' || echo 999)
            if [ "${UPTIME:-999}" -lt "$HEARTBEAT_MAX_AGE" ]; then
                echo "⚠️  Queue watcher just started (PID: $PID, ${UPTIME}s old)"
                exit 0
            fi
            echo "⚠️  PID $PID alive but no heartbeat after ${UPTIME}s — restarting"
        fi
        kill "$PID" 2>/dev/null || true
        sleep 1
    else
        echo "⚠️  Stale PID file (PID $PID not running) — starting fresh"
    fi
fi

mkdir -p experiments
nohup "$SCRIPT_DIR/queue_watcher.sh" > experiments/queue_watcher.log 2>&1 &
PID=$!
echo "$PID" > "$PID_FILE"

echo "✅ Queue watcher started (PID: $PID)"
echo "📝 Logs: experiments/queue_watcher.log"
