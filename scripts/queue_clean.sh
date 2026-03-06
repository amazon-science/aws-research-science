#!/bin/bash
# Remove stale dead-process entries from the running queue.
# Signals the watcher via done files — it owns queue.json.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(pwd)"
EXPERIMENTS_DIR="$PROJECT_DIR/experiments"
QUEUE_FILE="$EXPERIMENTS_DIR/queue.json"
DONE_DIR="$EXPERIMENTS_DIR/queue_done"

if [ ! -f "$QUEUE_FILE" ]; then
    echo "No queue file found."
    exit 0
fi

RUNNING_JOBS=$(jq -r '.running[] | @json' "$QUEUE_FILE" 2>/dev/null)

if [ -z "$RUNNING_JOBS" ]; then
    echo "✓ Running queue is empty — nothing to clean."
    exit 0
fi

DEAD_JOBS=()
ALIVE_JOBS=()
while IFS= read -r job; do
    [ -z "$job" ] || [ "$job" = "null" ] && continue
    PID=$(echo "$job" | jq -r '.pid')
    NAME=$(echo "$job" | jq -r '.name')
    GPU=$(echo "$job" | jq -r '.gpu // 0')
    if ! kill -0 "$PID" 2>/dev/null; then
        DEAD_JOBS+=("$NAME:$GPU")
        echo "💀 Dead: $NAME (PID $PID, GPU $GPU)"
    else
        ALIVE_JOBS+=("$NAME")
        echo "✓ Alive: $NAME (PID $PID, GPU $GPU)"
    fi
done <<< "$RUNNING_JOBS"

if [ ${#DEAD_JOBS[@]} -eq 0 ]; then
    echo ""
    echo "✓ All ${#ALIVE_JOBS[@]} running job(s) are alive — nothing to clean."
    exit 0
fi

mkdir -p "$DONE_DIR"
for ENTRY in "${DEAD_JOBS[@]}"; do
    NAME="${ENTRY%%:*}"
    GPU="${ENTRY##*:}"
    # Unique timestamp suffix avoids collision with watcher's own done files
    jq -n \
        --arg  name "$NAME" \
        --argjson gpu  "$GPU" \
        '{ exit_code: 1, name: $name, runtime: 999,
           gpu: $gpu, retry_count: 3 }' \
        > "$DONE_DIR/${NAME}_clean_$(date +%s).json"
done

echo ""
echo "✓ Signalled ${#DEAD_JOBS[@]} stale entry(s) for removal. Watcher will clean up within 10s."
