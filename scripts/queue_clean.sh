#!/bin/bash
# Remove stale dead-process entries from the running queue

QUEUE_FILE="experiments/queue.json"
LOCK_FILE="experiments/.queue.lock"

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
    GPU=$(echo "$job" | jq -r '.gpu')
    if ! kill -0 "$PID" 2>/dev/null; then
        DEAD_JOBS+=("$NAME")
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

# Signal dead jobs to the watcher via done files — it handles queue.json
mkdir -p experiments/queue_done
for NAME in "${DEAD_JOBS[@]}"; do
    echo "{\"exit_code\": 1, \"name\": \"$NAME\", \"runtime\": 999, \"gpu\": 0, \"retry_count\": 3}" \
        > "experiments/queue_done/${NAME}.json"
done

echo ""
echo "✓ Removed ${#DEAD_JOBS[@]} stale entry(s). ${#ALIVE_JOBS[@]} job(s) still running."
