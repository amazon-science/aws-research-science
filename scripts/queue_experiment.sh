#!/bin/bash
# Queue experiment for execution — drops an inbox file and returns instantly.
# No locks, no waiting. The watcher picks it up on its next cycle.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INBOX_DIR="experiments/queue_inbox"

# ── Parse arguments ────────────────────────────────────────────────────────
# Positional: name, command, [mem_mb]
# Named:      --after <dep> (repeatable)
EXP_NAME="$1"
COMMAND="$2"
GPU_MEM_NEEDED="8000"
DEPENDS_ON_JSON="[]"

i=3
while [ $i -le $# ]; do
    arg="${!i}"
    case "$arg" in
        --after)
            i=$((i+1))
            dep="${!i}"
            DEPENDS_ON_JSON=$(echo "$DEPENDS_ON_JSON" | jq --arg dep "$dep" '. += [$dep]')
            ;;
        *)
            if [[ "$arg" =~ ^[0-9]+$ ]]; then
                GPU_MEM_NEEDED="$arg"
            fi
            ;;
    esac
    i=$((i+1))
done

if [ -z "$EXP_NAME" ] || [ -z "$COMMAND" ]; then
    echo "Usage: $0 <exp_name> <command> [gpu_mem_needed_mb] [--after <dep>...]"
    echo "Example: $0 finetune 'python finetune.py' 8000 --after pretrain"
    exit 1
fi

# ── Capture launch environment at queue time ───────────────────────────────
QUEUE_WORKDIR="$PWD"
QUEUE_PYTHONPATH="${PYTHONPATH:-}"
QUEUE_VIRTUAL_ENV="${VIRTUAL_ENV:-}"
QUEUE_CONDA_ENV="${CONDA_DEFAULT_ENV:-}"

# ── Write inbox file ───────────────────────────────────────────────────────
mkdir -p "$INBOX_DIR"
mkdir -p experiments

# Use nanoseconds + name for a unique filename
TIMESTAMP=$(date +%s%N 2>/dev/null || date +%s)
JOB_FILE="$INBOX_DIR/job_${TIMESTAMP}_${EXP_NAME}.json"

jq -n \
    --arg name "$EXP_NAME" \
    --arg cmd "$COMMAND" \
    --arg mem "$GPU_MEM_NEEDED" \
    --arg queued "$(date -Iseconds)" \
    --arg workdir "$QUEUE_WORKDIR" \
    --arg pythonpath "$QUEUE_PYTHONPATH" \
    --arg virtual_env "$QUEUE_VIRTUAL_ENV" \
    --arg conda_env "$QUEUE_CONDA_ENV" \
    --argjson depends_on "$DEPENDS_ON_JSON" \
    '{
        name: $name,
        command: $cmd,
        gpu_mem_needed: ($mem | tonumber),
        queued_at: $queued,
        status: "waiting",
        retry_count: 0,
        notes: [],
        workdir: $workdir,
        pythonpath: $pythonpath,
        virtual_env: $virtual_env,
        conda_env: $conda_env,
        depends_on: $depends_on
    }' > "$JOB_FILE"

echo "✅ Queued: $EXP_NAME  (${GPU_MEM_NEEDED}MB)"
[ "$DEPENDS_ON_JSON" != "[]" ] && echo "   Depends on: $(echo "$DEPENDS_ON_JSON" | jq -r 'join(", ")')"
echo "   Watcher picks up within 10s"
