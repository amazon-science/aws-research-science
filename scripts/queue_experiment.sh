#!/bin/bash
# Queue experiment for execution when GPU available

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUEUE_FILE="experiments/queue.json"
LOCK_FILE="experiments/.queue.lock"

# Parse arguments — positional: name, command, [mem_mb]; named: --after <name> (repeatable)
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
    echo "Usage: $0 <exp_name> <command> [gpu_mem_needed_mb] [--after <dep_name>...]"
    echo "Example: $0 exp_finetune 'python finetune.py' 10000 --after exp_pretrain"
    echo "Example: $0 exp_eval 'python eval.py' 4000 --after exp_a --after exp_b"
    exit 1
fi

# Capture launch environment at queue time so the watcher can reproduce it
QUEUE_WORKDIR="$PWD"
QUEUE_PYTHONPATH="${PYTHONPATH:-}"
QUEUE_VIRTUAL_ENV="${VIRTUAL_ENV:-}"
QUEUE_CONDA_ENV="${CONDA_DEFAULT_ENV:-}"

# Initialize queue file if doesn't exist
mkdir -p experiments
if [ ! -f "$QUEUE_FILE" ]; then
    echo '{"queued": [], "running": [], "completed": []}' > "$QUEUE_FILE"
fi
# Migrate old queue files that lack completed array
jq 'if .completed == null then .completed = [] else . end' "$QUEUE_FILE" > "$QUEUE_FILE.tmp" && mv "$QUEUE_FILE.tmp" "$QUEUE_FILE"

# Create lock file if doesn't exist
touch "$LOCK_FILE"

# Use directory-based locking (portable, works without flock)
LOCK_ACQUIRED=0
for attempt in {1..30}; do
    if mkdir "$LOCK_FILE.dir" 2>/dev/null; then
        LOCK_ACQUIRED=1
        trap "rmdir '$LOCK_FILE.dir' 2>/dev/null" EXIT
        break
    fi
    sleep 0.5
done

if [ $LOCK_ACQUIRED -eq 0 ]; then
    echo "⚠️  Could not acquire lock after 15 seconds, adding to queue instead..."
    jq --arg name "$EXP_NAME" \
       --arg cmd "$COMMAND" \
       --arg mem "$GPU_MEM_NEEDED" \
       --arg queued "$(date -Iseconds)" \
       --arg workdir "$QUEUE_WORKDIR" \
       --arg pythonpath "$QUEUE_PYTHONPATH" \
       --arg virtual_env "$QUEUE_VIRTUAL_ENV" \
       --arg conda_env "$QUEUE_CONDA_ENV" \
       --argjson depends_on "$DEPENDS_ON_JSON" \
       '.queued += [{
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
       }]' "$QUEUE_FILE" > "$QUEUE_FILE.tmp" && mv "$QUEUE_FILE.tmp" "$QUEUE_FILE"
    echo "✅ Added $EXP_NAME to queue (couldn't acquire lock)"
    [ "$DEPENDS_ON_JSON" != "[]" ] && echo "🔗 Depends on: $(echo "$DEPENDS_ON_JSON" | jq -r 'join(", ")')"
    exit 0
fi

    # If this job has dependencies, always queue — watcher will launch when deps complete
    if [ "$DEPENDS_ON_JSON" != "[]" ]; then
        TIMESTAMP=$(date -Iseconds)
        jq --arg name "$EXP_NAME" \
           --arg cmd "$COMMAND" \
           --arg mem "$GPU_MEM_NEEDED" \
           --arg queued "$TIMESTAMP" \
           --arg workdir "$QUEUE_WORKDIR" \
           --arg pythonpath "$QUEUE_PYTHONPATH" \
           --arg virtual_env "$QUEUE_VIRTUAL_ENV" \
           --arg conda_env "$QUEUE_CONDA_ENV" \
           --argjson depends_on "$DEPENDS_ON_JSON" \
           '.queued += [{
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
           }]' "$QUEUE_FILE" > "$QUEUE_FILE.tmp" && mv "$QUEUE_FILE.tmp" "$QUEUE_FILE"
        echo "✅ Added $EXP_NAME to queue (waiting for deps)"
        echo "🔗 Depends on: $(echo "$DEPENDS_ON_JSON" | jq -r 'join(", ")')"
        rmdir "$LOCK_FILE.dir" 2>/dev/null
        trap - EXIT
        exit 0
    fi

    # Find best available GPU using round-robin + running job check
    IDLE_GPU=""
    if command -v nvidia-smi &> /dev/null; then
        # Get all GPUs that meet basic requirements (util < 30%, enough free memory)
        CANDIDATE_GPUS=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.free --format=csv,noheader,nounits 2>/dev/null | \
            awk -F',' -v mem="$GPU_MEM_NEEDED" '$2 < 30 && $3 > mem {print $1}')

        if [ -n "$CANDIDATE_GPUS" ]; then
            # Filter out GPUs that already have running jobs
            for gpu in $CANDIDATE_GPUS; do
                RUNNING_ON_GPU=$(jq -r --arg gpu "$gpu" '.running[] | select(.gpu == ($gpu | tonumber)) | .name' "$QUEUE_FILE" 2>/dev/null)

                if [ -z "$RUNNING_ON_GPU" ]; then
                    # This GPU has no running jobs - use it!
                    IDLE_GPU=$gpu
                    break
                fi
            done

            # If all GPUs have jobs, pick the one with fewest jobs
            if [ -z "$IDLE_GPU" ]; then
                IDLE_GPU=$(echo "$CANDIDATE_GPUS" | head -1)
            fi
        fi
    fi

    TIMESTAMP=$(date -Iseconds)

    if [ -n "$IDLE_GPU" ]; then
        # GPU selected via round-robin - launch immediately
        echo "🚀 GPU $IDLE_GPU selected, launching immediately..."

        # Start experiment tracking
        EXP_FILE=$("$SCRIPT_DIR/start_experiment.sh" "$EXP_NAME" "Auto-launched from queue" "$IDLE_GPU")

    # Launch in background — CUDA_VISIBLE_DEVICES inlined in command string to ensure
    # it's respected even by code that uses device_map={"": 0} or set_device() internally
    nohup bash -c "
        cd '$QUEUE_WORKDIR'
        export PYTHONPATH='$QUEUE_PYTHONPATH'
        export VIRTUAL_ENV='$QUEUE_VIRTUAL_ENV'
        export CONDA_DEFAULT_ENV='$QUEUE_CONDA_ENV'
        export EXP_FILE='$EXP_FILE'

        START_TIME=\$(date +%s)
        LAUNCH_CMD=\$(echo '$COMMAND' | sed 's/--device cuda:[0-9]*/--device cuda:0/g')
        CUDA_VISIBLE_DEVICES='$IDLE_GPU' \$LAUNCH_CMD
        EXIT_CODE=\$?
        END_TIME=\$(date +%s)
        RUNTIME=\$((END_TIME - START_TIME))

        if [ \$EXIT_CODE -eq 0 ]; then
            '$SCRIPT_DIR/complete_experiment.sh' completed '$EXP_FILE'
        else
            '$SCRIPT_DIR/complete_experiment.sh' failed '$EXP_FILE'
        fi

        # Remove from running queue
        jq '.running = [.running[] | select(.name != \"$EXP_NAME\")]' '$QUEUE_FILE' > '$QUEUE_FILE.tmp' && mv '$QUEUE_FILE.tmp' '$QUEUE_FILE'
    " > "experiments/${EXP_NAME}_output.log" 2>&1 &

    PID=$!

    # Add to running queue
    jq --arg name "$EXP_NAME" \
       --arg gpu "$IDLE_GPU" \
       --arg pid "$PID" \
       --arg started "$TIMESTAMP" \
       --arg exp_file "$EXP_FILE" \
       '.running += [{
           name: $name,
           gpu: ($gpu | tonumber),
           pid: ($pid | tonumber),
           started_at: $started,
           exp_file: $exp_file
       }]' "$QUEUE_FILE" > "$QUEUE_FILE.tmp" && mv "$QUEUE_FILE.tmp" "$QUEUE_FILE"

        echo "✅ Launched $EXP_NAME on GPU $IDLE_GPU (PID: $PID)"
        echo "📊 Experiment file: $EXP_FILE"
        echo "📝 Output log: experiments/${EXP_NAME}_output.log"

        # Hold lock for 45s to cover model loading window — prevents the next job
        # from seeing this GPU as free before its VRAM is actually claimed
        echo "⏸️  Holding lock for 45s (model loading window)..."
        sleep 45

        # Release lock
        rmdir "$LOCK_FILE.dir" 2>/dev/null
        trap - EXIT
else
    # No GPU available - add to queue
    echo "⏳ No GPU available, adding to queue..."

    jq --arg name "$EXP_NAME" \
       --arg cmd "$COMMAND" \
       --arg mem "$GPU_MEM_NEEDED" \
       --arg queued "$TIMESTAMP" \
       --arg workdir "$QUEUE_WORKDIR" \
       --arg pythonpath "$QUEUE_PYTHONPATH" \
       --arg virtual_env "$QUEUE_VIRTUAL_ENV" \
       --arg conda_env "$QUEUE_CONDA_ENV" \
       --argjson depends_on "$DEPENDS_ON_JSON" \
       '.queued += [{
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
       }]' "$QUEUE_FILE" > "$QUEUE_FILE.tmp" && mv "$QUEUE_FILE.tmp" "$QUEUE_FILE"

    echo "✅ Added $EXP_NAME to queue"
    echo "📊 Queue status:"
    jq -r '.queued | length | "   Queued: \(.)"' "$QUEUE_FILE"
    jq -r '.running | length | "   Running: \(.)"' "$QUEUE_FILE"

    # Release lock
    rmdir "$LOCK_FILE.dir" 2>/dev/null
    trap - EXIT
fi
