#!/bin/bash
# Queue experiment for execution when GPU available

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUEUE_FILE="experiments/queue.json"
LOCK_FILE="experiments/.queue.lock"

# Parse arguments
EXP_NAME="$1"
COMMAND="$2"
GPU_MEM_NEEDED="${3:-8000}"  # Default 8GB

if [ -z "$EXP_NAME" ] || [ -z "$COMMAND" ]; then
    echo "Usage: $0 <exp_name> <command> [gpu_mem_needed_mb]"
    echo "Example: $0 exp_rank16 'python train.py --rank 16' 10000"
    exit 1
fi

# Initialize queue file if doesn't exist
mkdir -p experiments
if [ ! -f "$QUEUE_FILE" ]; then
    echo '{"queued": [], "running": []}' > "$QUEUE_FILE"
fi

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
       '.queued += [{
           name: $name,
           command: $cmd,
           gpu_mem_needed: ($mem | tonumber),
           queued_at: $queued,
           status: "waiting",
           retry_count: 0,
           notes: []
       }]' "$QUEUE_FILE" > "$QUEUE_FILE.tmp" && mv "$QUEUE_FILE.tmp" "$QUEUE_FILE"
    echo "✅ Added $EXP_NAME to queue (couldn't acquire lock)"
    exit 0
fi

    # Check if GPU is available NOW (check both utilization AND memory usage)
    IDLE_GPU=""
    if command -v nvidia-smi &> /dev/null; then
        IDLE_GPU=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.free,memory.used --format=csv,noheader,nounits 2>/dev/null | \
            awk -F',' -v mem="$GPU_MEM_NEEDED" '$2 < 10 && $3 > mem && $4 < 1000 {print $1; exit}')
    fi

    TIMESTAMP=$(date -Iseconds)

    if [ -n "$IDLE_GPU" ]; then
        # Check if this GPU already has a running job
        RUNNING_ON_GPU=$(jq -r --arg gpu "$IDLE_GPU" '.running[] | select(.gpu == ($gpu | tonumber)) | .name' "$QUEUE_FILE" 2>/dev/null | head -1)

        if [ -n "$RUNNING_ON_GPU" ]; then
            echo "⏳ GPU $IDLE_GPU already has running job: $RUNNING_ON_GPU (startup race condition avoided)"
            echo "⏳ Adding to queue instead..."

            jq --arg name "$EXP_NAME" \
               --arg cmd "$COMMAND" \
               --arg mem "$GPU_MEM_NEEDED" \
               --arg queued "$TIMESTAMP" \
               '.queued += [{
                   name: $name,
                   command: $cmd,
                   gpu_mem_needed: ($mem | tonumber),
                   queued_at: $queued,
                   status: "waiting",
                   retry_count: 0,
                   notes: ["GPU was loading another job"]
               }]' "$QUEUE_FILE" > "$QUEUE_FILE.tmp" && mv "$QUEUE_FILE.tmp" "$QUEUE_FILE"

            echo "✅ Added $EXP_NAME to queue"
            exit 0
        fi

        # GPU truly available - launch immediately
        echo "🚀 GPU $IDLE_GPU available, launching immediately..."

        # Start experiment tracking
        EXP_FILE=$("$SCRIPT_DIR/start_experiment.sh" "$EXP_NAME" "Auto-launched from queue" "$IDLE_GPU")

    # Launch in background
    nohup bash -c "
        cd '$PWD'
        export CUDA_VISIBLE_DEVICES='$IDLE_GPU'
        export EXP_FILE='$EXP_FILE'

        START_TIME=\$(date +%s)
        $COMMAND
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

        # Hold lock for 20 seconds to ensure GPU shows memory usage
        echo "⏸️  Holding lock for 20s to prevent race conditions..."
        sleep 20

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
       '.queued += [{
           name: $name,
           command: $cmd,
           gpu_mem_needed: ($mem | tonumber),
           queued_at: $queued,
           status: "waiting",
           retry_count: 0,
           notes: []
       }]' "$QUEUE_FILE" > "$QUEUE_FILE.tmp" && mv "$QUEUE_FILE.tmp" "$QUEUE_FILE"

    echo "✅ Added $EXP_NAME to queue"
    echo "📊 Queue status:"
    jq -r '.queued | length | "   Queued: \(.)"' "$QUEUE_FILE"
    jq -r '.running | length | "   Running: \(.)"' "$QUEUE_FILE"

    # Release lock
    rmdir "$LOCK_FILE.dir" 2>/dev/null
    trap - EXIT
fi
