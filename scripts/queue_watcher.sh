#!/bin/bash
# Background daemon that watches queue and launches jobs when GPUs free

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUEUE_FILE="experiments/queue.json"
LOCK_FILE="experiments/.queue.lock"
WATCH_INTERVAL=30  # Check every 30 seconds

echo "🔄 Queue watcher started (checking every ${WATCH_INTERVAL}s)"

# Ensure lock file exists
mkdir -p experiments
touch "$LOCK_FILE"

while true; do
    # Check if queue file exists
    if [ ! -f "$QUEUE_FILE" ]; then
        sleep "$WATCH_INTERVAL"
        continue
    fi

    # Get queued jobs
    QUEUED_COUNT=$(jq '.queued | length' "$QUEUE_FILE" 2>/dev/null || echo "0")

    if [ "$QUEUED_COUNT" -eq 0 ]; then
        sleep "$WATCH_INTERVAL"
        continue
    fi

    # Check for available GPUs with file locking
    if command -v nvidia-smi &> /dev/null; then
        # Acquire lock before checking/launching
        (
            flock -x -w 5 200 || {
                echo "⚠️  [$(date +'%H:%M:%S')] Could not acquire lock, skipping this cycle"
                exit 1
            }

            # Get next queued job
            NEXT_JOB=$(jq -r '.queued[0] | @json' "$QUEUE_FILE")

            if [ "$NEXT_JOB" = "null" ] || [ -z "$NEXT_JOB" ]; then
                exit 0
            fi

            EXP_NAME=$(echo "$NEXT_JOB" | jq -r '.name')
            COMMAND=$(echo "$NEXT_JOB" | jq -r '.command')
            GPU_MEM_NEEDED=$(echo "$NEXT_JOB" | jq -r '.gpu_mem_needed')
            RETRY_COUNT=$(echo "$NEXT_JOB" | jq -r '.retry_count // 0')

            # Find idle GPU with enough memory
            IDLE_GPU=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.free --format=csv,noheader,nounits 2>/dev/null | \
                awk -F',' -v mem="$GPU_MEM_NEEDED" '$2 < 10 && $3 > mem {print $1; exit}')

            if [ -z "$IDLE_GPU" ]; then
                exit 0
            fi

            # Check if this GPU already has a running job
            RUNNING_ON_GPU=$(jq -r --arg gpu "$IDLE_GPU" '.running[] | select(.gpu == ($gpu | tonumber)) | .name' "$QUEUE_FILE" 2>/dev/null | head -1)

            if [ -n "$RUNNING_ON_GPU" ]; then
                echo "⏳ [$(date +'%H:%M:%S')] GPU $IDLE_GPU already has running job: $RUNNING_ON_GPU (startup race avoided)"
                exit 0
            fi

            echo "🚀 [$(date +'%H:%M:%S')] Launching $EXP_NAME on GPU $IDLE_GPU..."

            # Remove from queue
            jq '.queued = .queued[1:]' "$QUEUE_FILE" > "$QUEUE_FILE.tmp" && mv "$QUEUE_FILE.tmp" "$QUEUE_FILE"

            # Start experiment tracking
            EXP_FILE=$("$SCRIPT_DIR/start_experiment.sh" "$EXP_NAME" "Auto-launched from queue (attempt $((RETRY_COUNT + 1)))" "$IDLE_GPU")

            TIMESTAMP_START=$(date -Iseconds)

            # Add to running queue immediately
            jq --arg name "$EXP_NAME" \
               --arg gpu "$IDLE_GPU" \
               --arg pid "$$" \
               --arg started "$TIMESTAMP_START" \
               --arg exp_file "$EXP_FILE" \
               '.running += [{
                   name: $name,
                   gpu: ($gpu | tonumber),
                   pid: ($pid | tonumber),
                   started_at: $started,
                   exp_file: $exp_file
               }]' "$QUEUE_FILE" > "$QUEUE_FILE.tmp" && mv "$QUEUE_FILE.tmp" "$QUEUE_FILE"

            # Hold lock for 10 seconds to ensure GPU shows activity before next check
            echo "⏸️  [$(date +'%H:%M:%S')] Holding lock for 10s to prevent race conditions..."
            sleep 10

            # Launch job and monitor (background, outside the lock)
            (
                cd "$PWD"
                export CUDA_VISIBLE_DEVICES="$IDLE_GPU"
                export EXP_FILE="$EXP_FILE"

                START_TIME=$(date +%s)

                # Run the command
                eval "$COMMAND" > "experiments/${EXP_NAME}_output.log" 2>&1
                EXIT_CODE=$?

                END_TIME=$(date +%s)
                RUNTIME=$((END_TIME - START_TIME))

                # Remove from running queue (need lock for queue file access)
                (
                    flock -x 201
                    jq --arg name "$EXP_NAME" \
                       '.running = [.running[] | select(.name != $name)]' "$QUEUE_FILE" > "$QUEUE_FILE.tmp" && mv "$QUEUE_FILE.tmp" "$QUEUE_FILE"
                ) 201>"$LOCK_FILE"

                if [ $EXIT_CODE -eq 0 ]; then
                    # Success
                    "$SCRIPT_DIR/complete_experiment.sh" completed "$EXP_FILE"
                    echo "✅ [$(date +'%H:%M:%S')] $EXP_NAME completed successfully"
                elif [ $RUNTIME -lt 60 ]; then
                    # Fast failure - likely resource issue, retry
                    if [ $RETRY_COUNT -lt 3 ]; then
                        NEW_RETRY_COUNT=$((RETRY_COUNT + 1))
                        TIMESTAMP_NOW=$(date -Iseconds)
                        ERROR_MSG="Launch failed (exit $EXIT_CODE), retrying ($NEW_RETRY_COUNT/3)"

                        echo "⚠️  [$(date +'%H:%M:%S')] $EXP_NAME failed fast, re-queuing (attempt $NEW_RETRY_COUNT/3)"

                        # Re-queue with updated retry count (need lock)
                        (
                            flock -x 201
                            jq --arg name "$EXP_NAME" \
                               --arg cmd "$COMMAND" \
                               --arg mem "$GPU_MEM_NEEDED" \
                               --argjson retry "$NEW_RETRY_COUNT" \
                               --arg note "$ERROR_MSG" \
                               --arg time "$TIMESTAMP_NOW" \
                               '.queued += [{
                                   name: $name,
                                   command: $cmd,
                                   gpu_mem_needed: ($mem | tonumber),
                                   queued_at: $time,
                                   status: "waiting",
                                   retry_count: $retry,
                                   notes: [$note],
                                   last_error: "Exit code \($EXIT_CODE | tostring), runtime \($RUNTIME | tostring)s"
                               }]' "$QUEUE_FILE" > "$QUEUE_FILE.tmp" && mv "$QUEUE_FILE.tmp" "$QUEUE_FILE"
                        ) 201>"$LOCK_FILE"

                        "$SCRIPT_DIR/complete_experiment.sh" failed "$EXP_FILE"
                        "$SCRIPT_DIR/report_note.sh" "$ERROR_MSG" "$EXP_FILE"
                    else
                        # Max retries exceeded
                        echo "❌ [$(date +'%H:%M:%S')] $EXP_NAME failed 3 times, giving up"
                        "$SCRIPT_DIR/complete_experiment.sh" failed "$EXP_FILE"
                        "$SCRIPT_DIR/report_note.sh" "Failed 3 times, giving up (exit $EXIT_CODE)" "$EXP_FILE"
                    fi
                else
                    # Mid-run failure - likely code bug, don't retry
                    echo "❌ [$(date +'%H:%M:%S')] $EXP_NAME failed after ${RUNTIME}s (likely code issue)"
                    "$SCRIPT_DIR/complete_experiment.sh" failed "$EXP_FILE"
                    "$SCRIPT_DIR/report_note.sh" "Failed after ${RUNTIME}s (exit $EXIT_CODE)" "$EXP_FILE"
                fi
            ) &

        ) 200>"$LOCK_FILE"
    fi

    sleep "$WATCH_INTERVAL"
done
