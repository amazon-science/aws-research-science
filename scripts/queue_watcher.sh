#!/bin/bash
# Background daemon that watches queue and launches jobs when GPUs free

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUEUE_FILE="experiments/queue.json"
LOCK_FILE="experiments/.queue.lock"
WATCH_INTERVAL=30  # Check every 30 seconds

echo "🔄 Queue watcher started (checking every ${WATCH_INTERVAL}s)"

# Ensure experiments directory exists
mkdir -p experiments

# Portable locking functions (works without flock)
acquire_lock() {
    local max_wait=${1:-10}
    local wait_time=0
    while [ $wait_time -lt $max_wait ]; do
        if mkdir "$LOCK_FILE.dir" 2>/dev/null; then
            return 0
        fi
        sleep 0.5
        wait_time=$((wait_time + 1))
    done
    return 1
}

release_lock() {
    rmdir "$LOCK_FILE.dir" 2>/dev/null
}

while true; do
    # Check if queue file exists
    if [ ! -f "$QUEUE_FILE" ]; then
        sleep "$WATCH_INTERVAL"
        continue
    fi

    # Health check: Remove dead processes from running queue
    if acquire_lock 5; then
        RUNNING_JOBS=$(jq -r '.running[] | @json' "$QUEUE_FILE" 2>/dev/null)

        if [ -n "$RUNNING_JOBS" ]; then
            DEAD_JOBS=()
            while IFS= read -r job; do
                if [ -z "$job" ] || [ "$job" = "null" ]; then
                    continue
                fi

                PID=$(echo "$job" | jq -r '.pid')
                NAME=$(echo "$job" | jq -r '.name')

                # Check if process is still alive
                if ! kill -0 "$PID" 2>/dev/null; then
                    echo "🧹 [$(date +'%H:%M:%S')] Cleaning up dead job: $NAME (PID $PID)"
                    DEAD_JOBS+=("$NAME")
                fi
            done <<< "$RUNNING_JOBS"

            # Remove all dead jobs from running queue
            for NAME in "${DEAD_JOBS[@]}"; do
                jq --arg name "$NAME" \
                   '.running = [.running[] | select(.name != $name)]' \
                   "$QUEUE_FILE" > "$QUEUE_FILE.tmp" && mv "$QUEUE_FILE.tmp" "$QUEUE_FILE"
            done
        fi

        release_lock
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
        if ! acquire_lock 5; then
            echo "⚠️  [$(date +'%H:%M:%S')] Could not acquire lock, skipping this cycle"
            sleep "$WATCH_INTERVAL"
            continue
        fi

        # Migrate old queue files lacking completed array
        jq 'if .completed == null then .completed = [] else . end' "$QUEUE_FILE" > "$QUEUE_FILE.tmp" && mv "$QUEUE_FILE.tmp" "$QUEUE_FILE"

        # Scan all queued jobs to find the first one that is eligible to run:
        # - all depends_on entries are in the completed list
        # - a suitable GPU is available (free, or inherited from dep)
        NEXT_JOB=""
        NEXT_JOB_IDX=""
        IDLE_GPU=""

        QUEUE_LEN=$(jq '.queued | length' "$QUEUE_FILE")
        idx=0
        while [ $idx -lt "$QUEUE_LEN" ]; do
            CANDIDATE=$(jq -r --argjson idx "$idx" '.queued[$idx] | @json' "$QUEUE_FILE")
            [ "$CANDIDATE" = "null" ] && { idx=$((idx+1)); continue; }

            CAND_NAME=$(echo "$CANDIDATE" | jq -r '.name')
            CAND_MEM=$(echo "$CANDIDATE" | jq -r '.gpu_mem_needed')
            CAND_DEPS=$(echo "$CANDIDATE" | jq -r '.depends_on // [] | @json')

            # Check if all dependencies are in the completed list
            DEPS_MET=$(jq --argjson deps "$CAND_DEPS" '
                if ($deps | length) == 0 then true
                else
                    ($deps | map(. as $d | any(.completed[]; .name == $d)) | all)
                end
            ' "$QUEUE_FILE")

            if [ "$DEPS_MET" != "true" ]; then
                idx=$((idx+1))
                continue
            fi

            # Determine GPU: inherit from last completed dep, or find any free GPU
            INHERITED_GPU=""
            if [ "$CAND_DEPS" != "[]" ]; then
                # Find GPU used by last-completed dependency
                INHERITED_GPU=$(jq -r --argjson deps "$CAND_DEPS" '
                    [.completed[] | select(.name as $n | $deps | index($n) != null)]
                    | sort_by(.completed_at) | last | .gpu | tostring
                ' "$QUEUE_FILE")

                # Warn if deps ran on different GPUs
                UNIQUE_DEP_GPUS=$(jq -r --argjson deps "$CAND_DEPS" '
                    [.completed[] | select(.name as $n | $deps | index($n) != null) | .gpu] | unique | length
                ' "$QUEUE_FILE")
                if [ "$UNIQUE_DEP_GPUS" -gt 1 ]; then
                    echo "⚠️  [$(date +'%H:%M:%S')] $CAND_NAME deps ran on different GPUs — inheriting GPU $INHERITED_GPU (last completed)"
                fi

                # Verify inherited GPU has enough free memory
                GPU_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
                    --id="$INHERITED_GPU" 2>/dev/null | tr -d ' ' || echo "0")
                if [ "$GPU_FREE" -lt "$CAND_MEM" ]; then
                    echo "⚠️  [$(date +'%H:%M:%S')] Inherited GPU $INHERITED_GPU only has ${GPU_FREE}MB free (need ${CAND_MEM}MB) — waiting"
                    idx=$((idx+1))
                    continue
                fi
                IDLE_GPU="$INHERITED_GPU"
            else
                # No deps — find any free GPU
                CANDIDATE_GPUS=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.free --format=csv,noheader,nounits 2>/dev/null | \
                    awk -F',' -v mem="$CAND_MEM" '$2 < 30 && $3 > mem {print $1}')

                for gpu in $CANDIDATE_GPUS; do
                    RUNNING_ON_GPU=$(jq -r --arg gpu "$gpu" '.running[] | select(.gpu == ($gpu | tonumber)) | .name' "$QUEUE_FILE" 2>/dev/null)
                    if [ -z "$RUNNING_ON_GPU" ]; then
                        IDLE_GPU=$gpu
                        break
                    fi
                done

                # If all GPUs have jobs, pick the one with fewest
                if [ -z "$IDLE_GPU" ] && [ -n "$CANDIDATE_GPUS" ]; then
                    IDLE_GPU=$(for gpu in $CANDIDATE_GPUS; do
                        COUNT=$(jq -r --arg gpu "$gpu" '[.running[] | select(.gpu == ($gpu | tonumber))] | length' "$QUEUE_FILE")
                        echo "$COUNT $gpu"
                    done | sort -n | head -1 | awk '{print $2}')
                fi
            fi

            if [ -n "$IDLE_GPU" ]; then
                NEXT_JOB="$CANDIDATE"
                NEXT_JOB_IDX="$idx"
                GPU_MEM_NEEDED="$CAND_MEM"
                break
            fi

            idx=$((idx+1))
        done

        if [ -z "$NEXT_JOB" ] || [ -z "$IDLE_GPU" ]; then
            release_lock
            sleep "$WATCH_INTERVAL"
            continue
        fi

        EXP_NAME=$(echo "$NEXT_JOB" | jq -r '.name')
        COMMAND=$(echo "$NEXT_JOB" | jq -r '.command')
        RETRY_COUNT=$(echo "$NEXT_JOB" | jq -r '.retry_count // 0')
        DEPENDS_ON_JSON=$(echo "$NEXT_JOB" | jq -r '.depends_on // [] | @json')
        # Restore launch environment captured at queue time
        JOB_WORKDIR=$(echo "$NEXT_JOB" | jq -r '.workdir // ""')
        JOB_PYTHONPATH=$(echo "$NEXT_JOB" | jq -r '.pythonpath // ""')
        JOB_VIRTUAL_ENV=$(echo "$NEXT_JOB" | jq -r '.virtual_env // ""')
        JOB_CONDA_ENV=$(echo "$NEXT_JOB" | jq -r '.conda_env // ""')
        [ -z "$JOB_WORKDIR" ] && JOB_WORKDIR="$PWD"

            echo "🚀 [$(date +'%H:%M:%S')] Launching $EXP_NAME on GPU $IDLE_GPU..."

            # Remove from queue by index (may not be queued[0] if earlier jobs have unmet deps)
            jq --argjson idx "$NEXT_JOB_IDX" '.queued = (.queued[:$idx] + .queued[$idx+1:])' \
                "$QUEUE_FILE" > "$QUEUE_FILE.tmp" && mv "$QUEUE_FILE.tmp" "$QUEUE_FILE"

            # Start experiment tracking
            EXP_FILE=$("$SCRIPT_DIR/start_experiment.sh" "$EXP_NAME" "Auto-launched from queue (attempt $((RETRY_COUNT + 1)))" "$IDLE_GPU")

            TIMESTAMP_START=$(date -Iseconds)

            # Snapshot GPU memory before launch for post-launch verification
            GPU_MEM_BEFORE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
                --id="$IDLE_GPU" 2>/dev/null | tr -d ' ' || echo "0")

            # Launch job monitor in background
            (
                # Restore launch environment captured at queue time
                cd "$JOB_WORKDIR"
                [ -n "$JOB_PYTHONPATH" ] && export PYTHONPATH="$JOB_PYTHONPATH"
                [ -n "$JOB_VIRTUAL_ENV" ] && export VIRTUAL_ENV="$JOB_VIRTUAL_ENV"
                [ -n "$JOB_CONDA_ENV" ]   && export CONDA_DEFAULT_ENV="$JOB_CONDA_ENV"
                export EXP_FILE="$EXP_FILE"

                START_TIME=$(date +%s)

                # Normalize --device cuda:X to --device cuda:0 — with CUDA_VISIBLE_DEVICES=N
                # the assigned GPU is always logical device 0. Explicit physical indices like
                # --device cuda:2 bypass CUDA_VISIBLE_DEVICES in some frameworks (lm-eval etc.)
                LAUNCH_CMD=$(echo "$COMMAND" | sed 's/--device cuda:[0-9]*/--device cuda:0/g')

                # CUDA_VISIBLE_DEVICES inlined in command string — ensures it's respected
                # even by code using device_map={"": 0} or torch.cuda.set_device() internally
                eval "CUDA_VISIBLE_DEVICES='$IDLE_GPU' $LAUNCH_CMD" > "$JOB_WORKDIR/experiments/${EXP_NAME}_output.log" 2>&1
                EXIT_CODE=$?

                END_TIME=$(date +%s)
                RUNTIME=$((END_TIME - START_TIME))

                # Remove from running queue (need lock for queue file access)
                if acquire_lock 10; then
                    jq --arg name "$EXP_NAME" \
                       '.running = [.running[] | select(.name != $name)]' "$QUEUE_FILE" > "$QUEUE_FILE.tmp" && mv "$QUEUE_FILE.tmp" "$QUEUE_FILE"
                    release_lock
                fi

                if [ $EXIT_CODE -eq 0 ]; then
                    # Success — record in completed list so dependent jobs can check
                    "$SCRIPT_DIR/complete_experiment.sh" completed "$EXP_FILE"
                    echo "✅ [$(date +'%H:%M:%S')] $EXP_NAME completed successfully"
                    if acquire_lock 10; then
                        jq --arg name "$EXP_NAME" \
                           --arg gpu "$IDLE_GPU" \
                           --arg completed_at "$(date -Iseconds)" \
                           '.completed += [{name: $name, gpu: ($gpu | tonumber), completed_at: $completed_at}]' \
                           "$QUEUE_FILE" > "$QUEUE_FILE.tmp" && mv "$QUEUE_FILE.tmp" "$QUEUE_FILE"
                        release_lock
                    fi
                elif [ $RUNTIME -lt 60 ]; then
                    # Fast failure - likely resource issue, retry
                    if [ $RETRY_COUNT -lt 3 ]; then
                        NEW_RETRY_COUNT=$((RETRY_COUNT + 1))
                        TIMESTAMP_NOW=$(date -Iseconds)
                        ERROR_MSG="Launch failed (exit $EXIT_CODE), retrying ($NEW_RETRY_COUNT/3)"

                        echo "⚠️  [$(date +'%H:%M:%S')] $EXP_NAME failed fast, re-queuing (attempt $NEW_RETRY_COUNT/3)"

                        # Re-queue with updated retry count (need lock)
                        if acquire_lock 10; then
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
                            release_lock
                        fi

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

            # Capture the actual PID of the background job
            JOB_PID=$!

            # Add to running queue with correct PID
            jq --arg name "$EXP_NAME" \
               --arg gpu "$IDLE_GPU" \
               --arg pid "$JOB_PID" \
               --arg started "$TIMESTAMP_START" \
               --arg exp_file "$EXP_FILE" \
               '.running += [{
                   name: $name,
                   gpu: ($gpu | tonumber),
                   pid: ($pid | tonumber),
                   started_at: $started,
                   exp_file: $exp_file
               }]' "$QUEUE_FILE" > "$QUEUE_FILE.tmp" && mv "$QUEUE_FILE.tmp" "$QUEUE_FILE"

        echo "✅ [$(date +'%H:%M:%S')] Launched $EXP_NAME on GPU $IDLE_GPU (PID: $JOB_PID)"

        # Hold lock for 45s to cover model loading window — prevents the next queued job
        # from seeing this GPU as free before its VRAM is actually claimed
        echo "⏸️  [$(date +'%H:%M:%S')] Holding lock for 45s (model loading window)..."
        sleep 45

        release_lock

        # Post-launch memory verification: check that the assigned GPU actually got used
        # Runs outside the lock — just diagnostic, doesn't block further launches
        GPU_MEM_AFTER=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
            --id="$IDLE_GPU" 2>/dev/null | tr -d ' ' || echo "0")
        GPU_MEM_DELTA=$(( GPU_MEM_AFTER - GPU_MEM_BEFORE ))
        if [ "$GPU_MEM_DELTA" -lt 500 ]; then
            echo "⚠️  [$(date +'%H:%M:%S')] GPU $IDLE_GPU memory delta only ${GPU_MEM_DELTA}MB after 45s" \
                 "— $EXP_NAME may not be using the assigned GPU (CUDA_VISIBLE_DEVICES ignored?)" \
                 >> "experiments/${EXP_NAME}_output.log"
            echo "⚠️  [$(date +'%H:%M:%S')] WARNING: $EXP_NAME shows <500MB VRAM delta on GPU $IDLE_GPU — check CUDA device assignment"
        else
            echo "✅ [$(date +'%H:%M:%S')] GPU $IDLE_GPU VRAM +${GPU_MEM_DELTA}MB confirmed for $EXP_NAME"
        fi
    fi

    sleep "$WATCH_INTERVAL"
done
