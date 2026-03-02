#!/bin/bash
# Queue watcher — sole owner of queue.json.
# Processes inbox files, monitors job completions, launches eligible jobs.
# No external process writes queue.json directly.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUEUE_FILE="experiments/queue.json"
INBOX_DIR="experiments/queue_inbox"
DONE_DIR="experiments/queue_done"
WATCH_INTERVAL=10  # seconds — fast since no lock contention

echo "🔄 Queue watcher started (checking every ${WATCH_INTERVAL}s)"
mkdir -p experiments "$INBOX_DIR" "$DONE_DIR"

# ── Initialise queue file ──────────────────────────────────────────────────
init_queue() {
    if [ ! -f "$QUEUE_FILE" ]; then
        echo '{"queued": [], "running": [], "completed": []}' > "$QUEUE_FILE"
    fi
    # Migrate: ensure completed array exists
    local tmp
    tmp=$(jq 'if .completed == null then .completed = [] else . end' "$QUEUE_FILE") && \
        echo "$tmp" > "$QUEUE_FILE"
}

# ── Seed completed[] from experiment JSON files ───────────────────────────
seed_completed_from_experiments() {
    local seeded=0
    for exp_file in experiments/session-*/exp_*.json experiments/exp_*.json; do
        [ -f "$exp_file" ] || continue
        local status name gpu completed_at already
        status=$(jq -r '.status // ""' "$exp_file" 2>/dev/null)
        [ "$status" = "completed" ] || continue
        name=$(jq -r '.name // ""' "$exp_file" 2>/dev/null)
        [ -z "$name" ] && continue
        gpu=$(jq -r '.gpu // 0' "$exp_file" 2>/dev/null)
        completed_at=$(jq -r '.end_time // ""' "$exp_file" 2>/dev/null)
        already=$(jq -r --arg n "$name" '[.completed[] | select(.name == $n)] | length' "$QUEUE_FILE" 2>/dev/null || echo "0")
        if [ "$already" = "0" ]; then
            local tmp
            tmp=$(jq --arg n "$name" --arg g "$gpu" --arg t "$completed_at" \
                '.completed += [{name: $n, gpu: ($g | tonumber), completed_at: $t}]' \
                "$QUEUE_FILE") && echo "$tmp" > "$QUEUE_FILE"
            seeded=$((seeded+1))
        fi
    done
    [ $seeded -gt 0 ] && echo "📋 [$(date +'%H:%M:%S')] Seeded $seeded completed experiments"
}

# ── Process inbox files ────────────────────────────────────────────────────
process_inbox() {
    local count=0
    for job_file in "$INBOX_DIR"/job_*.json; do
        [ -f "$job_file" ] || continue
        local job name
        job=$(cat "$job_file" 2>/dev/null) || continue
        name=$(echo "$job" | jq -r '.name // ""')
        [ -z "$name" ] && { rm -f "$job_file"; continue; }

        local tmp
        tmp=$(echo "$job" | jq -s --argjson q "$(cat "$QUEUE_FILE")" \
            '$q | .queued += [$input[0]]' /dev/stdin 2>/dev/null)
        # simpler approach:
        tmp=$(jq --argjson job "$job" '.queued += [$job]' "$QUEUE_FILE") && \
            echo "$tmp" > "$QUEUE_FILE"
        rm -f "$job_file"
        echo "📥 [$(date +'%H:%M:%S')] Picked up: $name"
        count=$((count+1))
    done
    return $count
}

# ── Process completed/failed job signals ──────────────────────────────────
process_done() {
    for done_file in "$DONE_DIR"/*.json; do
        [ -f "$done_file" ] || continue
        local info exit_code exp_name runtime exp_file gpu retry_count
        info=$(cat "$done_file" 2>/dev/null) || continue
        exit_code=$(echo "$info" | jq -r '.exit_code // 1')
        exp_name=$(echo "$info" | jq -r '.name // ""')
        runtime=$(echo "$info" | jq -r '.runtime // 0')
        exp_file=$(echo "$info" | jq -r '.exp_file // ""')
        gpu=$(echo "$info" | jq -r '.gpu // 0')
        retry_count=$(echo "$info" | jq -r '.retry_count // 0')
        rm -f "$done_file"

        [ -z "$exp_name" ] && continue

        # Remove from running[]
        local tmp
        tmp=$(jq --arg n "$exp_name" '.running = [.running[] | select(.name != $n)]' "$QUEUE_FILE") && \
            echo "$tmp" > "$QUEUE_FILE"

        if [ "$exit_code" = "0" ]; then
            echo "✅ [$(date +'%H:%M:%S')] $exp_name completed"
            [ -n "$exp_file" ] && "$SCRIPT_DIR/complete_experiment.sh" completed "$exp_file" 2>/dev/null
            tmp=$(jq --arg n "$exp_name" --arg g "$gpu" --arg t "$(date -Iseconds)" \
                '.completed += [{name: $n, gpu: ($g | tonumber), completed_at: $t}]' \
                "$QUEUE_FILE") && echo "$tmp" > "$QUEUE_FILE"

        elif [ "$runtime" -lt 60 ] 2>/dev/null && [ "$retry_count" -lt 3 ]; then
            local new_retry=$((retry_count+1))
            echo "⚠️  [$(date +'%H:%M:%S')] $exp_name failed fast, re-queuing ($new_retry/3)"
            [ -n "$exp_file" ] && "$SCRIPT_DIR/complete_experiment.sh" failed "$exp_file" 2>/dev/null
            # Re-queue with incremented retry count
            local cmd workdir pythonpath venv conda mem deps
            cmd=$(echo "$info" | jq -r '.command // ""')
            workdir=$(echo "$info" | jq -r '.workdir // ""')
            pythonpath=$(echo "$info" | jq -r '.pythonpath // ""')
            venv=$(echo "$info" | jq -r '.virtual_env // ""')
            conda=$(echo "$info" | jq -r '.conda_env // ""')
            mem=$(echo "$info" | jq -r '.gpu_mem_needed // 8000')
            deps=$(echo "$info" | jq -r '.depends_on // [] | @json')
            tmp=$(jq --arg n "$exp_name" --arg c "$cmd" --arg m "$mem" \
                --argjson retry "$new_retry" --argjson deps "$deps" \
                --arg w "$workdir" --arg pp "$pythonpath" --arg ve "$venv" --arg ce "$conda" \
                '.queued += [{name: $n, command: $c, gpu_mem_needed: ($m|tonumber),
                  queued_at: now|todate, status: "waiting", retry_count: $retry,
                  workdir: $w, pythonpath: $pp, virtual_env: $ve, conda_env: $ce,
                  depends_on: $deps, notes: ["retry \($retry)/3"]}]' \
                "$QUEUE_FILE") && echo "$tmp" > "$QUEUE_FILE"
        else
            echo "❌ [$(date +'%H:%M:%S')] $exp_name failed (runtime ${runtime}s)"
            [ -n "$exp_file" ] && "$SCRIPT_DIR/complete_experiment.sh" failed "$exp_file" 2>/dev/null
        fi
    done
}

# ── Clean up dead PIDs (safety net) ───────────────────────────────────────
cleanup_dead_pids() {
    local running_jobs dead_jobs=()
    running_jobs=$(jq -r '.running[] | @json' "$QUEUE_FILE" 2>/dev/null)
    [ -z "$running_jobs" ] && return

    while IFS= read -r job; do
        [ -z "$job" ] || [ "$job" = "null" ] && continue
        local pid name
        pid=$(echo "$job" | jq -r '.pid')
        name=$(echo "$job" | jq -r '.name')
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "🧹 [$(date +'%H:%M:%S')] Dead PID $pid ($name) — creating done signal"
            dead_jobs+=("$name")
            # Create a done file so process_done handles it properly
            echo "{\"exit_code\": 1, \"name\": \"$name\", \"runtime\": 999, \"gpu\": $(echo "$job" | jq '.gpu'), \"retry_count\": 3}" \
                > "$DONE_DIR/${name}.json"
        fi
    done <<< "$running_jobs"
}

# ── Find and launch next eligible job ─────────────────────────────────────
launch_next() {
    local queue_len idx
    queue_len=$(jq '.queued | length' "$QUEUE_FILE" 2>/dev/null || echo 0)
    [ "$queue_len" -eq 0 ] && return

    # Get current GPU status once
    local gpu_info=""
    command -v nvidia-smi &>/dev/null && \
        gpu_info=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.free \
            --format=csv,noheader,nounits 2>/dev/null)

    idx=0
    while [ $idx -lt "$queue_len" ]; do
        local candidate cand_name cand_mem cand_deps deps_met
        candidate=$(jq -r --argjson i "$idx" '.queued[$i] | @json' "$QUEUE_FILE")
        [ "$candidate" = "null" ] && { idx=$((idx+1)); continue; }

        cand_name=$(echo "$candidate" | jq -r '.name')
        cand_mem=$(echo "$candidate" | jq -r '.gpu_mem_needed // 8000')
        cand_deps=$(echo "$candidate" | jq -r '.depends_on // [] | @json')

        # Check dependencies
        deps_met=$(jq --argjson deps "$cand_deps" '
            if ($deps | length) == 0 then true
            else ($deps | map(. as $d | any(.completed[]; .name == $d)) | all)
            end' "$QUEUE_FILE")

        if [ "$deps_met" != "true" ]; then
            idx=$((idx+1)); continue
        fi

        # Determine GPU
        local idle_gpu=""
        if [ "$cand_deps" != "[]" ]; then
            # Inherit GPU from last completed dependency
            idle_gpu=$(jq -r --argjson deps "$cand_deps" '
                [.completed[] | select(.name as $n | $deps | index($n) != null)]
                | sort_by(.completed_at) | last | .gpu | tostring' "$QUEUE_FILE")
            local gpu_free
            gpu_free=$(echo "$gpu_info" | awk -F',' -v g="$idle_gpu" \
                '$1+0 == g+0 {print $3+0}' | tr -d ' ')
            if [ -z "$gpu_free" ] || [ "${gpu_free:-0}" -lt "$cand_mem" ]; then
                echo "⏳ [$(date +'%H:%M:%S')] $cand_name: waiting for GPU $idle_gpu (need ${cand_mem}MB)"
                idx=$((idx+1)); continue
            fi
        else
            # Find any free GPU not in running[]
            while IFS=',' read -r gidx util mem_free; do
                gidx=$(echo "$gidx" | tr -d ' ')
                util=$(echo "$util" | tr -d ' ')
                mem_free=$(echo "$mem_free" | tr -d ' ')
                [ "${util:-100}" -lt 30 ] && [ "${mem_free:-0}" -gt "$cand_mem" ] || continue
                local running_on
                running_on=$(jq -r --arg g "$gidx" \
                    '.running[] | select(.gpu == ($g | tonumber)) | .name' \
                    "$QUEUE_FILE" 2>/dev/null)
                if [ -z "$running_on" ]; then
                    idle_gpu="$gidx"; break
                fi
            done <<< "$gpu_info"
        fi

        [ -z "$idle_gpu" ] && { idx=$((idx+1)); continue; }

        # ── Launch the job ────────────────────────────────────────────────
        local exp_name cmd workdir pythonpath venv conda retry
        exp_name="$cand_name"
        cmd=$(echo "$candidate" | jq -r '.command')
        workdir=$(echo "$candidate" | jq -r '.workdir // ""')
        pythonpath=$(echo "$candidate" | jq -r '.pythonpath // ""')
        venv=$(echo "$candidate" | jq -r '.virtual_env // ""')
        conda=$(echo "$candidate" | jq -r '.conda_env // ""')
        retry=$(echo "$candidate" | jq -r '.retry_count // 0')
        [ -z "$workdir" ] && workdir="$PWD"

        echo "🚀 [$(date +'%H:%M:%S')] Launching $exp_name on GPU $idle_gpu..."

        # Remove from queued[]
        local tmp
        tmp=$(jq --argjson i "$idx" '.queued = (.queued[:$i] + .queued[$i+1:])' \
            "$QUEUE_FILE") && echo "$tmp" > "$QUEUE_FILE"

        # Start experiment tracking
        local exp_file
        exp_file=$("$SCRIPT_DIR/start_experiment.sh" "$exp_name" \
            "Auto-launched (attempt $((retry+1)))" "$idle_gpu" 2>/dev/null) || exp_file=""

        local started_at; started_at=$(date -Iseconds)
        local mem_before
        mem_before=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
            --id="$idle_gpu" 2>/dev/null | tr -d ' ' || echo 0)

        # Build launch payload for done file
        local payload
        payload=$(echo "$candidate" | jq \
            --arg gpu "$idle_gpu" --arg ef "$exp_file" \
            '. + {gpu: ($gpu|tonumber), exp_file: $ef}')

        # Launch background job — writes done file on exit
        (
            [ -n "$workdir" ] && cd "$workdir"
            [ -n "$pythonpath" ] && export PYTHONPATH="$pythonpath"
            [ -n "$venv" ] && export VIRTUAL_ENV="$venv"
            [ -n "$conda" ] && export CONDA_DEFAULT_ENV="$conda"
            [ -n "$exp_file" ] && export EXP_FILE="$exp_file"

            START=$(date +%s)
            LAUNCH_CMD=$(echo "$cmd" | sed 's/--device cuda:[0-9]*/--device cuda:0/g')
            eval "CUDA_VISIBLE_DEVICES=$idle_gpu $LAUNCH_CMD" \
                > "${workdir}/experiments/${exp_name}_output.log" 2>&1
            EXIT_CODE=$?
            END=$(date +%s)
            RUNTIME=$((END - START))

            mkdir -p "$DONE_DIR"
            echo "$payload" | jq \
                --argjson ec "$EXIT_CODE" --argjson rt "$RUNTIME" \
                '. + {exit_code: $ec, runtime: $rt}' \
                > "$DONE_DIR/${exp_name}.json"
        ) &

        local job_pid=$!

        # Record in running[]
        tmp=$(jq --arg n "$exp_name" --arg g "$idle_gpu" --arg p "$job_pid" \
            --arg t "$started_at" --arg ef "$exp_file" \
            '.running += [{name: $n, gpu: ($g|tonumber), pid: ($p|tonumber),
              started_at: $t, exp_file: $ef}]' \
            "$QUEUE_FILE") && echo "$tmp" > "$QUEUE_FILE"

        echo "✅ [$(date +'%H:%M:%S')] Launched $exp_name on GPU $idle_gpu (PID: $job_pid)"

        # Post-launch VRAM check after 30s in background (non-blocking)
        (
            sleep 30
            mem_after=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
                --id="$idle_gpu" 2>/dev/null | tr -d ' ' || echo 0)
            delta=$((mem_after - mem_before))
            if [ "$delta" -lt 500 ]; then
                echo "⚠️  [$(date +'%H:%M:%S')] WARNING: $exp_name <500MB VRAM delta on GPU $idle_gpu — check CUDA assignment"
            fi
        ) &

        return 0  # launched one job, let caller loop again
    done
}

# ── Main loop ──────────────────────────────────────────────────────────────
init_queue
seed_completed_from_experiments

while true; do
    [ -f "$QUEUE_FILE" ] || init_queue

    process_done
    process_inbox
    cleanup_dead_pids
    launch_next

    sleep "$WATCH_INTERVAL"
done
