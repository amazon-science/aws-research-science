#!/bin/bash
# Queue watcher — sole owner of queue.json.
# Processes inbox files, monitors job completions, launches eligible jobs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Absolute paths anchored to CWD at startup ─────────────────────────────
# All subshells, background jobs and cd calls must use these absolute paths.
PROJECT_DIR="$(pwd)"
EXPERIMENTS_DIR="$PROJECT_DIR/experiments"
QUEUE_FILE="$EXPERIMENTS_DIR/queue.json"
INBOX_DIR="$EXPERIMENTS_DIR/queue_inbox"
DONE_DIR="$EXPERIMENTS_DIR/queue_done"
PID_FILE="$EXPERIMENTS_DIR/queue_watcher.pid"
WATCH_INTERVAL=10

echo "🔄 Queue watcher started (PID $$, checking every ${WATCH_INTERVAL}s)"
echo "   Project: $PROJECT_DIR"
mkdir -p "$EXPERIMENTS_DIR" "$INBOX_DIR" "$DONE_DIR"
echo $$ > "$PID_FILE"

# ── Helpers ────────────────────────────────────────────────────────────────
write_queue() {
    # Atomic write: write to .tmp then rename to avoid partial-file corruption
    local content="$1"
    echo "$content" > "${QUEUE_FILE}.tmp" && mv "${QUEUE_FILE}.tmp" "$QUEUE_FILE"
}

# ── Initialise queue file ──────────────────────────────────────────────────
init_queue() {
    if [ ! -f "$QUEUE_FILE" ]; then
        write_queue '{"queued": [], "running": [], "completed": []}'
        return
    fi
    local tmp
    tmp=$(jq 'if .completed == null then .completed = [] else . end' "$QUEUE_FILE") && \
        write_queue "$tmp"
}

# ── Seed completed[] from experiment JSON files ───────────────────────────
seed_completed_from_experiments() {
    local seeded=0
    for exp_file in \
        "$EXPERIMENTS_DIR"/session-*/exp_*.json \
        "$EXPERIMENTS_DIR"/exp_*.json; do
        [ -f "$exp_file" ] || continue
        local status name gpu completed_at already
        status=$(jq -r '.status // ""' "$exp_file" 2>/dev/null)
        [ "$status" = "completed" ] || continue
        name=$(jq -r '.name // ""' "$exp_file" 2>/dev/null)
        [ -z "$name" ] && continue
        gpu=$(jq -r '.gpu // 0' "$exp_file" 2>/dev/null)
        completed_at=$(jq -r '.end_time // ""' "$exp_file" 2>/dev/null)
        already=$(jq -r --arg n "$name" \
            '[.completed[] | select(.name == $n)] | length' \
            "$QUEUE_FILE" 2>/dev/null || echo "0")
        if [ "$already" = "0" ]; then
            local tmp
            tmp=$(jq --arg n "$name" --arg g "$gpu" --arg t "$completed_at" \
                '.completed += [{name: $n, gpu: ($g|tonumber), completed_at: $t}]' \
                "$QUEUE_FILE") && write_queue "$tmp"
            seeded=$((seeded+1))
        fi
    done
    [ $seeded -gt 0 ] && echo "📋 [$(date +'%H:%M:%S')] Seeded $seeded completed experiments"
}

# ── Process inbox files ────────────────────────────────────────────────────
process_inbox() {
    for job_file in "$INBOX_DIR"/job_*.json; do
        [ -f "$job_file" ] || continue
        local job name
        job=$(cat "$job_file" 2>/dev/null) || continue
        name=$(echo "$job" | jq -r '.name // ""')
        [ -z "$name" ] && { rm -f "$job_file"; continue; }
        local tmp
        tmp=$(jq --argjson job "$job" '.queued += [$job]' "$QUEUE_FILE") && \
            write_queue "$tmp"
        rm -f "$job_file"
        echo "📥 [$(date +'%H:%M:%S')] Picked up: $name"
    done
}

# ── Process completed/failed job signals ──────────────────────────────────
process_done() {
    for done_file in "$DONE_DIR"/*.json; do
        [ -f "$done_file" ] || continue
        local info exit_code exp_name runtime exp_file gpu retry_count
        info=$(cat "$done_file" 2>/dev/null) || continue
        rm -f "$done_file"   # remove immediately so duplicate processing is impossible

        exit_code=$(echo "$info" | jq -r '.exit_code // 1')
        exp_name=$(echo "$info"  | jq -r '.name // ""')
        runtime=$(echo "$info"   | jq -r '.runtime // 0')
        exp_file=$(echo "$info"  | jq -r '.exp_file // ""')
        gpu=$(echo "$info"       | jq -r '.gpu // 0')
        retry_count=$(echo "$info" | jq -r '.retry_count // 0')

        [ -z "$exp_name" ] && continue

        # Remove from running[]
        local tmp
        tmp=$(jq --arg n "$exp_name" \
            '.running = [.running[] | select(.name != $n)]' \
            "$QUEUE_FILE") && write_queue "$tmp"

        if [ "$exit_code" = "0" ]; then
            echo "✅ [$(date +'%H:%M:%S')] $exp_name completed"
            [ -n "$exp_file" ] && \
                "$SCRIPT_DIR/complete_experiment.sh" completed "$exp_file" 2>/dev/null || true
            tmp=$(jq --arg n "$exp_name" --arg g "$gpu" --arg t "$(date -Iseconds)" \
                '.completed += [{name: $n, gpu: ($g|tonumber), completed_at: $t}]' \
                "$QUEUE_FILE") && write_queue "$tmp"

        elif [ "${runtime}" -lt 60 ] 2>/dev/null && [ "${retry_count}" -lt 3 ]; then
            local new_retry=$((retry_count+1))
            echo "⚠️  [$(date +'%H:%M:%S')] $exp_name failed fast — re-queuing ($new_retry/3)"
            [ -n "$exp_file" ] && \
                "$SCRIPT_DIR/complete_experiment.sh" failed "$exp_file" 2>/dev/null || true
            local cmd workdir pythonpath venv conda mem deps
            cmd=$(echo "$info"        | jq -r '.command // ""')
            workdir=$(echo "$info"    | jq -r '.workdir // ""')
            pythonpath=$(echo "$info" | jq -r '.pythonpath // ""')
            venv=$(echo "$info"       | jq -r '.virtual_env // ""')
            conda=$(echo "$info"      | jq -r '.conda_env // ""')
            mem=$(echo "$info"        | jq -r '.gpu_mem_needed // 8000')
            deps=$(echo "$info"       | jq -c '.depends_on // []')
            tmp=$(jq \
                --arg  n  "$exp_name" \
                --arg  c  "$cmd"      \
                --arg  m  "$mem"      \
                --argjson retry "$new_retry" \
                --argjson deps  "$deps"      \
                --arg  w  "$workdir"  \
                --arg  pp "$pythonpath" \
                --arg  ve "$venv"     \
                --arg  ce "$conda"    \
                '.queued += [{
                    name: $n, command: $c,
                    gpu_mem_needed: ($m|tonumber),
                    queued_at: (now|todate),
                    status: "waiting",
                    retry_count: $retry,
                    workdir: $w, pythonpath: $pp,
                    virtual_env: $ve, conda_env: $ce,
                    depends_on: $deps,
                    notes: ["retry \($retry)/3"]
                }]' "$QUEUE_FILE") && write_queue "$tmp"
        else
            echo "❌ [$(date +'%H:%M:%S')] $exp_name failed (runtime ${runtime}s)"
            [ -n "$exp_file" ] && \
                "$SCRIPT_DIR/complete_experiment.sh" failed "$exp_file" 2>/dev/null || true
        fi
    done
}

# ── Clean up dead PIDs (safety net) ───────────────────────────────────────
cleanup_dead_pids() {
    local running_jobs
    running_jobs=$(jq -r '.running[] | @json' "$QUEUE_FILE" 2>/dev/null)
    [ -z "$running_jobs" ] && return

    while IFS= read -r job; do
        [ -z "$job" ] || [ "$job" = "null" ] && continue
        local pid name
        pid=$(echo "$job"  | jq -r '.pid')
        name=$(echo "$job" | jq -r '.name')
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "🧹 [$(date +'%H:%M:%S')] Dead PID $pid ($name) — creating done signal"
            jq -n \
                --arg  n  "$name" \
                --argjson g  "$(echo "$job" | jq '.gpu')" \
                '{ exit_code: 1, name: $n, runtime: 999,
                   gpu: $g, retry_count: 3 }' \
                > "$DONE_DIR/${name}_dead_$(date +%s).json"
        fi
    done <<< "$running_jobs"
}

# ── Find and launch next eligible job — returns 0 if launched, 1 if not ──
launch_next() {
    local queue_len
    queue_len=$(jq '.queued | length' "$QUEUE_FILE" 2>/dev/null || echo 0)
    [ "$queue_len" -eq 0 ] && return 1

    # Snapshot GPU status once per call
    local gpu_info=""
    command -v nvidia-smi &>/dev/null && \
        gpu_info=$(nvidia-smi \
            --query-gpu=index,utilization.gpu,memory.free \
            --format=csv,noheader,nounits 2>/dev/null)

    local idx=0
    while [ $idx -lt "$queue_len" ]; do
        local candidate cand_name cand_mem cand_deps deps_met
        candidate=$(jq -r --argjson i "$idx" '.queued[$i] | @json' "$QUEUE_FILE")
        [ "$candidate" = "null" ] && { idx=$((idx+1)); continue; }

        cand_name=$(echo "$candidate" | jq -r '.name')
        cand_mem=$(echo "$candidate"  | jq -r '.gpu_mem_needed // 8000')
        cand_deps=$(echo "$candidate" | jq -c '.depends_on // []')

        # Check all dependencies are satisfied
        deps_met=$(jq --argjson deps "$cand_deps" '
            if ($deps | length) == 0 then true
            else ($deps | map(. as $d | any(.completed[]; .name == $d)) | all)
            end' "$QUEUE_FILE")

        if [ "$deps_met" != "true" ]; then
            idx=$((idx+1)); continue
        fi

        # Determine which GPU to use
        local idle_gpu=""
        if [ "$cand_deps" != "[]" ]; then
            # --after job: inherit GPU from last completed dependency
            idle_gpu=$(jq -r --argjson deps "$cand_deps" '
                [.completed[] | select(.name as $n | $deps | index($n) != null)]
                | sort_by(.completed_at) | last | .gpu | tostring' "$QUEUE_FILE")

            # Verify inherited GPU has enough free memory
            local gpu_free
            gpu_free=$(echo "$gpu_info" | awk -F',' -v g="$idle_gpu" \
                'int($1) == int(g) {gsub(/ /,""); print int($3)}')
            if [ -z "$gpu_free" ] || [ "${gpu_free:-0}" -lt "$cand_mem" ]; then
                echo "⏳ [$(date +'%H:%M:%S')] $cand_name: waiting for GPU $idle_gpu (need ${cand_mem}MB, free: ${gpu_free:-?}MB)"
                idx=$((idx+1)); continue
            fi
        else
            # No deps: find any free GPU not already in running[]
            while IFS=',' read -r gidx util mem_free; do
                gidx=$(echo "$gidx"     | tr -d ' ')
                util=$(echo "$util"     | tr -d ' ')
                mem_free=$(echo "$mem_free" | tr -d ' ')
                [ "${util:-100}" -lt 30 ] && [ "${mem_free:-0}" -gt "$cand_mem" ] || continue
                local running_on
                running_on=$(jq -r --arg g "$gidx" \
                    '.running[] | select(.gpu == ($g|tonumber)) | .name' \
                    "$QUEUE_FILE" 2>/dev/null)
                if [ -z "$running_on" ]; then
                    idle_gpu="$gidx"; break
                fi
            done <<< "$gpu_info"
        fi

        [ -z "$idle_gpu" ] && { idx=$((idx+1)); continue; }

        # ── Launch ───────────────────────────────────────────────────────
        local exp_name cmd workdir pythonpath venv conda retry
        exp_name="$cand_name"
        cmd=$(echo "$candidate"       | jq -r '.command')
        workdir=$(echo "$candidate"   | jq -r '.workdir // ""')
        pythonpath=$(echo "$candidate"| jq -r '.pythonpath // ""')
        venv=$(echo "$candidate"      | jq -r '.virtual_env // ""')
        conda=$(echo "$candidate"     | jq -r '.conda_env // ""')
        retry=$(echo "$candidate"     | jq -r '.retry_count // 0')
        [ -z "$workdir" ] && workdir="$PROJECT_DIR"

        echo "🚀 [$(date +'%H:%M:%S')] Launching $exp_name on GPU $idle_gpu..."

        # Remove from queued[] atomically
        local tmp
        tmp=$(jq --argjson i "$idx" \
            '.queued = (.queued[:$i] + .queued[$i+1:])' \
            "$QUEUE_FILE") && write_queue "$tmp"

        # Start experiment tracking
        local exp_file=""
        exp_file=$("$SCRIPT_DIR/start_experiment.sh" \
            "$exp_name" "Auto-launched (attempt $((retry+1)))" \
            "$idle_gpu" 2>/dev/null) || exp_file=""

        local started_at; started_at=$(date -Iseconds)
        local mem_before
        mem_before=$(nvidia-smi --query-gpu=memory.used \
            --format=csv,noheader,nounits --id="$idle_gpu" \
            2>/dev/null | tr -d ' ' || echo 0)

        # Payload for done file — pass absolute DONE_DIR so cd doesn't break it
        local payload abs_done_dir
        abs_done_dir="$DONE_DIR"
        payload=$(echo "$candidate" | jq \
            --arg gpu "$idle_gpu" --arg ef "$exp_file" \
            '. + {gpu: ($gpu|tonumber), exp_file: $ef}')

        # Background job — uses absolute paths, immune to cd
        (
            set +e
            cd "$workdir" 2>/dev/null || true
            [ -n "$pythonpath" ] && export PYTHONPATH="$pythonpath"
            [ -n "$venv"       ] && export VIRTUAL_ENV="$venv"
            [ -n "$conda"      ] && export CONDA_DEFAULT_ENV="$conda"
            [ -n "$exp_file"   ] && export EXP_FILE="$exp_file"

            START=$(date +%s)
            LAUNCH_CMD=$(echo "$cmd" | sed 's/--device cuda:[0-9]*/--device cuda:0/g')
            eval "CUDA_VISIBLE_DEVICES=$idle_gpu $LAUNCH_CMD" \
                >> "$EXPERIMENTS_DIR/${exp_name}_output.log" 2>&1
            EXIT_CODE=$?
            END=$(date +%s)
            RUNTIME=$((END - START))

            # Write done file to absolute path — unique name avoids collision on retries
            mkdir -p "$abs_done_dir"
            echo "$payload" | jq \
                --argjson ec "$EXIT_CODE" --argjson rt "$RUNTIME" \
                '. + {exit_code: $ec, runtime: $rt}' \
                > "$abs_done_dir/${exp_name}_$(date +%s).json"
        ) &

        local job_pid=$!

        # Record in running[]
        tmp=$(jq \
            --arg  n  "$exp_name"  \
            --arg  g  "$idle_gpu"  \
            --arg  p  "$job_pid"   \
            --arg  t  "$started_at" \
            --arg  ef "$exp_file"   \
            '.running += [{
                name: $n, gpu: ($g|tonumber), pid: ($p|tonumber),
                started_at: $t, exp_file: $ef
            }]' "$QUEUE_FILE") && write_queue "$tmp"

        echo "✅ [$(date +'%H:%M:%S')] Launched $exp_name on GPU $idle_gpu (PID: $job_pid)"

        # Post-launch VRAM sanity check (non-blocking, 30s later)
        (
            sleep 30
            mem_after=$(nvidia-smi --query-gpu=memory.used \
                --format=csv,noheader,nounits --id="$idle_gpu" \
                2>/dev/null | tr -d ' ' || echo 0)
            delta=$((mem_after - mem_before))
            if [ "$delta" -lt 500 ]; then
                echo "⚠️  [$(date +'%H:%M:%S')] $exp_name: <500MB VRAM delta on GPU $idle_gpu — check CUDA assignment"
            fi
        ) &

        return 0  # signal: launched one, caller should loop
    done

    return 1  # nothing could be launched
}

# ── Main loop ──────────────────────────────────────────────────────────────
init_queue
seed_completed_from_experiments

while true; do
    [ -f "$QUEUE_FILE" ] || init_queue

    process_done
    process_inbox
    cleanup_dead_pids

    # Launch all eligible jobs this cycle — not just one
    while launch_next; do : ; done

    sleep "$WATCH_INTERVAL"
done
