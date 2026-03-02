#!/bin/bash
# Fast dashboard — bash + jq, no Python required

QUEUE_FILE="experiments/queue.json"

# ── Header ────────────────────────────────────────────────────────────────
NOW=$(date '+%H:%M:%S')
TOTAL=0; DONE=0; FAIL=0; RUNNING_EXP=0
if [ -f "$QUEUE_FILE" ]; then
    Q_RUNNING=$(jq '.running | length' "$QUEUE_FILE" 2>/dev/null || echo 0)
    Q_QUEUED=$(jq '.queued | length' "$QUEUE_FILE" 2>/dev/null || echo 0)
fi

# Count experiments
EXP_FILES=$(find experiments -name "exp_*.json" 2>/dev/null)
if [ -n "$EXP_FILES" ]; then
    while IFS= read -r f; do
        [ -f "$f" ] || continue
        s=$(jq -r '.status // ""' "$f" 2>/dev/null)
        TOTAL=$((TOTAL+1))
        case "$s" in completed) DONE=$((DONE+1)) ;;
                      failed)   FAIL=$((FAIL+1)) ;;
                      running)  RUNNING_EXP=$((RUNNING_EXP+1)) ;;
        esac
    done <<< "$EXP_FILES"
fi

printf "ML Dashboard  %s   %d exp  +%d -%d ~%d" \
    "$NOW" "$TOTAL" "$DONE" "$FAIL" "$RUNNING_EXP"

if [ "${Q_RUNNING:-0}" -gt 0 ] || [ "${Q_QUEUED:-0}" -gt 0 ]; then
    printf "   %d running  %d queued" "${Q_RUNNING:-0}" "${Q_QUEUED:-0}"
fi
echo ""
echo ""

# ── Queue ─────────────────────────────────────────────────────────────────
if [ -f "$QUEUE_FILE" ]; then
    RUNNING_COUNT=$(jq '.running | length' "$QUEUE_FILE" 2>/dev/null || echo 0)
    QUEUED_COUNT=$(jq '.queued | length' "$QUEUE_FILE" 2>/dev/null || echo 0)

    if [ "$RUNNING_COUNT" -gt 0 ] || [ "$QUEUED_COUNT" -gt 0 ]; then
        echo "Queue"
        if [ "$RUNNING_COUNT" -gt 0 ]; then
            jq -r '.running[] | "  \(.name)  GPU \(.gpu)  running  \(.started_at[:16] // "")"' \
                "$QUEUE_FILE" 2>/dev/null
        fi
        if [ "$QUEUED_COUNT" -gt 0 ]; then
            jq -r '.queued[] | "  \(.name)  --  waiting\(if .retry_count > 0 then " (retry \(.retry_count)/3)" else "" end)"' \
                "$QUEUE_FILE" 2>/dev/null
        fi
        echo ""
    fi
fi

# ── GPUs ──────────────────────────────────────────────────────────────────
if command -v nvidia-smi &>/dev/null; then
    echo "GPUs"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu \
        --format=csv,noheader,nounits 2>/dev/null | \
    while IFS=',' read -r idx name util mem_used mem_total temp; do
        idx=$(echo "$idx" | xargs)
        name=$(echo "$name" | xargs | cut -c1-12)
        util=$(echo "$util" | xargs)
        mem_used=$(echo "$mem_used" | xargs)
        mem_total=$(echo "$mem_total" | xargs)
        temp=$(echo "$temp" | xargs)
        mem_gb=$(( mem_used / 1024 ))
        tot_gb=$(( mem_total / 1024 ))
        printf "  GPU %s  %-12s  %3d%%  %2d/%dGB  %d°C\n" \
            "$idx" "$name" "$util" "$mem_gb" "$tot_gb" "$temp"
    done
    echo ""
fi

# ── Processes ─────────────────────────────────────────────────────────────
PROCS=$(ps aux 2>/dev/null | grep -E 'python.*(train|experiment|eval)' | grep -v grep | head -3)
if [ -n "$PROCS" ]; then
    echo "Processes"
    echo "$PROCS" | awk '{
        pid=$2; cpu=$3; mem=$4
        cmd=""
        for(i=11;i<=NF && length(cmd)<60;i++) cmd=cmd" "$i
        printf "  %-8s  %5s%%  %4s%%  %s\n", pid, cpu, mem, substr(cmd,1,60)
    }'
    echo ""
fi

# ── Experiments ───────────────────────────────────────────────────────────
if [ -n "$EXP_FILES" ]; then
    echo "Experiments"
    # Sort by modification time, show most recent 10
    echo "$EXP_FILES" | xargs ls -t 2>/dev/null | head -10 | while IFS= read -r f; do
        [ -f "$f" ] || continue
        name=$(jq -r '.name // "?"' "$f" 2>/dev/null)
        status=$(jq -r '.status // "?"' "$f" 2>/dev/null)
        gpu=$(jq -r '.gpu // "?"' "$f" 2>/dev/null)
        metrics=$(jq -r '.metrics // {} | to_entries[:2] | map("\(.key)=\(.value | if type=="number" then (.*1000|round/1000|tostring) else . end)") | join("  ")' "$f" 2>/dev/null)

        case "$status" in
            completed) st="ok " ;;
            running)   st="~  " ;;
            failed)    st="err" ;;
            *)         st="-  " ;;
        esac

        if [ -n "$metrics" ]; then
            printf "  %s  %-30s  GPU:%-2s  %s\n" "$st" "$name" "$gpu" "$metrics"
        else
            printf "  %s  %-30s  GPU:%-2s\n" "$st" "$name" "$gpu"
        fi
    done
fi
