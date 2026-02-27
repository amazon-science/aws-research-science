#!/bin/bash
# List experiments from JSON files written by the queue system

QUEUE_FILE="experiments/queue.json"

# Collect all experiment JSON files, newest first
EXP_FILES=$(find experiments -name "exp_*.json" 2>/dev/null | xargs ls -t 2>/dev/null)

if [ -z "$EXP_FILES" ]; then
    echo "No experiments found."
    echo ""
    echo "💡 Queue an experiment:"
    echo "   ./scripts/queue_experiment.sh <name> 'python train.py' [gpu_mem_mb]"
    exit 0
fi

echo "📊 EXPERIMENTS (recent 20)"
echo "──────────────────────────────────────────────────────────────────"

no_metrics_count=0
shown=0

while IFS= read -r exp_file; do
    [ -f "$exp_file" ] || continue
    [ $shown -ge 20 ] && break

    name=$(jq -r '.name // "unknown"' "$exp_file" 2>/dev/null)
    status=$(jq -r '.status // "unknown"' "$exp_file" 2>/dev/null)
    gpu=$(jq -r '.gpu // "?"' "$exp_file" 2>/dev/null)
    start=$(jq -r '.start_time // ""' "$exp_file" 2>/dev/null | cut -c1-16 | tr 'T' ' ')
    metrics=$(jq -r '.metrics // {} | to_entries | map("\(.key)=\(.value | if type=="number" then (.*1000|round/1000|tostring) else tostring end)") | join("  ")' "$exp_file" 2>/dev/null)

    case "$status" in
        completed) emoji="✅" ;;
        running)   emoji="🔄" ;;
        failed)    emoji="❌" ;;
        *)         emoji="⏸️ " ;;
    esac

    printf "%s %-30s GPU:%-2s  %s\n" "$emoji" "$name" "$gpu" "$start"
    if [ -n "$metrics" ]; then
        # Truncate long metric lines
        if [ ${#metrics} -gt 80 ]; then
            echo "     ${metrics:0:77}..."
        else
            echo "     $metrics"
        fi
    else
        no_metrics_count=$((no_metrics_count + 1))
    fi

    shown=$((shown + 1))

done <<< "$EXP_FILES"

total=$(echo "$EXP_FILES" | grep -c "exp_" 2>/dev/null || echo 0)
[ $total -gt 20 ] && echo "  ... and $((total - 20)) older experiments"

# Single hint at the bottom if any experiments lack metrics
if [ $no_metrics_count -gt 0 ]; then
    echo ""
    echo "  ($no_metrics_count experiments have no metrics — add report_metric.sh to training scripts)"
fi

# Queue summary from queue.json
if [ -f "$QUEUE_FILE" ]; then
    running=$(jq '.running | length' "$QUEUE_FILE" 2>/dev/null || echo 0)
    queued=$(jq '.queued | length' "$QUEUE_FILE" 2>/dev/null || echo 0)
    echo ""
    echo "──────────────────────────────────────────────────────────────────"
    printf "🔄 Running: %s   ⏳ Queued: %s   →  /ds:queue for details\n" "$running" "$queued"
fi
