#!/bin/bash
# List all experiment sessions

echo "Experiment Sessions:"
echo ""

for session_dir in experiments/session-*/; do
    if [ -d "$session_dir" ]; then
        SESSION_ID=$(basename "$session_dir" | sed 's/session-//')
        EXP_COUNT=$(ls "$session_dir"/exp_*.json 2>/dev/null | wc -l)
        RUNNING=$(grep -l '"status": "running"' "$session_dir"/exp_*.json 2>/dev/null | wc -l)
        COMPLETED=$(grep -l '"status": "completed"' "$session_dir"/exp_*.json 2>/dev/null | wc -l)

        # Get latest experiment time
        LATEST=$(ls -t "$session_dir"/exp_*.json 2>/dev/null | head -1)
        if [ -n "$LATEST" ]; then
            LAST_TIME=$(jq -r '.start_time' "$LATEST" 2>/dev/null | cut -d'T' -f1)
        else
            LAST_TIME="N/A"
        fi

        echo "Session: $SESSION_ID"
        echo "  Experiments: $EXP_COUNT ($RUNNING running, $COMPLETED completed)"
        echo "  Last activity: $LAST_TIME"
        echo ""
    fi
done

# Check for old non-session experiments
OLD_EXPS=$(ls experiments/exp_*.json 2>/dev/null | wc -l)
if [ "$OLD_EXPS" -gt 0 ]; then
    echo "Legacy (no session): $OLD_EXPS experiments"
    echo ""
fi
