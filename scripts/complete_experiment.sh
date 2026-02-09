#!/bin/bash
# Mark an experiment as completed or failed

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <status> [experiment_file]"
    echo "Status: completed | failed"
    echo "Example: $0 completed"
    echo "Example: $0 failed experiments/exp_lora_rank6_2026-02-05-19-30.json"
    exit 1
fi

STATUS="$1"
EXP_FILE="$2"

# Validate status
if [ "$STATUS" != "completed" ] && [ "$STATUS" != "failed" ]; then
    echo "Error: Status must be 'completed' or 'failed'"
    exit 1
fi

# If no experiment file specified, find the most recent running experiment
if [ -z "$EXP_FILE" ]; then
    EXP_FILE=$(ls -t experiments/exp_*.json 2>/dev/null | head -1)
    if [ -z "$EXP_FILE" ]; then
        echo "Error: No experiment found."
        exit 1
    fi
fi

# Check if file exists
if [ ! -f "$EXP_FILE" ]; then
    echo "Error: Experiment file not found: $EXP_FILE"
    exit 1
fi

# Update status and end time
TIMESTAMP=$(date -Iseconds)
TEMP_FILE=$(mktemp)
jq ".status = \"$STATUS\" | .end_time = \"$TIMESTAMP\"" "$EXP_FILE" > "$TEMP_FILE"
mv "$TEMP_FILE" "$EXP_FILE"

echo "✓ Experiment marked as $STATUS: $EXP_FILE"
