#!/bin/bash
# Report a metric to the most recent running experiment

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <metric_name> <value> [experiment_file]"
    echo "Example: $0 loss 0.45"
    echo "Example: $0 accuracy 0.89 experiments/exp_lora_rank6_2026-02-05-19-30.json"
    exit 1
fi

METRIC_NAME="$1"
VALUE="$2"
EXP_FILE="$3"

# If no experiment file specified, find the most recent running experiment
if [ -z "$EXP_FILE" ]; then
    EXP_FILE=$(ls -t experiments/exp_*.json 2>/dev/null | head -1)
    if [ -z "$EXP_FILE" ]; then
        echo "Error: No experiment found. Start an experiment first."
        exit 1
    fi
fi

# Check if file exists
if [ ! -f "$EXP_FILE" ]; then
    echo "Error: Experiment file not found: $EXP_FILE"
    exit 1
fi

# Update metrics using jq
TEMP_FILE=$(mktemp)
jq ".metrics.\"$METRIC_NAME\" = $VALUE" "$EXP_FILE" > "$TEMP_FILE"
mv "$TEMP_FILE" "$EXP_FILE"

echo "✓ Reported $METRIC_NAME=$VALUE to $EXP_FILE"
