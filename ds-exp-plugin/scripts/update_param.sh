#!/bin/bash
# Update a parameter in an experiment

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <param_name> <value> [experiment_file]"
    echo "Example: $0 rank 6"
    echo "Example: $0 learning_rate 0.001 experiments/exp_lora_rank6_2026-02-05-19-30.json"
    exit 1
fi

PARAM_NAME="$1"
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

# Determine if value is a number or string
if [[ "$VALUE" =~ ^[0-9]+\.?[0-9]*$ ]]; then
    # Numeric value
    TEMP_FILE=$(mktemp)
    jq ".params.\"$PARAM_NAME\" = $VALUE" "$EXP_FILE" > "$TEMP_FILE"
    mv "$TEMP_FILE" "$EXP_FILE"
else
    # String value
    TEMP_FILE=$(mktemp)
    jq ".params.\"$PARAM_NAME\" = \"$VALUE\"" "$EXP_FILE" > "$TEMP_FILE"
    mv "$TEMP_FILE" "$EXP_FILE"
fi

echo "✓ Updated param $PARAM_NAME=$VALUE in $EXP_FILE"
