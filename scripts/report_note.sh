#!/bin/bash
# Add a timestamped note to an experiment

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <note_text> [experiment_file]"
    echo "Example: $0 'Loss spiked at step 450, investigating'"
    exit 1
fi

NOTE_TEXT="$1"
EXP_FILE="$2"

# If no experiment file specified, find the most recent running experiment
if [ -z "$EXP_FILE" ]; then
    # Check session directory first
    if [ -f .claude_session ]; then
        SESSION_ID=$(cat .claude_session)
        EXP_FILE=$(ls -t experiments/session-${SESSION_ID}/exp_*.json 2>/dev/null | head -1)
    fi

    # Fall back to legacy experiments
    if [ -z "$EXP_FILE" ]; then
        EXP_FILE=$(ls -t experiments/exp_*.json 2>/dev/null | head -1)
    fi

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

# Add timestamped note to notes array
TIMESTAMP=$(date -Iseconds)
TEMP_FILE=$(mktemp)

# Initialize notes array if it doesn't exist, then append
jq ".notes = (.notes // []) + [{\"time\": \"$TIMESTAMP\", \"text\": \"$NOTE_TEXT\"}]" "$EXP_FILE" > "$TEMP_FILE"
mv "$TEMP_FILE" "$EXP_FILE"

echo "✓ Added note to $EXP_FILE"
echo "  [$TIMESTAMP] $NOTE_TEXT"
