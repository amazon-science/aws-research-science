#!/bin/bash
# Start a new experiment (creates JSON file)

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <experiment_name> <description> [gpu_id] [pid]"
    echo "Example: $0 lora_rank6 'LoRA with rank 6' 2 12345"
    exit 1
fi

EXP_NAME="$1"
DESCRIPTION="$2"
GPU_ID="${3:-null}"
PID="${4:-null}"
SESSION_ID="${5:-}"

# Get session ID from .claude_session file if not provided
if [ -z "$SESSION_ID" ] && [ -f .claude_session ]; then
    SESSION_ID=$(cat .claude_session)
fi

# Create session-specific experiments directory
if [ -n "$SESSION_ID" ]; then
    EXP_DIR="experiments/session-${SESSION_ID}"
else
    EXP_DIR="experiments"
fi
mkdir -p "$EXP_DIR"

# Generate timestamp and filename
TIMESTAMP=$(date -Iseconds)
DATE=$(date +%Y-%m-%d-%H-%M-%S)
FILENAME="$EXP_DIR/exp_${EXP_NAME}_${DATE}.json"

# Create experiment JSON file
cat > "$FILENAME" << EOF
{
  "name": "$EXP_NAME",
  "description": "$DESCRIPTION",
  "status": "running",
  "start_time": "$TIMESTAMP",
  "end_time": null,
  "params": {},
  "metrics": {},
  "gpu": $GPU_ID,
  "pid": $PID,
  "session_id": "${SESSION_ID:-unknown}",
  "file": "$FILENAME"
}
EOF

echo "$FILENAME"
