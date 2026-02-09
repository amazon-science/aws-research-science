#!/bin/bash
# Clear completed experiments from current session

set -e

# Get current session ID
SESSION_ID=""
if [ -f .claude_session ]; then
    SESSION_ID=$(cat .claude_session)
fi

if [ -z "$SESSION_ID" ]; then
    echo "No session ID found. Run from a Claude Code session."
    exit 1
fi

# Find session directory
SESSION_DIR="experiments/session-${SESSION_ID}"
if [ ! -d "$SESSION_DIR" ]; then
    echo "No experiments found for current session."
    exit 0
fi

# Create cleared directory
CLEARED_DIR="$SESSION_DIR/.cleared/$(date +%Y-%m-%d-%H-%M-%S)"
mkdir -p "$CLEARED_DIR"

# Move completed experiments
MOVED=0
for exp_file in "$SESSION_DIR"/exp_*.json; do
    if [ -f "$exp_file" ]; then
        STATUS=$(jq -r '.status' "$exp_file" 2>/dev/null || echo "unknown")
        if [ "$STATUS" = "completed" ]; then
            mv "$exp_file" "$CLEARED_DIR/"
            ((MOVED++))
        fi
    fi
done

echo "✓ Cleared $MOVED completed experiment(s)"
echo "  Moved to: $CLEARED_DIR"
