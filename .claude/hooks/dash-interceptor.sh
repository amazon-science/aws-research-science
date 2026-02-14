#!/bin/bash
# Intercepts /ds:* commands and runs scripts directly (no Claude API call)

# Read JSON input from stdin
INPUT=$(cat)
PROMPT=$(echo "$INPUT" | jq -r '.prompt // empty')

# Resolve project directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

# Strip whitespace from prompt
TRIMMED=$(echo "$PROMPT" | xargs)

# Route /ds:* commands to their scripts
case "$TRIMMED" in
  /ds:dash)
    OUTPUT=$("$PROJECT_DIR/scripts/list_experiments.sh" 2>&1)
    ;;
  /ds:queue)
    OUTPUT=$("$PROJECT_DIR/scripts/queue_status.sh" 2>&1)
    ;;
  /ds:dash-sessions)
    OUTPUT=$("$PROJECT_DIR/scripts/list_sessions.sh" 2>&1)
    ;;
  /ds:dash-clear)
    OUTPUT=$("$PROJECT_DIR/scripts/clear_experiments.sh" 2>&1)
    ;;
  /ds:dash-all)
    # Combine experiments + sessions for a full view
    OUTPUT=$("$PROJECT_DIR/scripts/list_experiments.sh" 2>&1)
    OUTPUT+=$'\n\n'
    OUTPUT+=$("$PROJECT_DIR/scripts/list_sessions.sh" 2>&1)
    ;;
  *)
    # Not a /ds: command — allow normal processing
    exit 0
    ;;
esac

# Stop Claude entirely and show output to user
jq -n --arg output "$OUTPUT" '{
  continue: false,
  stopReason: $output
}'
exit 0
