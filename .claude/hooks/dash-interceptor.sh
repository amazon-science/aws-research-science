#!/bin/bash
# Intercepts /ds:dash command and runs dashboard directly

# Read JSON input from stdin
INPUT=$(cat)
PROMPT=$(echo "$INPUT" | jq -r '.prompt // empty')

# Check if prompt is exactly "/ds:dash" (with optional whitespace)
if [[ "$PROMPT" =~ ^[[:space:]]*/ds:dash[[:space:]]*$ ]]; then
  # Get the script directory
  SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
  PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

  # Run the dashboard script (using list_experiments.sh since it has no dependencies)
  DASHBOARD_OUTPUT=$("$PROJECT_DIR/scripts/list_experiments.sh" 2>&1)

  # Return the output as additional context and block the original prompt
  jq -n --arg output "$DASHBOARD_OUTPUT" '{
    decision: "block",
    reason: "Dashboard displayed by hook",
    hookSpecificOutput: {
      hookEventName: "UserPromptSubmit",
      additionalContext: $output
    }
  }'
  exit 0
fi

# For any other prompt, allow it to proceed normally
exit 0
