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
  /ds:reload)
    # Re-inject plugin context (env state + CLAUDE.md) without touching history
    # Does NOT block — Claude sees the fresh context and acknowledges it
    PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT:-$PROJECT_DIR}"
    ENV_OUTPUT=$(bash "$PLUGIN_ROOT/scripts/env_check.sh" 2>&1 | jq -r '.hookSpecificOutput.additionalContext // empty' 2>/dev/null || true)
    CLAUDE_MD=""
    [ -f "$PLUGIN_ROOT/CLAUDE.md" ] && CLAUDE_MD=$(cat "$PLUGIN_ROOT/CLAUDE.md")
    COMBINED="[Plugin context refreshed]\n\n${ENV_OUTPUT}\n\n${CLAUDE_MD}"
    jq -n --arg ctx "$COMBINED" '{
      hookSpecificOutput: {
        hookEventName: "UserPromptSubmit",
        additionalContext: $ctx
      }
    }'
    exit 0
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
