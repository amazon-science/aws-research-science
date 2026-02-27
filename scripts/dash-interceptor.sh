#!/bin/bash
# Intercepts /ds:* commands and runs scripts directly (no Claude API call)

# Read JSON input from stdin
INPUT=$(cat)
PROMPT=$(echo "$INPUT" | jq -r '.prompt // empty')

# Resolve directories
# CLAUDE_PLUGIN_ROOT: set by Claude Code to plugin install path
# CLAUDE_PROJECT_DIR: set by Claude Code to user's project directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$PWD}"

# Strip whitespace from prompt
TRIMMED=$(echo "$PROMPT" | xargs)

# Route /ds:* commands to their scripts
case "$TRIMMED" in
  /ds:dash)
    if python3 -c "import rich" 2>/dev/null; then
      COLS=$(tput cols 2>/dev/null || echo 100)
      OUTPUT=$(cd "$PROJECT_DIR" && COLUMNS=$COLS python3 "$PLUGIN_ROOT/scripts/dashboard.py" --once --all 2>&1)
    else
      OUTPUT=$("$PLUGIN_ROOT/scripts/list_experiments.sh" 2>&1)
    fi
    ;;
  /ds:queue)
    OUTPUT=$("$PLUGIN_ROOT/scripts/queue_status.sh" 2>&1)
    ;;
  /ds:dash-sessions)
    OUTPUT=$("$PLUGIN_ROOT/scripts/list_sessions.sh" 2>&1)
    ;;
  /ds:dash-clear)
    OUTPUT=$("$PLUGIN_ROOT/scripts/clear_experiments.sh" 2>&1)
    ;;
  /ds:dash-all)
    if python3 -c "import rich" 2>/dev/null; then
      COLS=$(tput cols 2>/dev/null || echo 100)
      OUTPUT=$(cd "$PROJECT_DIR" && COLUMNS=$COLS python3 "$PLUGIN_ROOT/scripts/dashboard.py" --once --all 2>&1)
    else
      OUTPUT=$("$PLUGIN_ROOT/scripts/list_experiments.sh" 2>&1)
      OUTPUT+=$'\n\n'
      OUTPUT+=$("$PLUGIN_ROOT/scripts/list_sessions.sh" 2>&1)
    fi
    ;;
  /ds:reload)
    # Re-inject plugin context (env state + CLAUDE.md) without touching history
    # Does NOT block — Claude sees the fresh context and acknowledges it
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
