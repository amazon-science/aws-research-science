#!/bin/bash
# DS Experiment Plugin - Status Line
# Shows: Model | GPU utilization & memory | Process count

# Read JSON input from stdin
input=$(cat)

# Extract Claude Code context
MODEL=$(echo "$input" | jq -r '.model.display_name // "Claude"')
CURRENT_DIR=$(echo "$input" | jq -r '.workspace.current_dir // "~"')

# Extract token usage
TOKENS_IN=$(echo "$input" | jq -r '.context_window.total_input_tokens // 0')
TOKENS_OUT=$(echo "$input" | jq -r '.context_window.total_output_tokens // 0')
TOKENS_LIMIT=$(echo "$input" | jq -r '.context_window.context_window_size // 0')
USED_PCT=$(echo "$input" | jq -r '.context_window.used_percentage // 0')

# Format model info with token percentage
MODEL_INFO="$MODEL"
if [ "$TOKENS_LIMIT" -gt 0 ]; then
    TOKENS_TOTAL=$((TOKENS_IN + TOKENS_OUT))
    TOKENS_K=$((TOKENS_TOTAL / 1000))
    LIMIT_K=$((TOKENS_LIMIT / 1000))
    USED_PCT=$((TOKENS_TOTAL * 100 / TOKENS_LIMIT))
    MODEL_INFO="$MODEL_INFO | ${TOKENS_K}k/${LIMIT_K}k (${USED_PCT}%)"
fi

# Get GPU info (quick check, timeout 1s)
GPU_STATUS=""
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(timeout 1 nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null)
    if [ $? -eq 0 ] && [ -n "$GPU_INFO" ]; then
        GPU_PARTS=()

        while IFS=',' read -r idx util mem_used mem_total; do
            idx=$(echo "$idx" | xargs)
            util=$(echo "$util" | xargs)
            mem_used=$(echo "$mem_used" | xargs)
            mem_total=$(echo "$mem_total" | xargs)

            # Calculate memory percentage
            if [ "$mem_total" -gt 0 ]; then
                mem_pct=$((mem_used * 100 / mem_total))
            else
                mem_pct=0
            fi

            # Color codes: green (<30%), orange (30-70%), red (>70%)
            # Use the higher of memory or utilization for color
            max_pct=$util
            if [ "$mem_pct" -gt "$util" ]; then
                max_pct=$mem_pct
            fi

            COLOR_RESET="\033[0m"
            if [ "$max_pct" -lt 30 ]; then
                COLOR="\033[32m"  # Green (idle)
            elif [ "$max_pct" -le 70 ]; then
                COLOR="\033[33m"  # Orange/Yellow (moderate)
            else
                COLOR="\033[31m"  # Red (busy)
            fi

            # Format: |0. 045%/078% — fixed width (3 digits each) prevents
            # stale characters bleeding through when values shrink between redraws
            GPU_PARTS+=("$(printf "${COLOR}|${idx}. %3d%%/%3d%%${COLOR_RESET}" "$mem_pct" "$util")")
        done <<< "$GPU_INFO"

        # Join all GPUs with space
        GPU_STATUS="GPU# (mem/util) $(IFS=' '; echo -e "${GPU_PARTS[*]}")"
    fi
fi

# Count all background Python processes
PROC_COUNT=0
if command -v ps &> /dev/null; then
    PROC_COUNT=$(ps aux 2>/dev/null | grep -E 'python' | grep -v grep | grep -v 'jupyter' | grep -v 'dashboard.py' | grep -v 'statusline.sh' | wc -l)
fi

# Always show process count with consistent width (3 digits padded) in a box
# Using box drawing characters: ┃ for vertical bars
# Pad to 3 digits to handle up to 999 processes
PROC_STATUS=$(printf "\033[36m┃Proc:%3d┃\033[0m" "$PROC_COUNT")

# Combine status parts
STATUS_PARTS=()
[ -n "$GPU_STATUS" ] && STATUS_PARTS+=("$GPU_STATUS")
STATUS_PARTS+=("$PROC_STATUS")

# Build final status line
DIR_NAME=$(basename "$CURRENT_DIR")

# Colors: Orange for model bar, Yellow for folder
ORANGE="\033[38;5;208m"  # Claude Code orange
YELLOW="\033[33m"
COLOR_RESET="\033[0m"

if [ ${#STATUS_PARTS[@]} -gt 0 ]; then
    STATUS=$(IFS=' | '; echo -e "${STATUS_PARTS[*]}")
    echo -e "${ORANGE}[$MODEL_INFO]${COLOR_RESET} ${YELLOW}📁 $DIR_NAME${COLOR_RESET} | $STATUS"
else
    echo -e "${ORANGE}[$MODEL_INFO]${COLOR_RESET} ${YELLOW}📁 $DIR_NAME${COLOR_RESET}"
fi
