#!/bin/bash
# Background training monitor - watches for completion
# Triggered async after Bash commands that start training

# Read hook input to get the command that was run
input=$(cat)
cmd=$(echo "$input" | jq -r '.tool_input.command // empty' 2>/dev/null)

# Only monitor if this looks like a training command
if [[ ! "$cmd" =~ train|experiment|rl|sft ]]; then
    exit 0
fi

# Extract likely log file from command
log_file=""
if [[ "$cmd" =~ \>\>?[[:space:]]*([^[:space:]]+\.log) ]]; then
    log_file="${BASH_REMATCH[1]}"
elif [[ "$cmd" =~ --log[_-]file[[:space:]]+([^[:space:]]+) ]]; then
    log_file="$1"
fi

# Default monitoring strategy
# Check every 5 minutes for up to 4 hours (training timeout)
sleep_interval=300
max_iterations=48
iteration=0

while [ $iteration -lt $max_iterations ]; do
    sleep $sleep_interval
    iteration=$((iteration + 1))

    # Check if training process still running
    if ! ps aux | grep -E "python.*(train|experiment)" | grep -v grep > /dev/null; then
        # Training finished

        # Look for evaluation results
        result_file=""
        for f in results/*.json outputs/eval*.json results_*.json; do
            if [ -f "$f" ] && [ "$(stat -c %Y "$f" 2>/dev/null || stat -f %m "$f" 2>/dev/null)" -gt "$(($(date +%s) - 600))" ]; then
                result_file="$f"
                break
            fi
        done

        if [ -n "$result_file" ]; then
            # Extract key metrics
            test_pass=$(jq -r '.test_pass_pct // .accuracy // .score // "N/A"' "$result_file" 2>/dev/null)

            echo "{
              \"systemMessage\": \"✅ Training complete!\\nResults: $test_pass% test pass\\nFile: $result_file\\n\\nUpdate RESULTS_TABLE.md and commit results.\"
            }"
        else
            echo "{
              \"systemMessage\": \"⚠️  Training process finished but no results file found.\\nCheck logs and verify experiment completed successfully.\"
            }"
        fi

        exit 0
    fi

    # Check log file for progress (if we found one)
    if [ -n "$log_file" ] && [ -f "$log_file" ]; then
        # Look for completion markers
        if tail -100 "$log_file" | grep -qiE "(training complete|evaluation done|experiment finished)"; then
            echo "{
              \"systemMessage\": \"📊 Training appears complete (log marker found). Check results and update table.\"
            }"
            exit 0
        fi

        # Extract latest loss/step if visible
        latest_progress=$(tail -20 "$log_file" | grep -oE "(step|epoch)[[:space:]]*[0-9]+" | tail -1)
        if [ -n "$latest_progress" ]; then
            # Silent progress tracking - don't spam
            :
        fi
    fi
done

# Timeout after 4 hours
echo "{
  \"systemMessage\": \"⏰ Training monitor timeout (4h). Process may still be running. Check manually with /dash or ps aux.\"
}"

exit 0
