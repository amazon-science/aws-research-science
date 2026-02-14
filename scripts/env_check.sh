#!/bin/bash
# Environment check on startup - shows GPU status, disk, git
# Outputs as additionalContext for Claude

# Read JSON from stdin to get session ID
if [ -t 0 ]; then
    # Running standalone, no input
    output=""
else
    # Get input and extract session ID
    input=$(cat)
    SESSION_ID=$(echo "$input" | jq -r '.session_id // ""' 2>/dev/null)
    if [ -n "$SESSION_ID" ]; then
        echo "$SESSION_ID" > .claude_session
    fi
fi

# Resolve plugin root — CLAUDE_PLUGIN_ROOT is set by Claude Code when running plugin hooks
# Fall back to script's parent directory for local development
PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

output=""

# GPU status - find idle GPUs
if command -v nvidia-smi &> /dev/null; then
    idle_gpus=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.free --format=csv,noheader,nounits 2>/dev/null | \
        awk -F',' '$2 < 10 && $3 > 10000 {printf "GPU %s: IDLE (%s MB free)\n", $1, $3}')

    if [ -n "$idle_gpus" ]; then
        output="$output\n🔥 Available GPUs:\n$idle_gpus"
    else
        output="$output\n⚠️  No idle GPUs (all busy)"
    fi
fi

# Experiment tracking status
if [ -d experiments ]; then
    exp_count=$(ls experiments/exp_*.json 2>/dev/null | wc -l)
    running_count=$(grep -l '"status": "running"' experiments/exp_*.json 2>/dev/null | wc -l)
    if [ "$exp_count" -gt 0 ]; then
        output="$output\n📊 Experiments: $exp_count total, $running_count running"
    else
        output="$output\n📊 Tracking initialized (no experiments yet)"
    fi
else
    output="$output\n⚠️  No tracking (run: ${PLUGIN_ROOT}/scripts/init_tracking.sh)"
fi

# Running processes
training_count=$(ps aux | grep -E "python.*(train|experiment)" | grep -v grep | wc -l)
if [ "$training_count" -gt 0 ]; then
    output="$output\n⚙️  $training_count training process(es) running"
fi

# Disk
disk_usage=$(df -h . | tail -1 | awk '{print $5}' | tr -d '%')
if [ "$disk_usage" -gt 90 ]; then
    output="$output\n⚠️  Disk: ${disk_usage}% used (running low!)"
fi

# Add experiment tracking protocol
instructions="\n\n🔬 CORAL DS PLUGIN: Automatic Experiment Tracking\n\nWhen user requests training/ML tasks, automatically track and queue experiments:\n\n**PREFERRED: Use Job Queue (auto-launches when GPU free)**\n${PLUGIN_ROOT}/scripts/queue_experiment.sh \"exp_name\" \"python train.py\" [gpu_mem_mb]\n\nQueue system:\n- Checks GPU availability\n- Launches immediately if GPU free\n- Queues and auto-launches later if GPUs busy\n- Retries failed launches (up to 3x)\n- Start watcher: ${PLUGIN_ROOT}/scripts/queue_start_watcher.sh\n- Check status: ${PLUGIN_ROOT}/scripts/queue_status.sh\n\n**Manual tracking (if not using queue):**\n1. Start: EXP_FILE=\$(${PLUGIN_ROOT}/scripts/start_experiment.sh \"name\" \"desc\" [gpu])\n2. In training code:\n   import subprocess\n   def report(name, val): subprocess.run(['${PLUGIN_ROOT}/scripts/report_metric.sh', name, str(val)])\n   def note(text): subprocess.run(['${PLUGIN_ROOT}/scripts/report_note.sh', text])\n3. Complete: ${PLUGIN_ROOT}/scripts/complete_experiment.sh completed \$EXP_FILE\n\nView: /ds:dash"

output="$output$instructions"

# Output as JSON for additionalContext
echo "{
  \"hookSpecificOutput\": {
    \"hookEventName\": \"SessionStart\",
    \"additionalContext\": \"$output\"
  }
}"

exit 0
