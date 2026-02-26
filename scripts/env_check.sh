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
instructions="\n\n🔬 CORAL DS PLUGIN — MANDATORY WORKFLOW\n\nEvery time the user asks to run, train, or queue any experiment, follow these steps in order:\n\n1. INSTRUMENT THE TRAINING SCRIPT FIRST\n   1a. Add report_metric.sh calls at every eval/logging point:\n   import subprocess\n   def report(name, val): subprocess.run(['${PLUGIN_ROOT}/scripts/report_metric.sh', name, str(val)])\n   def note(text): subprocess.run(['${PLUGIN_ROOT}/scripts/report_note.sh', text])\n   1b. Fix any unconditional CUDA_VISIBLE_DEVICES override — scripts that do\n   os.environ[\"CUDA_VISIBLE_DEVICES\"] = args.device.split(\":\")[-1] will overwrite\n   the plugin's GPU assignment. Wrap with: if \"CUDA_VISIBLE_DEVICES\" not in os.environ:\n   Same for torch.cuda.set_device() calls.\n\n2. QUEUE THE EXPERIMENT\n   ${PLUGIN_ROOT}/scripts/queue_experiment.sh \"exp_name\" \"python train.py\" [gpu_mem_mb]\n   For sequential jobs on same GPU: add --after <dep_name>\n\n3. START WATCHER IF NOT RUNNING\n   ${PLUGIN_ROOT}/scripts/queue_start_watcher.sh\n\n4. TELL THE USER TO MONITOR\n   Always tell the user: use /ds:dash to see metrics, /ds:queue for queue status\n   Never queue silently without saying how to monitor."

output="$output$instructions"

# Output as JSON for additionalContext
echo "{
  \"hookSpecificOutput\": {
    \"hookEventName\": \"SessionStart\",
    \"additionalContext\": \"$output\"
  }
}"

exit 0
