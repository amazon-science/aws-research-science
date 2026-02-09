# CORAL DS: ML Experiment Tracking Plugin for Claude Code

**Zero-overhead ML experiment tracking with intelligent job queuing**

Automatic experiment tracking, GPU monitoring, smart job queuing, and live dashboard. Just talk naturally - tracking and queuing happen automatically.

## Features

### 🔬 Automatic Experiment Tracking
Just say "train a model with LoRA rank 6" and Claude automatically:
- Creates experiment JSON file
- Adds metric reporting to training code
- Tracks parameters and metrics
- Shows results in dashboard

**No extra commands needed!**

### 🔄 Intelligent Job Queue (NEW)
Say "try rank 4, 8, and 16" and Claude automatically:
- Checks GPU availability
- Launches immediately if GPU free
- Queues jobs when GPUs busy and auto-launches when free
- Retries failed launches (OOM, device errors) up to 3 times
- Tracks all jobs in unified queue

**No babysitting required - queue them all at once!**

### 📊 Live Dashboard (`/ds:dash`)
One command to see everything:
```
🔥 ML Experiment Dashboard  |  19:16:04  |  📊 1 exp (✅0 ❌0 🔄1)  |  🔄 Queue: 1 running, 2 waiting

Job         Status         Info
exp_rank4   ✓ GPU 0       Started 2026-02-06T20:34
exp_rank8   ⏳ Waiting     Queued
exp_rank16  ⏳ Waiting     Queued

GPU  Name         Util  Memory    Temp  St
0    NVIDIA A10G  0%    0GB/22GB  20°C  ✓ (green)
1    NVIDIA A10G  55%   8GB/22GB  45°C  ~ (orange)
2    NVIDIA A10G  98%   12GB/22GB 52°C  ! (red)

PID     CPU%  Mem%  Command
328233  100%  1.1%  python train_superlora_comparison.py...

Experiment      Status  Metric    Value  Time
test_lora_rank6 🔄      accuracy  0.89   5m ago
```

### 📊 Smart Status Line
Color-coded GPU status at the bottom:
```
[Sonnet 4.5 | 90k/200k (45%)] 📁 dir | GPU# (mem/util) |0. 0%/0% |1. 55%/45% |2. 98%/52% ┃Proc:  3┃
```
- Green: <30% (available)
- Orange: 30-70% (moderate)
- Red: >70% (busy)

### 🔥 GPU Status on Startup
Shows available GPUs when you start:
```
🔥 Available GPUs:
GPU 0: IDLE (22GB free)
GPU 3: IDLE (22GB free)
📊 Experiments: 1 total, 1 running
```

## Installation

### Option 1: Install via Plugin Marketplace (Recommended)

```bash
# Add the marketplace
/plugin marketplace add amazon-science/aws-research-science#plugins

# Install the plugin
/plugin install ds@coral
```

### Option 2: Direct Plugin Directory (Development)

```bash
# Clone the repository
git clone --branch plugins https://github.com/amazon-science/aws-research-science.git coral-ds-plugin

# Start Claude with the plugin
claude --plugin-dir /path/to/coral-ds-plugin
```

### Configuration

#### Enable status line (optional)
Add to your `.claude/settings.json`:
```json
{
  "statusLine": {
    "type": "command",
    "command": "${CLAUDE_PLUGIN_ROOT}/scripts/statusline.sh",
    "padding": 0
  }
}
```

Or copy the example:
```bash
# View the example config
cat coral-ds-plugin/settings.example.json

# Manually add to your settings
```

#### Start queue watcher (recommended)
For automatic job queuing:
```bash
cd /path/to/coral-ds-plugin
./scripts/queue_start_watcher.sh
```

#### Enable output style (optional)
For thoughtful, scientist-mode responses:
```bash
/output-style ds:Precise
```

## Commands

### `/ds:dash` - Dashboard
Shows GPU status, queue, running processes, and experiments in a compact view.

**Usage:**
```bash
/ds:dash
```

**Output:** Compact, single-screen dashboard with color-coded status and queue section.

### `/ds:queue` - Queue Status
Shows current job queue status.

**Usage:**
```bash
/ds:queue
```

**Output:** Running jobs, queued jobs, and watcher daemon status.

### `/ds:dash-all` - All Sessions
View experiments from all Claude Code sessions.

### `/ds:dash-sessions` - List Sessions
See all experiment sessions.

### `/ds:dash-clear` - Clear Completed
Remove completed experiments from current session.

## Status Line

The status line shows at the bottom of Claude Code:
- **Model name** - Current Claude model (Opus, Sonnet, Haiku)
- **Directory** - Current working directory
- **GPU status** - Idle/busy GPU count
- **Training jobs** - Number of active training processes

**Example:**
```
[Sonnet] 📁 superweights | 🔥 2 idle, 2 busy | ⚙️ 2 training
```

Updates automatically as you work.

## What This Plugin Does NOT Do

❌ **No auto-commits** - Git is manual (or use `/background git commit ...`)
❌ **No hook noise** - Only runs on startup
❌ **No background monitoring** - Use `/background` when you need it
❌ **No Stop hooks** - Claude exits normally

**Philosophy:** Provide tools when you need them, stay invisible otherwise.

## File Structure

```
coral-ds-plugin/
├── .claude-plugin/
│   └── plugin.json              # Plugin manifest
├── commands/
│   ├── dash.md                  # Dashboard command
│   ├── dash-all.md              # All sessions dashboard
│   ├── dash-sessions.md         # List sessions
│   ├── dash-clear.md            # Clear completed
│   └── queue.md                 # Queue status command
├── hooks/
│   └── hooks.json               # SessionStart hook
├── output-styles/
│   └── precise.md               # Scientist mode output style
├── scripts/
│   ├── env_check.sh             # Startup GPU check + experiment tracking protocol
│   ├── dashboard.py             # Dashboard UI with queue section
│   ├── statusline.sh            # Status line generator
│   ├── start_experiment.sh      # Initialize experiment
│   ├── report_metric.sh         # Report metrics
│   ├── report_note.sh           # Add notes
│   ├── update_param.sh          # Track parameters
│   ├── complete_experiment.sh   # Mark experiment done
│   ├── queue_experiment.sh      # Queue job for execution
│   ├── queue_watcher.sh         # Background queue daemon
│   ├── queue_start_watcher.sh   # Start queue watcher
│   ├── queue_stop_watcher.sh    # Stop queue watcher
│   └── queue_status.sh          # Show queue status
├── experiments/                  # Experiment data (created at runtime)
│   ├── session-{id}/            # Session-specific experiments
│   │   └── exp_*.json           # Experiment JSON files
│   ├── queue.json               # Job queue state
│   └── queue_watcher.log        # Queue watcher logs
├── settings.example.json        # Status line config
└── README.md
```

## Scripts

All scripts are inside the plugin at `scripts/`:

**Core Tracking:**
- **start_experiment.sh** - Initialize experiment JSON
- **report_metric.sh** - Report metrics from training code
- **report_note.sh** - Add timestamped notes/observations
- **update_param.sh** - Track hyperparameters
- **complete_experiment.sh** - Mark experiment as completed/failed

**Job Queue:**
- **queue_experiment.sh** - Queue job with auto-launch
- **queue_watcher.sh** - Background daemon (monitors queue every 30s)
- **queue_start_watcher.sh** - Start the watcher daemon
- **queue_stop_watcher.sh** - Stop the watcher daemon
- **queue_status.sh** - Show queue status

**Display:**
- **env_check.sh** - Shows GPU/experiment status on startup
- **dashboard.py** - Compact TUI dashboard with Rich (includes queue section)
- **statusline.sh** - Generates status line with GPU info

## Customization

### Status Line
Edit `scripts/statusline.sh` to customize what appears in the status bar.

### Dashboard
Edit `scripts/dashboard.py` to change table layout, colors, or add sections.

### Startup Check
Edit `scripts/env_check.sh` to change what's shown on startup.

## Requirements

- **nvidia-smi** - For GPU monitoring
- **Python 3.6+** with `rich` - For dashboard
- **jq** - For JSON parsing in status line
- **Git** - For branch info (optional)

Install requirements:
```bash
pip install rich
```

## Usage Examples

### Queue Multiple Experiments
```bash
# Claude automatically uses queue_experiment.sh when you say:
"Try LoRA with rank 4, 8, and 16"

# Or manually:
./scripts/queue_experiment.sh "exp_rank4" "python train.py --rank 4" 6000
./scripts/queue_experiment.sh "exp_rank8" "python train.py --rank 8" 8000
./scripts/queue_experiment.sh "exp_rank16" "python train.py --rank 16" 12000
```

### Track Metrics in Training Code
```python
import subprocess

def report(name, val):
    subprocess.run(['./scripts/report_metric.sh', name, str(val)])

def note(text):
    subprocess.run(['./scripts/report_note.sh', text])

# During training
for epoch in range(epochs):
    loss = train_epoch()
    acc = evaluate()

    report('loss', loss)
    report('accuracy', acc)

    if loss_plateau_detected():
        note('Loss plateau at epoch {epoch}, trying LR adjustment')
```

### Monitor Progress
```bash
# View dashboard
/ds:dash

# Check queue
/ds:queue

# Watch experiment files
tail -f experiments/session-*/exp_*.json
```

## Troubleshooting

### Queue watcher not starting
```bash
# Check if already running
ps aux | grep queue_watcher

# View logs
tail -f experiments/queue_watcher.log

# Restart
./scripts/queue_stop_watcher.sh
./scripts/queue_start_watcher.sh
```

### Experiments not being tracked
```bash
# Verify EXP_FILE environment variable is set in training process
echo $EXP_FILE

# Check experiment files exist
ls -la experiments/session-*/

# Make scripts executable
chmod +x scripts/*.sh
```

### Status line not appearing
```bash
# Test manually
./scripts/statusline.sh

# Check jq is installed
which jq
```

### Dashboard not showing
```bash
# Install Rich
pip install rich

# Test nvidia-smi
nvidia-smi

# Run dashboard directly
python scripts/dashboard.py --once
```

## Key Features Summary

✅ **Automatic tracking** - No manual logging needed
✅ **Smart job queue** - Auto-launches when GPUs free
✅ **Retry logic** - Handles transient failures automatically
✅ **Session isolation** - Clean separation by Claude Code session
✅ **Live dashboard** - Real-time view of GPUs, queue, experiments
✅ **Notes API** - Add observations during training
✅ **Color-coded GPU status** - Green/orange/red indicators
✅ **Scientist mode** - Optional thoughtful output style

## Architecture

**Data Storage:**
- Experiments: `experiments/session-{id}/exp_*.json`
- Queue state: `experiments/queue.json`
- Session ID: `.claude_session`

**Queue System:**
- Checks GPU availability (<10% util, >needed memory)
- Fast failure (<60s): Retry up to 3x (likely resource issue)
- Mid-run failure (>60s): No retry (likely code bug)
- Background daemon runs every 30s

**Integration:**
- SessionStart hook provides Claude with tracking protocol
- Claude automatically uses `queue_experiment.sh` for training requests
- Manual usage also supported via scripts

## Version

**2.0.0** - Job queue system with smart retry logic

## Author

Amazon CORAL Lab

## Keywords

ml, experiment-tracking, gpu-monitoring, job-queue, automatic-retry, claude-code-plugin
