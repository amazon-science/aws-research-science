# CORAL: Data Science Plugin Marketplace for Claude Code

**A collection of data science plugins for ML experimentation and visualization**

This marketplace provides plugins for ML experiment tracking, GPU monitoring, job queuing, and publication-quality diagram generation.

## Plugins

### 📊 DS Plugin - ML Experiment Tracking
**Zero-overhead ML experiment tracking with intelligent job queuing**

Automatic experiment tracking, GPU monitoring, smart job queuing, and live dashboard. Just talk naturally - tracking and queuing happen automatically.

**Install:** `/plugin install ds@coral`

### 📈 Plot Plugin - Publication-Quality Diagrams
**Generate publication-quality diagrams with iterative refinement**

Create LaTeX/TikZ diagrams (neural networks, flowcharts, architecture diagrams) with automatic iterative refinement until publication-ready.

**Install:** `/plugin install plot@coral`
**Docs:** [plot-plugin/README.md](plot-plugin/README.md)

---

## Quick Start

**1. Add the marketplace:**
```bash
/plugin marketplace add amazon-science/aws-research-science#plugins
```

**2. Install plugins:**
```bash
/plugin install ds@coral     # ML experiment tracking
```

Or...

```bash
/plugin install plot@coral   # Diagram generation 
```

**3. Configure DS plugin (recommended):**
```bash
/ds:statusline-setup         # Enable GPU status line at bottom
/output-style ds:Precise     # Enable thoughtful scientist mode
```

Then restart Claude Code to activate the status line.

---

# DS Plugin: ML Experiment Tracking

The rest of this README covers the DS plugin in detail.

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

## DS Plugin Quick Start

Get up and running in 2 minutes:

### 1. Install the DS Plugin

Start Claude Code in your project directory:
```bash
cd your-ml-project
claude .
```

Inside Claude, add the marketplace and install the DS plugin:
```bash
/plugin marketplace add amazon-science/aws-research-science#plugins
/plugin install ds@coral
```

### 2. Set Up Status Line (Optional but Recommended)

Enable GPU monitoring at the bottom of your screen:
```bash
/ds:statusline-setup
```

Then restart Claude Code to see live GPU status.

### 3. Enable Precise Mode (Optional)

For more thoughtful, scientifically rigorous responses:
```bash
/output-style ds:Precise
```

### 4. Start Using It

Just talk naturally! Say:
- "Train a model with LoRA rank 8"
- "Try batch sizes 16, 32, and 64"
- "Run this experiment on GPU 2"

Claude automatically:
- ✅ Tracks experiments in JSON files
- ✅ Adds metric reporting to your code
- ✅ Queues jobs when GPUs are busy
- ✅ Shows everything in `/ds:dash`

**That's it!** No configuration files, no manual setup. Start training and check `/ds:dash` to see your experiments.

---

## How Experiment Tracking Works

### Automatic Tracking

When you ask Claude to train a model, it automatically:

1. **Creates an experiment file** (`experiments/session-*/exp_name.json`)
2. **Adds metric reporting** to your training code:
   ```python
   import subprocess
   def report(name, val):
       subprocess.run(['./scripts/report_metric.sh', name, str(val)])
   ```
3. **Tracks everything**: parameters, metrics, GPU usage, timestamps
4. **Shows in dashboard**: View with `/ds:dash`

### Manual Metric Reporting

Claude adds this code automatically, but you can also add it manually:

```python
import subprocess

# Report any metric
def report(name, value):
    subprocess.run(['./scripts/report_metric.sh', name, str(value)])

# Add observations
def note(text):
    subprocess.run(['./scripts/report_note.sh', text])

# In your training loop
for epoch in range(num_epochs):
    loss = train_one_epoch()
    val_acc = evaluate()

    report('train_loss', loss)
    report('val_accuracy', val_acc)
    report('learning_rate', optimizer.param_groups[0]['lr'])

    if loss > prev_loss:
        note(f'Loss increased from {prev_loss:.4f} to {loss:.4f}')
```

### View Results

Check your experiments anytime:
```bash
/ds:dash              # Live dashboard
/ds:queue             # Job queue status
/ds:dash-all          # All sessions
```

### Experiment Data

All data is stored in JSON files:
```bash
experiments/
├── session-abc123/
│   ├── exp_lora_rank8.json
│   └── exp_batch32.json
└── queue.json
```

Each file contains:
- Experiment metadata (name, description, start time)
- Parameters (learning rate, batch size, etc.)
- Metrics (loss, accuracy, etc.) with timestamps
- Notes and observations
- GPU assignment and resource usage

---

## DS Plugin Installation

**TL;DR:** See [DS Plugin Quick Start](#ds-plugin-quick-start) for the fastest way to get running.

### Method 1: Plugin Marketplace (Recommended)

Inside Claude Code:
```bash
/plugin marketplace add amazon-science/aws-research-science#plugins
/plugin install ds@coral      # ML experiment tracking
/plugin install plot@coral    # (Optional) Diagram generation
```

Then optionally:
```bash
/ds:statusline-setup          # Enable GPU status line
/output-style ds:Precise      # Enable thoughtful scientist mode
```

### Method 2: Direct Plugin Directory (Development)

For plugin development or offline use:

```bash
# Clone the repository
git clone --branch plugins https://github.com/amazon-science/aws-research-science.git coral-ds-plugin

# Start Claude with the plugin
claude --plugin-dir ./coral-ds-plugin
```

### Advanced Configuration

**Manual status line setup** (if `/ds:statusline-setup` doesn't work):

Add to `~/.claude/settings.json`:
```json
{
  "statusLine": {
    "type": "command",
    "command": "${CLAUDE_PLUGIN_ROOT}/scripts/statusline.sh",
    "padding": 0
  }
}
```

**Queue watcher** (for automatic job launching when GPUs free up):
```bash
# Start the background watcher
${CLAUDE_PLUGIN_ROOT}/scripts/queue_start_watcher.sh

# Check if it's running
ps aux | grep queue_watcher

# Stop it if needed
${CLAUDE_PLUGIN_ROOT}/scripts/queue_stop_watcher.sh
```

## Requirements

- **NVIDIA GPU** with `nvidia-smi` - For GPU monitoring
- **Python 3.6+** - For dashboard and scripts
- **Python package: `rich`** - For terminal UI
  ```bash
  pip install rich
  ```
- **jq** - For JSON parsing (optional, for status line)
- **Git** - For repository info (optional)

## Available Commands

### DS Plugin Commands

| Command | Description |
|---------|-------------|
| `/ds:dash` | Live dashboard showing GPUs, queue, processes, and experiments |
| `/ds:queue` | Current job queue status |
| `/ds:dash-all` | View experiments from all Claude sessions |
| `/ds:dash-sessions` | List all experiment sessions |
| `/ds:dash-clear` | Clear completed experiments from current session |
| `/ds:help` | Show help information |
| `/ds:statusline-setup` | Configure GPU status line at bottom |
| `/output-style ds:Precise` | Enable thoughtful scientist mode |

### Plot Plugin Commands

| Command | Description |
|---------|-------------|
| `/plot:tex` | Generate publication-quality LaTeX/TikZ diagrams with iterative refinement |

See [plot-plugin/README.md](plot-plugin/README.md) for plot plugin documentation.

### Command Details

**`/ds:dash` - Live Dashboard**

Shows everything in one view:
- GPU status (utilization, memory, temperature)
- Job queue (running and waiting)
- Active processes
- Experiment results

**`/ds:queue` - Queue Status**

Running jobs, queued jobs, and watcher daemon status.

**Status Line**

After running `/ds:statusline-setup`, see GPU status at the bottom:
```
[Sonnet 4.5 | 90k/200k (45%)] 📁 dir | GPU# (mem/util) |0. 0%/0% |1. 55%/45% |2. 98%/52% ┃Proc: 3┃
```

- **Green**: <30% util (available)
- **Orange**: 30-70% util (moderate)
- **Red**: >70% util (busy)

Updates automatically as you work.

## Philosophy

This plugin follows a "tools when needed" approach:

✅ **What it does:**
- Automatic experiment tracking
- GPU monitoring and job queuing
- Live dashboard and status line

❌ **What it doesn't do:**
- No auto-commits (use `/background git commit ...`)
- No hook spam (only runs on startup)
- No constant background monitoring
- No exit hooks (Claude exits normally)

**Philosophy:** Provide tools when you need them, stay invisible otherwise.

## File Structure

```
aws-research-science/
├── .claude-plugin/
│   ├── marketplace.json         # Marketplace definition
│   └── plugin.json              # DS plugin manifest
├── commands/                    # DS plugin commands
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
├── plot-plugin/                  # Plot plugin directory
│   ├── .claude-plugin/
│   │   └── plugin.json          # Plot plugin manifest
│   ├── commands/
│   │   └── tex.md               # /plot:tex command
│   ├── scripts/
│   │   └── compile_diagram.sh   # LaTeX/TikZ compilation
│   ├── examples/                # Sample diagrams
│   └── README.md                # Plot plugin documentation
├── settings.example.json        # Status line config
└── README.md                    # This file
```

## Advanced: Manual Scripts

For advanced users who want direct script access:

**Experiment Tracking:**
```bash
# Start an experiment manually
EXP_FILE=$(${CLAUDE_PLUGIN_ROOT}/scripts/start_experiment.sh "exp_name" "description" [gpu_id])

# Report metrics
${CLAUDE_PLUGIN_ROOT}/scripts/report_metric.sh metric_name value

# Add notes
${CLAUDE_PLUGIN_ROOT}/scripts/report_note.sh "observation text"

# Mark complete
${CLAUDE_PLUGIN_ROOT}/scripts/complete_experiment.sh completed $EXP_FILE
```

**Job Queue:**
```bash
# Queue a job
${CLAUDE_PLUGIN_ROOT}/scripts/queue_experiment.sh "name" "command" [memory_mb]

# Queue management
${CLAUDE_PLUGIN_ROOT}/scripts/queue_start_watcher.sh   # Start daemon
${CLAUDE_PLUGIN_ROOT}/scripts/queue_stop_watcher.sh    # Stop daemon
${CLAUDE_PLUGIN_ROOT}/scripts/queue_status.sh          # Check status
```

**Display:**
```bash
# Run dashboard directly
python ${CLAUDE_PLUGIN_ROOT}/scripts/dashboard.py --once

# Generate status line
${CLAUDE_PLUGIN_ROOT}/scripts/statusline.sh
```


## Usage Examples

### Natural Language Training

Just talk to Claude naturally:

**Example 1: Single experiment**
```
You: "Train a LoRA model with rank 8, learning rate 1e-4"
```
Claude will:
- Create experiment JSON
- Add metric reporting to code
- Launch training
- Track everything in `/ds:dash`

**Example 2: Multiple experiments**
```
You: "Try LoRA with rank 4, 8, and 16"
```
Claude will:
- Queue all three experiments
- Launch first one immediately (if GPU available)
- Auto-launch others when GPUs free up
- Track all in the queue

**Example 3: Compare approaches**
```
You: "Compare LoRA vs full fine-tuning on this dataset"
```
Claude will:
- Set up both experiments
- Configure proper baselines
- Run them (queued if needed)
- Help you analyze results

### Manual Experiment Queuing

If you prefer manual control:

```bash
# Queue experiments manually
${CLAUDE_PLUGIN_ROOT}/scripts/queue_experiment.sh "exp_name" "python train.py --args" [gpu_mem_mb]

# Examples:
${CLAUDE_PLUGIN_ROOT}/scripts/queue_experiment.sh "lora_rank4" "python train.py --rank 4" 6000
${CLAUDE_PLUGIN_ROOT}/scripts/queue_experiment.sh "lora_rank8" "python train.py --rank 8" 8000
${CLAUDE_PLUGIN_ROOT}/scripts/queue_experiment.sh "lora_rank16" "python train.py --rank 16" 12000
```

### Monitoring Commands

```bash
/ds:dash              # Live dashboard - GPU, queue, experiments
/ds:queue             # Job queue status
/ds:dash-all          # All experiments across sessions
/ds:dash-sessions     # List all sessions
/ds:dash-clear        # Clear completed experiments
```

## Troubleshooting

### Marketplace refresh fails with SSH authentication error

If you see `SSH authentication failed` when adding or refreshing the marketplace, this is a [known Claude Code issue](https://github.com/anthropics/claude-code/issues/13553). Claude Code internally uses SSH URLs for GitHub repos, which fails if SSH keys aren't configured.

**Fix:** Configure git to rewrite SSH URLs to HTTPS:
```bash
git config --global url."https://github.com/".insteadOf git@github.com:
```

If the repository is private, you'll also need a GitHub token:
```bash
export GITHUB_TOKEN=ghp_your_token_here
```

You can generate a token at [github.com/settings/tokens](https://github.com/settings/tokens) (needs `repo` scope for private repos).

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

## How It Works

### Job Queue with Race Condition Prevention

The queue system prevents multiple experiments from launching simultaneously on the same GPU (which causes OOM):

**File Locking**:
- All queue operations use exclusive file locks (`experiments/.queue.lock`)
- Only one process can launch jobs at a time
- Prevents simultaneous GPU checks during startup window

**GPU Reservation Check**:
```bash
# Before launching, check if GPU already has a running job
# This catches the "startup race" where GPU shows free but job is loading
if GPU_has_running_job:
    add_to_queue()
else:
    launch_job()
    hold_lock_for_10s()  # Ensures GPU shows activity
```

**Smart Retry Logic**:
- Fast failure (<60s): Retry up to 3x (likely OOM or device busy)
- Late failure (>60s): No retry (likely code bug)
- Background watcher checks queue every 30s

**How it prevents OOM**:
1. Job 1 acquires lock, launches on GPU 0, holds lock for 10s
2. Job 2 waits for lock, then sees Job 1 in running queue
3. Job 2 queues instead of launching
4. Result: Sequential launches, no OOM

### Experiment Tracking

**Automatic via SessionStart Hook**:
- Claude is told about tracking protocol on startup
- When you say "train a model", Claude automatically uses queue scripts
- No configuration needed

**JSON Storage**:
```
experiments/
├── session-{id}/
│   ├── exp_name.json     # Params, metrics, notes, GPU, timestamps
│   └── ...
├── queue.json             # Running and queued jobs
└── .cleared/              # Archived completed experiments
```

Each experiment JSON contains:
- Metadata (name, description, start/end time, status)
- Parameters (learning rate, batch size, etc.)
- Metrics (loss, accuracy, etc.) with timestamps
- Notes and observations
- GPU assignment and PID

### GPU Monitoring

**Status Line**:
- Runs `statusline.sh` script configured in settings
- Queries `nvidia-smi` for GPU stats
- Color-codes based on utilization: green (<30%), orange (30-70%), red (>70%)

**Dashboard**:
- Python script using Rich library for terminal UI
- Shows GPUs, processes, queue, and experiments in one view
- Updates on demand with `/ds:dash`

## Customization

Want to customize the plugin behavior? All scripts are in `${CLAUDE_PLUGIN_ROOT}/scripts/`:

**Status Line** (`statusline.sh`):
- Modify GPU status thresholds
- Change display format
- Add custom metrics

**Dashboard** (`dashboard.py`):
- Customize table layout
- Add/remove sections
- Change color scheme

**Startup Check** (`env_check.sh`):
- Modify what's shown on session start
- Add custom environment checks
- Change tracking protocol messages

After making changes, restart Claude Code to see updates.

---

## Version

**2.0.0** - Job queue system with smart retry logic

## Author

Amazon CORAL Lab

## License

MIT

## Keywords

ml, experiment-tracking, gpu-monitoring, job-queue, automatic-retry, claude-code-plugin, data-science, visualization, latex, tikz, publication-quality, diagrams
