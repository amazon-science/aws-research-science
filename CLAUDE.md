# CORAL DS Plugin — Claude Context

This plugin provides ML experiment tracking, GPU monitoring, and intelligent job queuing for Claude Code sessions.

## What This Plugin Does

When active, this plugin:
- Injects GPU availability and experiment status at session start
- Provides commands for viewing experiments and queue status
- Tracks experiments automatically when you run training jobs
- Queues jobs to launch when GPUs become free

## Commands

All `/ds:*` commands are handled instantly by a hook — they run scripts directly with no Claude inference needed:

| Command | What it does |
|:--------|:-------------|
| `/ds:dash` | List all experiments in current session |
| `/ds:queue` | Show job queue status (running, waiting, completed) |
| `/ds:dash-all` | Experiments across all sessions |
| `/ds:dash-sessions` | List all experiment sessions |
| `/ds:dash-clear` | Archive completed experiments |
| `/ds:help` | Show quick help |
| `/ds:reload` | Re-inject plugin context (env state + instructions) into current session without touching history |

## Job Queue

### Basic queuing
```bash
${CLAUDE_PLUGIN_ROOT}/scripts/queue_experiment.sh "exp_name" "python train.py --args" [gpu_mem_mb]
```
- Launches immediately if a GPU is free
- Queues and auto-launches later if GPUs are busy
- Retries fast failures (OOM, device errors) up to 3 times
- Start the background watcher: `${CLAUDE_PLUGIN_ROOT}/scripts/queue_start_watcher.sh`

### Chaining experiments sequentially (--after)
Use `--after <name>` to run a job only after another completes. The dependent job inherits the same GPU automatically.

```bash
${CLAUDE_PLUGIN_ROOT}/scripts/queue_experiment.sh pretrain "python pretrain.py" 12000
${CLAUDE_PLUGIN_ROOT}/scripts/queue_experiment.sh finetune "python finetune.py"  8000 --after pretrain
${CLAUDE_PLUGIN_ROOT}/scripts/queue_experiment.sh eval     "python eval.py"      4000 --after finetune
```

Multiple dependencies (waits for all, inherits last completed dep's GPU):
```bash
${CLAUDE_PLUGIN_ROOT}/scripts/queue_experiment.sh merge "python merge.py" 4000 --after run_a --after run_b
```

Key behaviors:
- Jobs with `--after` are always queued, never launched immediately
- If the inherited GPU has insufficient free VRAM, the watcher waits and retries next cycle
- If deps ran on different GPUs, warns and uses the last-completed dep's GPU
- Independent jobs in the queue skip past blocked deps and run on other GPUs

## Mandatory Workflow — Follow This Every Time

When a user asks to run, train, or queue any experiment, you MUST follow these steps in order. Do not skip any step.

### Step 1 — Instrument the training script
Before queuing, make two changes to the training script:

**1a. Add `report_metric.sh` calls** so metrics appear in `/ds:dash`. Find every place that logs or prints a metric and add:

```python
import subprocess

def report(name, val):
    subprocess.run(['${CLAUDE_PLUGIN_ROOT}/scripts/report_metric.sh', name, str(val)])

def note(text):
    subprocess.run(['${CLAUDE_PLUGIN_ROOT}/scripts/report_note.sh', text])

# Add at every eval/logging point in the training loop:
report('train_loss', loss)
report('val_accuracy', val_acc)
```

**1b. Fix any unconditional `CUDA_VISIBLE_DEVICES` override — this is the critical GPU assignment fix.**

The plugin sets `CUDA_VISIBLE_DEVICES=N` before launch so the job runs on the assigned GPU. However, many training scripts unconditionally overwrite this from their `--device` argument, defeating the assignment entirely. **The plugin's queue-time sed normalization (`--device cuda:X` → `--device cuda:0`) is only a partial safeguard — it does not prevent this override.** The script guard is required.

Find any pattern like:
```python
# BAD — silently overwrites plugin's CUDA_VISIBLE_DEVICES=N with 0:
gpu_id = args.device.split(":")[-1]
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
```

Change it to:
```python
# GOOD — respects plugin's GPU assignment:
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    gpu_id = args.device.split(":")[-1]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
```

Also find and wrap any `torch.cuda.set_device()` calls with the same guard:
```python
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    torch.cuda.set_device(args.device)
```

**This guard is required for GPU assignment to work correctly. Do not skip it.**

If you cannot find the training script or it has no logging, ask the user before queuing.

### Step 2 — Queue the experiment
```bash
${CLAUDE_PLUGIN_ROOT}/scripts/queue_experiment.sh "exp_name" "python train.py --args" [gpu_mem_mb]
```

For sequential jobs on the same GPU, use `--after`:
```bash
${CLAUDE_PLUGIN_ROOT}/scripts/queue_experiment.sh finetune "python finetune.py" 8000 --after pretrain
```

### Step 3 — Tell the user how to monitor
After queuing, always tell the user:
- Run `/ds:dash` to see experiment progress and metrics
- Run `/ds:queue` to check job queue status

Never queue silently without telling the user how to monitor.

### Step 4 — Start the watcher if not running
Check if the queue watcher is running. If not, start it:
```bash
${CLAUDE_PLUGIN_ROOT}/scripts/queue_start_watcher.sh
```

## GPU Assignment

- Queue picks the first idle GPU (utilization <30%, enough free VRAM)
- `CUDA_VISIBLE_DEVICES` is inlined in the command string, not just exported, to ensure it is respected by code using `device_map={"": 0}` or `torch.cuda.set_device()`
- After launch, the lock is held for 45s to cover model loading before the next job can claim the same GPU
- Post-launch: watcher checks VRAM delta after 45s and warns if <500MB (silent device assignment failure)

## Data Storage

```
experiments/
├── session-{id}/
│   ├── exp_name.json     # params, metrics, notes, GPU, timestamps
│   └── ...
├── queue.json            # running, queued, completed lists
└── queue_watcher.log     # watcher daemon output
```

## Rules

- Always instrument training scripts with `report_metric.sh` before queuing — never queue an uninstrumented script without asking the user
- Always use `queue_experiment.sh`, never launch training directly — this ensures GPU assignment, retry, and chaining work correctly
- Always tell the user to check `/ds:dash` after queuing — never queue silently
