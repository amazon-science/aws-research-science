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

## Experiment Tracking

### Automatic (preferred)
When you queue a job, the plugin creates an experiment file automatically. Before queuing, instrument the training script to call `report_metric.sh` so the dashboard shows live progress:

```python
import subprocess

def report(name, val):
    subprocess.run(['${CLAUDE_PLUGIN_ROOT}/scripts/report_metric.sh', name, str(val)])

def note(text):
    subprocess.run(['${CLAUDE_PLUGIN_ROOT}/scripts/report_note.sh', text])

# In training loop:
report('train_loss', loss)
report('val_accuracy', val_acc)
note('Learning rate reduced at epoch 10')
```

### Manual tracking
```bash
EXP_FILE=$(${CLAUDE_PLUGIN_ROOT}/scripts/start_experiment.sh "name" "description" [gpu_id])
# ... run training ...
${CLAUDE_PLUGIN_ROOT}/scripts/complete_experiment.sh completed $EXP_FILE
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

## Philosophy

- **Automatic**: just say "train a model" — Claude queues and tracks without being asked
- **Instrument first**: always add `report_metric.sh` calls to training scripts before queuing so results are visible in `/ds:dash`
- **Queue, don't launch directly**: use `queue_experiment.sh` so GPU assignment, retry, and chaining work correctly
- **Non-intrusive**: only runs on session start, no constant background hooks
