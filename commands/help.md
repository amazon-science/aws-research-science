# DS Experiment Plugin - Quick Help

## How It Works

**Training tasks are automatically tracked!** Just say "train a model with rank 6" and Claude will:
- ✅ Create experiment JSON file
- ✅ Add metric reporting to training code
- ✅ Track progress automatically
- ✅ Show results in `/ds:dash`

No extra commands needed!

## Commands

- `/ds:dash` - View dashboard (GPUs, processes, experiments)
- `/ds:queue` - Job queue status
- `/ds:reload` - Re-inject plugin context after a plugin update (no history loss)
- `/ds:help` - Show this help

## Optional: Precise Output Style

For a more thoughtful, collaborative AI partner:

`/output-style ds:Precise`

This changes Claude to be:
- More thorough in explanations
- Proactive about suggesting alternatives
- Discusses tradeoffs openly
- Acts like a thoughtful lab partner

Default Claude works fine too - this is optional for users who want more scientific rigor.

## Job Queue

Queue experiments to run when GPUs are free:
- Say "queue this experiment" or "run these back to back"
- Claude will call `queue_experiment.sh` automatically

**Chaining experiments (sequential, same GPU):**
```
queue_experiment.sh pretrain  "python pretrain.py"  12000
queue_experiment.sh finetune  "python finetune.py"   8000 --after pretrain
queue_experiment.sh eval      "python eval.py"       4000 --after finetune
```
- `--after <name>` waits for that job to complete before launching
- Inherits the same GPU as the dependency automatically
- Multiple deps: `--after exp_a --after exp_b` (waits for all)

Check queue status: `/ds:queue`

## Background Tasks

Claude automatically runs long tasks in the background. Check status with:
- `/tasks` - List all background tasks with status
- `Ctrl+B` - Quick view of background tasks
- `/ds:dash` - Dashboard with GPUs, processes, experiments

**Tip**: Say "run this in the background" to explicitly request background execution.

## Statusline

Shows: `[Model | tokens] 📁 dir | #0:mem%/util% | #1:mem%/util% ... | ┃Proc:N┃`

Color-coded GPUs:
- Green: <30% (available)
- Orange: 30-70% (moderate)
- Red: >70% (busy)

## Setup

The statusline is configured in `~/.claude/settings.json`:
```json
{
  "statusLine": {
    "type": "command",
    "command": "/path/to/coral-ds-plugin/scripts/statusline.sh"
  }
}
```

Run `/ds:statusline-setup` to configure automatically.
