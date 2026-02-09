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

## Background Tasks

Claude automatically runs long tasks in the background. Check status with:
- `/tasks` - List all background tasks with status
- `Ctrl+B` - Quick view of background tasks
- `/ds:dash` - Dashboard with GPUs, processes, experiments

**Tip**: Say "run this in the background" to explicitly request background execution.

## Statusline

Shows: `[Model | tokens] 📁 dir | GPU# (mem/util) |0. X%/Y% ... Proc:N`

Color-coded GPUs:
- Green: <30% util (available)
- Orange: 30-70% util (moderate)
- Red: >70% util (busy)

Format: `|GPU#. memory%/utilization%`

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
