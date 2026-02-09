---
description: Set up the GPU status line at the bottom of Claude Code
---

To enable the status line showing GPU info at the bottom of Claude Code:

1. Add this to your `~/.claude/settings.json`:

```json
{
  "statusLine": {
    "type": "command",
    "command": "${CLAUDE_PLUGIN_ROOT}/scripts/statusline.sh",
    "padding": 0
  }
}
```

2. Restart Claude Code

The status line will show: `[Model] 📁 directory | 🔥 GPU status | ⚙️ training jobs`

Would you like me to add this to your settings.json now?
