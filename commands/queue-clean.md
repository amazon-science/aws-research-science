---
description: Remove stale dead-process entries from the running queue so GPUs show as free
---

Execute immediately without explanation:

```bash
bash ${CLAUDE_PLUGIN_ROOT}/scripts/queue_clean.sh
```

Show the output as-is. If dead jobs were removed, also run `/ds:queue` to show the updated queue state.
