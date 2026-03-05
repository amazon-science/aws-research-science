#!/bin/bash
# Show queue status with job lineage chains

QUEUE_FILE="experiments/queue.json"
PID_FILE="experiments/queue_watcher.pid"

if [ ! -f "$QUEUE_FILE" ]; then
    echo "📋 Queue is empty"
    exit 0
fi

echo "📋 Queue Status"
echo ""

# Running jobs
RUNNING_COUNT=$(jq '.running | length' "$QUEUE_FILE")
if [ "$RUNNING_COUNT" -gt 0 ]; then
    echo "✓ Running ($RUNNING_COUNT):"
    jq -r '.running[] | "  \(.name)  GPU \(.gpu)  (PID: \(.pid))  started \(.started_at[:16] // "")"' \
        "$QUEUE_FILE"
    echo ""
fi

# Queued jobs with lineage chains
QUEUED_COUNT=$(jq '.queued | length' "$QUEUE_FILE")
if [ "$QUEUED_COUNT" -gt 0 ]; then
    echo "⏳ Queued ($QUEUED_COUNT):"

    # Build and display chains using Python for graph traversal
    python3 - "$QUEUE_FILE" <<'PYEOF'
import json, sys

with open(sys.argv[1]) as f:
    data = json.load(f)

running = {j['name']: j for j in data.get('running', [])}
queued  = {j['name']: j for j in data.get('queued',  [])}
active  = dict(running, **queued)

# Build child map: name -> [names of queued jobs that depend on it]
children = {n: [] for n in active}
for name, job in queued.items():
    for dep in job.get('depends_on', []):
        if dep in active:
            children.setdefault(dep, []).append(name)

# Jobs that are a dependency of another active job
has_active_parent = {
    name
    for name, job in queued.items()
    for dep in job.get('depends_on', [])
    if dep in active
}

# Roots = active jobs with no active parent
roots = [n for n in active if n not in has_active_parent]

def fmt(name):
    if name in running:
        gpu = running[name].get('gpu', '?')
        return f"{name}[GPU {gpu}]"
    job = queued.get(name, {})
    retry = job.get('retry_count', 0)
    suffix = f" (retry {retry}/3)" if retry else ""
    return f"{name}{suffix}"

def chains(name):
    """Return list of chain strings rooted at name."""
    kids = children.get(name, [])
    if not kids:
        return [fmt(name)]
    result = []
    for kid in kids:
        for sub in chains(kid):
            result.append(fmt(name) + " → " + sub)
    return result

printed = set()
for root in roots:
    for chain in chains(root):
        # Only print queued chains (skip pure-running standalone lines
        # already shown above), but show chains that START with running
        first_name = root
        if first_name in queued or " → " in chain:
            print(f"  {chain}")
            printed.add(root)
PYEOF

    echo ""
fi

# Watcher status
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "🔄 Watcher: running (PID: $PID)"
    else
        echo "⚠️  Watcher: not running (stale PID) — run queue_start_watcher.sh"
    fi
else
    echo "⚠️  Watcher: not running — run queue_start_watcher.sh"
fi
