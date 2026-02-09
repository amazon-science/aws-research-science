#!/bin/bash
# Complete an experiment: update metrics, commit results, update table
# Auto-called by Claude after evaluation

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <experiment_name> <results_json_path>"
    echo "Example: $0 rl_v5_zero_tolerance outputs/eval_rl_v5.json"
    exit 1
fi

EXP_NAME="$1"
RESULTS_JSON="$2"

if [ ! -f "$RESULTS_JSON" ]; then
    echo "❌ Results file not found: $RESULTS_JSON"
    exit 1
fi

echo "📊 Completing experiment: $EXP_NAME"
echo "📁 Results file: $RESULTS_JSON"

# 1. Ensure we're on the experiment branch
current_branch=$(git branch --show-current)
expected_branch="exp/$EXP_NAME"

if [ "$current_branch" != "$expected_branch" ]; then
    echo "⚠️  Current branch: $current_branch"
    echo "⚠️  Expected branch: $expected_branch"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Aborted. Switch to branch: git checkout $expected_branch"
        exit 1
    fi
fi

# 2. Parse results and update metrics.yaml
echo "📝 Updating experiments/metrics.yaml with results..."

python3 - "$RESULTS_JSON" "$EXP_NAME" << 'PYEOF'
import yaml
import json
import sys
from datetime import datetime
from pathlib import Path

results_path = sys.argv[1]
exp_name = sys.argv[2]

# Load results
with open(results_path, 'r') as f:
    results = json.load(f)

# Load current metrics
with open('experiments/metrics.yaml', 'r') as f:
    metrics = yaml.safe_load(f)

# Update status
metrics['status'] = 'completed'
metrics['experiment'] = exp_name
metrics['timestamp'] = datetime.now().isoformat()

# Extract primary metric (try different keys)
primary_value = (
    results.get('test_pass_pct') or
    results.get('accuracy') or
    results.get('score') or
    results.get('primary_metric') or
    0.0
)

metrics['metrics'] = {
    'primary_metric': 'test_pass_rate',
    'value': primary_value
}

# Update evaluation metrics
metrics['evaluation_metrics'] = {}
for key, value in results.items():
    if isinstance(value, (int, float, str)):
        metrics['evaluation_metrics'][key] = value

# Update git info
import subprocess
current_commit = subprocess.run(
    ['git', 'rev-parse', 'HEAD'],
    capture_output=True, text=True
).stdout.strip()

metrics['git_info']['commit'] = current_commit

# Write updated metrics
with open('experiments/metrics.yaml', 'w') as f:
    yaml.dump(metrics, f, default_flow_style=False, sort_keys=False)

print(f"✅ Metrics updated")
print(f"   Primary metric: {primary_value:.4f}")
print(f"   Total metrics: {len(metrics['evaluation_metrics'])}")
PYEOF

echo "✅ experiments/metrics.yaml updated"

# 3. Copy results to results/ directory
echo "📁 Copying results to results/ directory..."
mkdir -p results/
cp "$RESULTS_JSON" "results/${EXP_NAME}_results.json"
echo "✅ Results saved: results/${EXP_NAME}_results.json"

# 4. Update RESULTS_TABLE.md
echo "📊 Updating RESULTS_TABLE.md..."

python3 - "$EXP_NAME" << 'PYEOF'
import yaml
import sys
from datetime import datetime
from pathlib import Path

exp_name = sys.argv[1]

# Load experiment metadata
with open('experiments/params.yaml', 'r') as f:
    params = yaml.safe_load(f)

with open('experiments/metrics.yaml', 'r') as f:
    metrics = yaml.safe_load(f)

# Extract info
exp_date = params['experiment']['date']
model_name = params['model'].get('name', 'Unknown')
branch = params['experiment']['git_branch']
primary_metric = metrics['metrics'].get('primary_metric', 'unknown')
value = metrics['metrics'].get('value', 0.0)

# Determine status
if value > 0.3:
    status = "✅ Success"
elif value > 0.0:
    status = "⚠️  Partial"
else:
    status = "❌ Failed"

# Read current table
table_path = Path('RESULTS_TABLE.md')
if not table_path.exists():
    print("⚠️  RESULTS_TABLE.md not found, creating...")
    lines = [
        "# Master Experiment Results\n",
        "\n",
        "## Summary Table\n",
        "\n",
        "| Branch | Experiment | Date | Model | Primary Metric | Value | Status | Git Tag |\n",
        "|--------|-----------|------|-------|----------------|-------|--------|---------|\n",
    ]
else:
    with open(table_path, 'r') as f:
        lines = f.readlines()

# Find table insertion point (after header row with dashes)
insert_idx = None
for i, line in enumerate(lines):
    if line.strip().startswith('|---'):
        insert_idx = i + 1
        break

if insert_idx is None:
    print("❌ Could not find table header in RESULTS_TABLE.md")
    sys.exit(1)

# Create new row
new_row = f"| {branch} | {exp_name} | {exp_date} | {model_name} | {primary_metric} | {value:.4f} | {status} | - |\n"

# Insert new row
lines.insert(insert_idx, new_row)

# Write back
with open(table_path, 'w') as f:
    f.writelines(lines)

print(f"✅ RESULTS_TABLE.md updated")
print(f"   Experiment: {exp_name}")
print(f"   Value: {value:.4f}")
print(f"   Status: {status}")
PYEOF

echo "✅ RESULTS_TABLE.md updated"

# 5. Get summary metrics for commit message
SUMMARY=$(python3 << 'PYEOF'
import yaml
with open('experiments/metrics.yaml', 'r') as f:
    metrics = yaml.safe_load(f)

primary_value = metrics['metrics'].get('value', 0.0)
eval_metrics = metrics.get('evaluation_metrics', {})

print(f"Primary: {primary_value:.4f}")
for key, value in list(eval_metrics.items())[:3]:
    if isinstance(value, (int, float)):
        print(f"{key}: {value:.4f}")
PYEOF
)

# 6. Commit all results
echo "💾 Committing experiment results..."

git add experiments/ results/ RESULTS_TABLE.md

git commit -m "results: Complete experiment '$EXP_NAME'

Results:
$SUMMARY

Files updated:
- experiments/metrics.yaml (full metrics)
- results/${EXP_NAME}_results.json (detailed results)
- RESULTS_TABLE.md (summary table)

Branch: $(git branch --show-current)
Commit: $(git rev-parse --short HEAD)

Co-Authored-By: Claude DS Agent <claude-experiments@anthropic.com>"

echo ""
echo "✅ Experiment '$EXP_NAME' completed and committed!"
echo ""
echo "📊 Results summary:"
echo "$SUMMARY"
echo ""
echo "📁 Files updated:"
echo "   - experiments/metrics.yaml"
echo "   - results/${EXP_NAME}_results.json"
echo "   - RESULTS_TABLE.md"
echo ""
echo "🔍 View results:"
echo "   cat experiments/metrics.yaml"
echo "   cat RESULTS_TABLE.md"
echo ""
echo "🏷️  Tag this experiment:"
echo "   ./scripts/auto_experiment_tag.sh '$EXP_NAME' 'tag-name' 'description'"
echo ""
echo "📊 Compare to other experiments:"
echo "   ./scripts/compare_experiments.sh exp1 exp2"
echo ""
