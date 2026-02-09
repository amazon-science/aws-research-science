#!/bin/bash
# Compare two experiments
# Usage: ./compare_experiments.sh exp1_name exp2_name

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <exp1_name> <exp2_name>"
    echo "Example: $0 rl_v3_ast rl_v4_multilevel"
    exit 1
fi

EXP1="$1"
EXP2="$2"

BRANCH1="exp/$EXP1"
BRANCH2="exp/$EXP2"

echo "📊 Comparing: $EXP1 vs $EXP2"
echo "─────────────────────────────────────────"
echo ""

# Check branches exist
if ! git show-ref --verify --quiet "refs/heads/$BRANCH1"; then
    echo "❌ Branch not found: $BRANCH1"
    exit 1
fi

if ! git show-ref --verify --quiet "refs/heads/$BRANCH2"; then
    echo "❌ Branch not found: $BRANCH2"
    exit 1
fi

# Compare parameters
echo "═══════════════════════════════════════════"
echo "📋 PARAMETER DIFFERENCES"
echo "═══════════════════════════════════════════"
echo ""

git diff "$BRANCH1:experiments/params.yaml" "$BRANCH2:experiments/params.yaml" || echo "No parameter differences"

echo ""
echo "═══════════════════════════════════════════"
echo "📊 METRIC DIFFERENCES"
echo "═══════════════════════════════════════════"
echo ""

git diff "$BRANCH1:experiments/metrics.yaml" "$BRANCH2:experiments/metrics.yaml" || echo "No metric differences"

echo ""
echo "═══════════════════════════════════════════"
echo "📈 SIDE-BY-SIDE COMPARISON"
echo "═══════════════════════════════════════════"
echo ""

python3 - "$EXP1" "$EXP2" << 'PYEOF'
import yaml
import sys
import subprocess

exp1 = sys.argv[1]
exp2 = sys.argv[2]

def get_yaml_from_branch(branch, file_path):
    """Get YAML file from a git branch"""
    try:
        result = subprocess.run(
            ['git', 'show', f'{branch}:{file_path}'],
            capture_output=True, text=True, check=True
        )
        return yaml.safe_load(result.stdout)
    except:
        return {}

# Load metrics from both branches
m1 = get_yaml_from_branch(f'exp/{exp1}', 'experiments/metrics.yaml')
m2 = get_yaml_from_branch(f'exp/{exp2}', 'experiments/metrics.yaml')

# Load params for additional context
p1 = get_yaml_from_branch(f'exp/{exp1}', 'experiments/params.yaml')
p2 = get_yaml_from_branch(f'exp/{exp2}', 'experiments/params.yaml')

# Print header
print(f"{'Metric':<35} | {exp1:<20} | {exp2:<20} | Delta")
print("─" * 95)

# Primary metric
primary_val1 = m1.get('metrics', {}).get('value', 0)
primary_val2 = m2.get('metrics', {}).get('value', 0)
delta = primary_val2 - primary_val1
sign = "+" if delta > 0 else ""
emoji = "📈" if delta > 0 else "📉" if delta < 0 else "━"

print(f"{'PRIMARY METRIC':<35} | {primary_val1:>19.4f} | {primary_val2:>19.4f} | {emoji} {sign}{delta:>6.4f}")
print("─" * 95)

# Evaluation metrics
eval1 = m1.get('evaluation_metrics', {})
eval2 = m2.get('evaluation_metrics', {})

all_keys = set(eval1.keys()) | set(eval2.keys())
for key in sorted(all_keys):
    v1 = eval1.get(key, 0)
    v2 = eval2.get(key, 0)

    # Skip non-numeric
    if not isinstance(v1, (int, float)) or not isinstance(v2, (int, float)):
        continue

    delta = v2 - v1
    sign = "+" if delta > 0 else ""

    print(f"{key:<35} | {v1:>19.4f} | {v2:>19.4f} | {sign}{delta:>10.4f}")

print()
print("─" * 95)
print("Model & Training")
print("─" * 95)

# Model info
model1 = p1.get('model', {}).get('name', 'Unknown')
model2 = p2.get('model', {}).get('name', 'Unknown')
print(f"{'Model':<35} | {model1:<20} | {model2:<20} | ")

# Training info
train1 = p1.get('training', {})
train2 = p2.get('training', {})

for key in ['epochs', 'batch_size', 'learning_rate']:
    v1 = train1.get(key, 'N/A')
    v2 = train2.get(key, 'N/A')
    print(f"{key:<35} | {str(v1):<20} | {str(v2):<20} | ")

print()
PYEOF

echo ""
echo "═══════════════════════════════════════════"
echo "📝 SUMMARY"
echo "═══════════════════════════════════════════"
echo ""
echo "Experiment 1: $EXP1 (branch: $BRANCH1)"
echo "Experiment 2: $EXP2 (branch: $BRANCH2)"
echo ""
echo "🔍 View detailed diff:"
echo "   git diff $BRANCH1 $BRANCH2"
echo ""
echo "📂 View experiment details:"
echo "   git checkout $BRANCH1 && cat experiments/metrics.yaml"
echo "   git checkout $BRANCH2 && cat experiments/metrics.yaml"
echo ""
