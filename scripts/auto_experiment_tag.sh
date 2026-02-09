#!/bin/bash
# Tag a successful experiment
# Auto-called by Claude for good results

set -e

if [ $# -lt 3 ]; then
    echo "Usage: $0 <experiment_name> <tag_name> <tag_message>"
    echo "Example: $0 rl_v5 best-0.5pct 'Best test pass rate: 0.5%'"
    exit 1
fi

EXP_NAME="$1"
TAG_NAME="$2"
TAG_MESSAGE="$3"

echo "🏷️  Tagging experiment: $EXP_NAME as $TAG_NAME"

# Ensure we're on the experiment branch
BRANCH_NAME="exp/$EXP_NAME"
current_branch=$(git branch --show-current)

if [ "$current_branch" != "$BRANCH_NAME" ]; then
    echo "🌿 Switching to branch: $BRANCH_NAME"
    git checkout "$BRANCH_NAME"
fi

# Read metrics for tag annotation
METRICS=$(cat experiments/metrics.yaml 2>/dev/null || echo "")

# Create annotated tag
echo "📝 Creating annotated tag: $TAG_NAME"

git tag -a "$TAG_NAME" -m "$TAG_MESSAGE

Experiment: $EXP_NAME
Branch: $BRANCH_NAME
Date: $(date +%Y-%m-%d)
Commit: $(git rev-parse --short HEAD)

=== Metrics ===
$METRICS

Tagged by: Claude DS Agent
Timestamp: $(date -Iseconds)"

echo "✅ Tag created: $TAG_NAME"

# Update RESULTS_TABLE.md with tag info
echo "📊 Updating RESULTS_TABLE.md with tag..."

python3 - "$EXP_NAME" "$TAG_NAME" << 'PYEOF'
import sys
from pathlib import Path

exp_name = sys.argv[1]
tag_name = sys.argv[2]

table_path = Path('RESULTS_TABLE.md')
if not table_path.exists():
    print("⚠️  RESULTS_TABLE.md not found")
    sys.exit(0)

with open(table_path, 'r') as f:
    lines = f.readlines()

# Find and update the row for this experiment
updated = False
for i, line in enumerate(lines):
    if f'exp/{exp_name}' in line and '|' in line:
        # Replace the last column (tag) with the new tag
        parts = line.split('|')
        if len(parts) >= 8:
            parts[-2] = f' {tag_name} '
            lines[i] = '|'.join(parts)
            updated = True
            break

if updated:
    with open(table_path, 'w') as f:
        f.writelines(lines)
    print(f"✅ Updated RESULTS_TABLE.md with tag: {tag_name}")
else:
    print(f"⚠️  Could not find experiment in RESULTS_TABLE.md")
PYEOF

# Commit the tag update
git add RESULTS_TABLE.md
git commit -m "tag: Mark experiment '$EXP_NAME' as '$TAG_NAME'

Tag: $TAG_NAME
Message: $TAG_MESSAGE

Co-Authored-By: Claude DS Agent <claude-experiments@anthropic.com>" || true

echo ""
echo "✅ Experiment '$EXP_NAME' tagged as '$TAG_NAME'"
echo ""
echo "📝 Tag details:"
echo "   Name: $TAG_NAME"
echo "   Message: $TAG_MESSAGE"
echo "   Branch: $BRANCH_NAME"
echo ""
echo "🔍 View tag:"
echo "   git show $TAG_NAME"
echo ""
echo "📋 List all tags:"
echo "   git tag -l -n1"
echo ""
echo "🔄 Checkout this experiment later:"
echo "   git checkout $TAG_NAME"
echo ""
