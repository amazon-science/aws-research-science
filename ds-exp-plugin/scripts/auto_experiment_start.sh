#!/bin/bash
# Start a new experiment with git branch and parameter tracking
# Auto-called by Claude when starting experiments

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <experiment_name> <description> [base_branch]"
    echo "Example: $0 rl_v5_zero_tolerance 'Zero-tolerance rewards' main"
    exit 1
fi

EXP_NAME="$1"
EXP_DESCRIPTION="$2"
BASE_BRANCH="${3:-main}"

echo "🔬 Starting new experiment: $EXP_NAME"

# 1. Ensure we're in a git repo
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "❌ Not in a git repository. Run ./scripts/init_git_tracking.sh first"
    exit 1
fi

# 2. Save current work if on base branch
current_branch=$(git branch --show-current)
if [ "$current_branch" = "$BASE_BRANCH" ]; then
    echo "📝 Saving work on $BASE_BRANCH..."
    git add -A
    git commit -m "chore: Save work before branching to exp/$EXP_NAME" || true
fi

# 3. Checkout base branch
echo "🌿 Switching to base branch: $BASE_BRANCH"
git checkout "$BASE_BRANCH" 2>/dev/null || git checkout -b "$BASE_BRANCH"

# 4. Create experiment branch
BRANCH_NAME="exp/$EXP_NAME"
echo "🌿 Creating experiment branch: $BRANCH_NAME"

if git show-ref --verify --quiet "refs/heads/$BRANCH_NAME"; then
    echo "⚠️  Branch $BRANCH_NAME already exists"
    read -p "Overwrite? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Aborted"
        exit 1
    fi
    git branch -D "$BRANCH_NAME"
fi

git checkout -b "$BRANCH_NAME"

# 5. Update params.yaml
echo "📝 Updating experiments/params.yaml..."

cat > experiments/params.yaml << EOF
# Experiment: $EXP_NAME
# Description: $EXP_DESCRIPTION
# Created: $(date +%Y-%m-%d\ %H:%M:%S)

experiment:
  name: "$EXP_NAME"
  description: "$EXP_DESCRIPTION"
  date: "$(date +%Y-%m-%d)"
  git_branch: "$BRANCH_NAME"
  base_branch: "$BASE_BRANCH"
  created_by: "Claude DS Agent"

model:
  name: null  # To be filled by training script
  architecture: null
  parameters: null

training:
  epochs: null
  batch_size: null
  learning_rate: null
  optimizer: null

data:
  dataset: null
  num_samples: null
  split: null

hardware:
  gpu_id: null
  gpu_name: null
  memory_allocated_mb: null

status: "configured"
timestamp: "$(date -Iseconds)"
EOF

echo "✅ experiments/params.yaml updated"

# 6. Reset metrics.yaml for new experiment
echo "📊 Initializing experiments/metrics.yaml..."

cat > experiments/metrics.yaml << EOF
# Experiment Metrics: $EXP_NAME
# Updated: $(date +%Y-%m-%d\ %H:%M:%S)

experiment: "$EXP_NAME"
status: "not_started"

metrics:
  primary_metric: null
  value: null

training_metrics:
  total_steps: null
  final_loss: null
  training_time_hours: null
  start_time: null
  end_time: null

evaluation_metrics: {}

hardware_metrics:
  gpu_id: null
  gpu_utilization_avg: null
  memory_used_mb: null
  memory_peak_mb: null

git_info:
  branch: "$BRANCH_NAME"
  commit: null  # Will be filled after training
  base_commit: "$(git rev-parse $BASE_BRANCH)"

timestamp: "$(date -Iseconds)"
EOF

echo "✅ experiments/metrics.yaml initialized"

# 7. Commit experiment configuration
echo "💾 Committing experiment configuration..."

git add experiments/params.yaml experiments/metrics.yaml

git commit -m "feat(exp): Start experiment '$EXP_NAME'

Description: $EXP_DESCRIPTION

Branch: $BRANCH_NAME
Base: $BASE_BRANCH
Status: Configured

Configuration files updated:
- experiments/params.yaml (experiment parameters)
- experiments/metrics.yaml (metrics template)

Next steps:
1. Update params.yaml with model/training config
2. Run training script
3. Evaluate and commit results

Co-Authored-By: Claude DS Agent <claude-experiments@anthropic.com>"

echo ""
echo "✅ Experiment '$EXP_NAME' started successfully!"
echo ""
echo "📍 Details:"
echo "   Branch: $BRANCH_NAME"
echo "   Base: $BASE_BRANCH"
echo "   Description: $EXP_DESCRIPTION"
echo ""
echo "📝 Configuration files:"
echo "   - experiments/params.yaml"
echo "   - experiments/metrics.yaml"
echo ""
echo "🚀 Next steps:"
echo "   1. Update experiments/params.yaml with your config"
echo "   2. Run your training script"
echo "   3. Call ./scripts/auto_experiment_complete.sh after evaluation"
echo ""
echo "💡 View current branch: git branch --show-current"
echo "💡 View config: cat experiments/params.yaml"
echo ""
