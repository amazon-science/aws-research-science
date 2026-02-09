#!/bin/bash
# Initialize git-native experiment tracking
# Auto-called by Claude when starting experiments

set -e

PROJECT_DIR="${1:-.}"
cd "$PROJECT_DIR"

echo "🔧 Initializing git-native experiment tracking in $PROJECT_DIR"

# Check if already initialized
if git rev-parse --git-dir > /dev/null 2>&1; then
    echo "✅ Git repository already exists"
    exit 0
fi

# Initialize git
git init
git config user.name "Claude DS Agent"
git config user.email "claude-experiments@anthropic.com"

echo "✅ Git initialized"

# Create .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
.venv/
venv/
*.egg-info/
.Python

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Large model files (>100MB) - track with DVC if needed
outputs/*/checkpoint-*/
outputs/*/pytorch_model.bin
*.safetensors
*.pth
*.pt
*.bin
model.safetensors

# Logs
*.log
logs/
wandb/

# Data cache
.cache/
mteb___*/
hub/
datasets/
models--*/

# Results (track with git, but large eval outputs with DVC)
# results/*.json are tracked
# outputs/*.json are tracked

# OS
.DS_Store
.nfs*
*.swp
*.swo

# DVC (if used)
/outputs/*.dvc
/data/*.dvc

# IDE
.vscode/
.idea/
*.code-workspace
EOF

echo "✅ .gitignore created"

# Create directory structure
mkdir -p experiments configs results outputs src data scripts

# Initialize params.yaml template
cat > experiments/params.yaml << 'EOF'
# Experiment parameters (auto-updated by Claude)
experiment:
  name: "baseline"
  description: "Initial baseline experiment"
  date: "2026-02-02"
  git_branch: "main"
  base_branch: "main"

model:
  name: "unknown"
  parameters: null

training:
  epochs: null
  batch_size: null
  learning_rate: null

data:
  dataset: null
  num_samples: null

hardware:
  gpu_id: null
  gpu_name: null

status: "initialized"
EOF

echo "✅ experiments/params.yaml created"

# Initialize metrics.yaml template
cat > experiments/metrics.yaml << 'EOF'
# Experiment metrics (auto-updated after evaluation)
experiment: "baseline"
status: "no_results"

metrics:
  primary_metric: null
  value: null

training_metrics:
  total_steps: null
  final_loss: null
  training_time_hours: null

evaluation_metrics: {}

hardware_metrics:
  gpu_utilization_avg: null
  memory_used_mb: null
EOF

echo "✅ experiments/metrics.yaml created"

# Initialize RESULTS_TABLE.md
cat > RESULTS_TABLE.md << 'EOF'
# Master Experiment Results

Last updated: $(date +%Y-%m-%d)

## Summary Table

| Branch | Experiment | Date | Model | Primary Metric | Value | Status | Git Tag |
|--------|-----------|------|-------|----------------|-------|--------|---------|
| main | - | $(date +%Y-%m-%d) | - | - | - | ✅ Initialized | - |

## Best Results
- None yet

## Failed Experiments
- None yet

## In Progress
- None

## Notes
- This table is auto-updated by Claude after each experiment
- Use `git diff` to compare experiments
- Use `git checkout exp/<name>` to view experiment details
- Use `git tag -l` to see tagged successful experiments
EOF

echo "✅ RESULTS_TABLE.md created"

# Create README if it doesn't exist
if [ ! -f README.md ]; then
    cat > README.md << 'EOF'
# ML Experiment Tracking

This project uses git-native experiment tracking.

## Quick Start

### View all experiments
```bash
git branch | grep exp/
```

### View results
```bash
cat RESULTS_TABLE.md
```

### Compare experiments
```bash
./scripts/compare_experiments.sh exp1_name exp2_name
```

### Reproduce an experiment
```bash
git checkout exp/experiment_name
cat experiments/params.yaml  # View parameters
```

### View experiment history
```bash
git log --graph --oneline --all
```

## Structure

- `experiments/` - Current experiment parameters and metrics
- `configs/` - Experiment configuration files
- `results/` - Small result files (tracked in git)
- `outputs/` - Large model artifacts (ignored or DVC-tracked)
- `src/` - Source code
- `data/` - Data files
- `scripts/` - Automation scripts
- `RESULTS_TABLE.md` - Centralized results table

## Experiment Workflow

1. Claude creates new branch: `exp/<experiment_name>`
2. Updates `experiments/params.yaml` with config
3. Runs training and commits checkpoints
4. Evaluates and updates `experiments/metrics.yaml`
5. Updates `RESULTS_TABLE.md`
6. Tags successful experiments

All automated by Claude DS Agent!
EOF
    echo "✅ README.md created"
fi

# Initial commit
git add .gitignore experiments/ configs/ results/ outputs/ src/ data/ scripts/ RESULTS_TABLE.md README.md
git commit -m "chore: Initialize git-native experiment tracking

- Set up experiment tracking structure
- Created params.yaml and metrics.yaml templates
- Initialized RESULTS_TABLE.md for centralized results
- Configured .gitignore for ML artifacts
- Added automation scripts

System: Git-native ML experiment tracking
Managed by: Claude DS Agent
Date: $(date +%Y-%m-%d)

Co-Authored-By: Claude DS Agent <claude-experiments@anthropic.com>"

echo ""
echo "✅ Git-native experiment tracking initialized!"
echo ""
echo "📁 Structure created:"
echo "   - experiments/ (params.yaml, metrics.yaml)"
echo "   - configs/ (experiment configs)"
echo "   - results/ (tracked results)"
echo "   - outputs/ (large artifacts, .gitignored)"
echo "   - scripts/ (automation)"
echo ""
echo "📝 Next steps:"
echo "   1. Start an experiment: ./scripts/auto_experiment_start.sh <name> <description>"
echo "   2. View results: cat RESULTS_TABLE.md"
echo "   3. List experiments: git branch | grep exp/"
echo ""
