#!/bin/bash
# Initialize simple experiment tracking (no git required)

set -e

PROJECT_DIR="${1:-.}"
cd "$PROJECT_DIR"

echo "🔧 Initializing experiment tracking in $PROJECT_DIR"

# Create experiments directory
mkdir -p experiments

# Create a simple README
cat > experiments/README.md << 'EOF'
# Experiment Tracking

Each experiment is stored as a JSON file with all metadata and metrics.

## View experiments
- `/ds-exp:dash` - Dashboard view
- `ls experiments/*.json` - List all experiments
- `cat experiments/exp_*.json` - View specific experiment

## File format
```json
{
  "name": "experiment_name",
  "description": "What this experiment does",
  "status": "running|completed|failed",
  "start_time": "2026-02-05T19:30:00",
  "end_time": "2026-02-05T20:15:00",
  "params": {"rank": 6, "lr": 0.001},
  "metrics": {"loss": 0.45, "accuracy": 0.89},
  "gpu": 2,
  "pid": 12345
}
```
EOF

echo "✅ Experiment tracking initialized!"
echo ""
echo "📁 Structure created:"
echo "   - experiments/ (JSON files for each experiment)"
echo ""
echo "📝 Next steps:"
echo "   1. Run training with metric reporting"
echo "   2. View dashboard: /ds-exp:dash"
echo ""
