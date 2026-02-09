#!/bin/bash
# List all experiments with their status and results

set -e

echo "🔬 ALL EXPERIMENTS"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

# Get all experiment branches
experiment_branches=$(git branch | grep "exp/" | sed 's/^\*//' | xargs)

if [ -z "$experiment_branches" ]; then
    echo "No experiments found."
    echo ""
    echo "💡 Start an experiment:"
    echo "   ./scripts/auto_experiment_start.sh <name> <description>"
    exit 0
fi

for branch in $experiment_branches; do
    branch=$(echo "$branch" | xargs)

    # Get metrics from branch
    metrics=$(git show "$branch:experiments/metrics.yaml" 2>/dev/null) || continue

    # Parse key info
    exp_name=$(echo "$metrics" | grep "^experiment:" | head -1 | awk '{print $2}')
    status=$(echo "$metrics" | grep "^status:" | head -1 | awk '{print $2}')
    primary_value=$(echo "$metrics" | grep -A 1 "^metrics:" | grep "value:" | awk '{print $2}')

    # Get date from params
    params=$(git show "$branch:experiments/params.yaml" 2>/dev/null) || continue
    exp_date=$(echo "$params" | grep "  date:" | head -1 | awk '{print $2}')

    # Status emoji
    case "$status" in
        completed)
            status_emoji="✅"
            ;;
        training|in_progress)
            status_emoji="🔄"
            ;;
        failed)
            status_emoji="❌"
            ;;
        configured|not_started)
            status_emoji="⏸️ "
            ;;
        *)
            status_emoji="❓"
            ;;
    esac

    # Print experiment info
    echo "$status_emoji $branch"
    echo "   Name: $exp_name"
    echo "   Date: $exp_date"
    echo "   Status: $status"
    echo "   Primary Metric: ${primary_value:-N/A}"
    echo ""
done

echo "═══════════════════════════════════════════════════════════════════"
echo "🏷️  TAGGED EXPERIMENTS"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

tags=$(git tag -l)
if [ -z "$tags" ]; then
    echo "No tagged experiments yet."
    echo ""
    echo "💡 Tag a successful experiment:"
    echo "   ./scripts/auto_experiment_tag.sh <exp_name> <tag> <message>"
else
    git tag -l -n1 | while read -r tag message; do
        echo "📌 $tag"
        echo "   $message"
        echo ""
    done
fi

echo "═══════════════════════════════════════════════════════════════════"
echo "📊 QUICK ACTIONS"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "View results table:"
echo "   cat RESULTS_TABLE.md"
echo ""
echo "Compare experiments:"
echo "   ./scripts/compare_experiments.sh exp1_name exp2_name"
echo ""
echo "View experiment details:"
echo "   git checkout exp/<name>"
echo "   cat experiments/params.yaml"
echo "   cat experiments/metrics.yaml"
echo ""
echo "Start new experiment:"
echo "   ./scripts/auto_experiment_start.sh <name> <description>"
echo ""
