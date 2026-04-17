#!/bin/bash
# 🚀 Start Optimized 8K Comprehensive Experiment with MPS Acceleration
# Run this script to start the optimized experiment

echo "🚀 STARTING OPTIMIZED 8K COMPREHENSIVE EXPERIMENT"
echo "================================================="
echo "📊 Configuration:"
echo "   • Total Experiments: 8,100"
echo "   • GreedyLR: 3,240 experiments (40%) - Factor=0.9, patience=[1,10]"
echo "   • Other Schedulers: 4,860 experiments (60%)"
echo "   • Model types: 12 (all analytical + neural including transformers)"
echo "   • Noise types: 9 × 3 strengths (full robustness testing)"
echo "   • Training steps: 200 (optimized for speed)"
echo "   • MPS GPU acceleration: ✅"
echo "   • Expected runtime: ~2.8 hours"
echo "   • Incremental saving every 50 experiments: ✅"
echo ""

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Check dependencies
echo "📦 Checking dependencies..."
python -c "import torch, pandas, numpy, psutil; print('✅ All dependencies available')"

# Clear any existing progress files
echo "🧹 Clearing previous results..."
rm -f robust_results.json robust_progress.json robust_errors.log 2>/dev/null
rm -rf checkpoints/ 2>/dev/null

echo ""
echo "🎯 STARTING EXPERIMENT IN BACKGROUND..."
echo "Files:"
echo "   • Results: robust_results.json"
echo "   • Progress: robust_progress.json" 
echo "   • Errors: robust_errors.log"
echo "   • Checkpoints: checkpoints/"
echo ""
echo "To monitor progress, run: python monitor_robust.py"
echo ""

# Start experiment in background
nohup python robust_comprehensive_experiment.py > experiment_output.log 2>&1 &
EXPERIMENT_PID=$!

echo "✅ Experiment started with PID: $EXPERIMENT_PID"
echo "📝 Output logged to: experiment_output.log"
echo ""
echo "🔍 Starting monitor in 5 seconds..."
sleep 5

# Start monitor
python monitor_robust.py