#!/usr/bin/env python3
"""
Quick fixes for the experiment bugs
"""

import sys
sys.path.append('/Users/subshrey/Projects/greedylr_research')

# Test the fixes
def test_fixes():
    print("🔧 Testing experiment fixes...")
    
    # 1. Test AdvancedGreedyLR parameters
    from comprehensive_scheduler_experiment import AdvancedGreedyLR
    import torch.optim as optim
    import torch.nn as nn
    
    # Create dummy model and optimizer
    model = nn.Linear(10, 1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Test correct AdvancedGreedyLR initialization
    try:
        scheduler = AdvancedGreedyLR(optimizer, patience=10, min_lr=1e-6, factor=0.5)
        print("✅ AdvancedGreedyLR initialization works")
    except Exception as e:
        print(f"❌ AdvancedGreedyLR error: {e}")
    
    # 2. Check for ResidualBlock
    try:
        from comprehensive_scheduler_experiment import ResidualBlock
        print("✅ ResidualBlock found")
    except ImportError:
        print("❌ ResidualBlock missing - need to define it")
    
    print("🎯 Fixes needed:")
    print("1. Change 'initial_lr' to 'min_lr' in balanced_100k_experiment.py")
    print("2. Add ResidualBlock definition if missing")
    print("3. Fix model variable scope issues")

if __name__ == "__main__":
    test_fixes()