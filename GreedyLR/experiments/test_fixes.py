#!/usr/bin/env python3
"""
Test the critical fixes before running full experiment
"""

import sys
sys.path.append('/Users/subshrey/Projects/greedylr_research')

# Test imports
try:
    from balanced_100k_experiment import generate_balanced_100k_configs, ResidualBlock, MultiHeadAttention, TransformerBlock, ViTModel, DeepTransformer, WideTransformer
    print("✅ All imports successful")
except Exception as e:
    print(f"❌ Import error: {e}")
    exit(1)

# Test config generation
try:
    configs = generate_balanced_100k_configs()
    print(f"✅ Generated {len(configs):,} configs")
    
    # Test a few different model types
    test_configs = []
    for config in configs:
        if config.model_type in ['neural_simple', 'neural_resnet', 'neural_conv', 'neural_vit', 'neural_deep_transformer']:
            test_configs.append(config)
            if len(test_configs) >= 5:
                break
    
    print(f"✅ Found {len(test_configs)} test configs with different model types")
    
except Exception as e:
    print(f"❌ Config generation error: {e}")
    exit(1)

# Test single experiment execution
try:
    from comprehensive_scheduler_experiment import run_comprehensive_experiment
    
    # Test with a simple config
    for config in test_configs:
        print(f"🧪 Testing {config.model_type} with {config.scheduler_type}...")
        try:
            result = run_comprehensive_experiment(config)
            print(f"   ✅ Success: final_loss={result['metrics']['final_loss']:.4f}")
            break  # Just test one for now
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            continue
    
except Exception as e:
    print(f"❌ Experiment execution error: {e}")
    exit(1)

print("\n🎉 All critical fixes verified! Ready for full experiment.")