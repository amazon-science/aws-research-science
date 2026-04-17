#!/usr/bin/env python3
"""
Optimized 8,100 experiment configuration
- GreedyLR: factor=0.9, min_lr=1e-5, patience=[1,10]
- All model types, all noise types, all problem variants
- MPS GPU acceleration support
"""

import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append('/Users/subshrey/Projects/greedylr_research')
from comprehensive_scheduler_experiment import ExperimentConfig

# Enable MPS if available
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🚀 Using MPS (Apple GPU) acceleration")
else:
    device = torch.device("cpu")
    print("💻 Using CPU (no GPU acceleration)")

def generate_optimized_configs():
    """Generate 8,100 optimized experiment configurations"""
    print("🏗️  Generating optimized experiment configurations...")
    
    configs = []
    config_id = 0
    
    # FIXED PARAMETERS (Optimal values)
    greedy_factor = 0.9
    greedy_min_lr = 1e-5
    greedy_patience_values = [1, 10]  # Aggressive vs Conservative
    
    # ALL MODEL TYPES (Comprehensive coverage)
    model_types = [
        'quadratic', 'rosenbrock', 'rastrigin', 'ackley',  # Analytical
        'neural_simple', 'neural_resnet', 'neural_conv', 'neural_attention',  # Basic neural
        'neural_vit', 'neural_deep_transformer', 'neural_wide_transformer', 'neural_multi_head'  # Advanced neural
    ]
    
    # ALL NOISE TYPES (Full robustness testing)
    noise_types = [
        'periodic_spike', 'random_spike', 'plateau', 'burst', 'adversarial',
        'oscillatory', 'gaussian', 'drift', 'none'
    ]
    
    # NOISE STRENGTHS
    noise_strengths = [0.1, 0.5, 1.0]  # Low, Medium, High
    
    # PROBLEM VARIANTS (Per model type)
    problem_variants = {
        'quadratic': ['well_conditioned', 'ill_conditioned', 'very_ill_conditioned', 'adversarial', 'sparse'],
        'rosenbrock': ['standard', 'extended', 'noisy', 'constrained', 'multi_objective'],
        'rastrigin': ['standard', 'shifted', 'rotated', 'scaled', 'hybrid'],
        'ackley': ['standard', 'extended', 'shifted', 'rotated', 'modified'],
        'neural_simple': ['easy', 'normal', 'hard', 'dropout', 'batch_norm'],
        'neural_resnet': ['shallow', 'deep', 'wide', 'bottleneck', 'dense'],
        'neural_conv': ['standard', 'deep', 'wide', 'separable', 'dilated'],
        'neural_attention': ['single_head', 'multi_head', 'self_attention', 'cross_attention', 'sparse'],
        'neural_vit': ['tiny', 'small', 'base', 'large', 'patch_variants'],
        'neural_deep_transformer': ['6_layer', '12_layer', '18_layer', '24_layer', 'adaptive_depth'],
        'neural_wide_transformer': ['256_dim', '512_dim', '1024_dim', '2048_dim', 'adaptive_width'],
        'neural_multi_head': ['4_heads', '8_heads', '16_heads', '32_heads', 'mixed_heads']
    }
    
    print("🎯 EXPERIMENT CONFIGURATION:")
    print(f"   • GreedyLR: factor={greedy_factor}, min_lr={greedy_min_lr}, patience={greedy_patience_values}")
    print(f"   • Model types: {len(model_types)}")
    print(f"   • Noise types: {len(noise_types)} × {len(noise_strengths)} strengths")
    print(f"   • Problem variants: ~5 per model type")
    print(f"   • Training steps: 200 (optimized)")
    print(f"   • Device: {device}")
    print()
    
    # Generate all experiment configurations
    for model_type in model_types:
        variants = problem_variants.get(model_type, ['standard'])
        
        for variant in variants:
            for noise_type in noise_types:
                for strength in noise_strengths:
                    
                    # GREEDY LR EXPERIMENTS (2 patience values)
                    for patience in greedy_patience_values:
                        scheduler_params = {
                            'min_lr': greedy_min_lr,
                            'patience': patience,
                            'factor': greedy_factor
                        }
                        
                        config = ExperimentConfig(
                            model_type=model_type,
                            scheduler_type='greedy',
                            noise_type=noise_type,
                            noise_strength=strength,
                            problem_variant=variant,
                            total_steps=200,  # Optimized for speed
                            scheduler_params=scheduler_params
                        )
                        configs.append(config)
                        config_id += 1
                    
                    # OTHER SCHEDULER EXPERIMENTS
                    for scheduler_type in ['cosine', 'cosine_restarts', 'exponential']:
                        
                        if scheduler_type == 'cosine':
                            scheduler_params = {'T_max': 200, 'eta_min': greedy_min_lr}
                        elif scheduler_type == 'cosine_restarts':
                            scheduler_params = {'T_0': 50, 'T_mult': 2.0, 'eta_min': greedy_min_lr}
                        else:  # exponential
                            scheduler_params = {'gamma': 0.95}
                        
                        config = ExperimentConfig(
                            model_type=model_type,
                            scheduler_type=scheduler_type,
                            noise_type=noise_type,
                            noise_strength=strength,
                            problem_variant=variant,
                            total_steps=200,
                            scheduler_params=scheduler_params
                        )
                        configs.append(config)
                        config_id += 1
    
    # Shuffle for randomized execution order
    np.random.shuffle(configs)
    
    print("✅ Configuration Summary:")
    total_configs = len(configs)
    greedy_configs = sum(1 for c in configs if c.scheduler_type == 'greedy')
    other_configs = total_configs - greedy_configs
    
    print(f"   • Total experiments: {total_configs:,}")
    print(f"   • GreedyLR: {greedy_configs:,} ({greedy_configs/total_configs*100:.1f}%)")
    print(f"   • Others: {other_configs:,} ({other_configs/total_configs*100:.1f}%)")
    print(f"   • Expected runtime: ~{total_configs*1.25/3600:.1f} hours with MPS")
    print()
    
    return configs

def main():
    """Test the optimized configuration"""
    configs = generate_optimized_configs()
    
    # Show sample configurations
    print("📋 Sample configurations:")
    for i, config in enumerate(configs[:5]):
        print(f"   {i+1}. {config.scheduler_type} on {config.model_type} with {config.noise_type}")
    
    return configs

if __name__ == "__main__":
    main()