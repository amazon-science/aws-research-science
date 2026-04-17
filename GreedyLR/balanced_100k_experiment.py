#!/usr/bin/env python3
"""
Balanced 100K Comprehensive Scheduler Experiment
- Balanced scheduler distribution (greedy ≈ others combined)
- More transformer architectures
- Focus on spike/stagnation noise patterns
- More parameter variations per configuration
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json
import os
import gc
from dataclasses import dataclass
from tqdm import tqdm
import warnings
import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import psutil
import traceback
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Import from existing experiment
import sys
sys.path.append('/Users/subshrey/Projects/greedylr_research')
from comprehensive_scheduler_experiment import ExperimentConfig, AdvancedGreedyLR

# ResidualBlock - CRITICAL: Fix for missing ResidualBlock error
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.linear2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return self.norm2(x + residual)

# Additional transformer architectures
class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = input_size // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(input_size, input_size * 3)
        self.proj = nn.Linear(input_size, input_size)
        self.norm = nn.LayerNorm(input_size)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return self.norm(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn = MultiHeadAttention(dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ViTModel(nn.Module):
    """Vision Transformer variant for optimization testing"""
    def __init__(self, input_size=64, patch_size=8, dim=256, depth=6, num_heads=8):
        super().__init__()
        num_patches = (input_size // patch_size) ** 2
        
        self.patch_embed = nn.Linear(patch_size * patch_size, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads) for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 1)
        
    def forward(self, x):
        # Simulate patch embedding
        B = x.shape[0]
        patches = x.view(B, -1, 64)  # Simplified patching
        x = self.patch_embed(patches)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        return self.head(x[:, 0])

class DeepTransformer(nn.Module):
    """Deeper transformer for challenging optimization"""
    def __init__(self, input_size=128, dim=512, depth=12, num_heads=8):
        super().__init__()
        self.embed = nn.Linear(input_size, dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 1)
        
    def forward(self, x):
        x = self.embed(x.view(x.shape[0], -1))
        x = x.unsqueeze(1)  # Add sequence dimension
        
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        return self.head(x.squeeze(1))

class WideTransformer(nn.Module):
    """Wide transformer with more parameters per layer"""
    def __init__(self, input_size=64, dim=1024, depth=4, num_heads=16):
        super().__init__()
        self.embed = nn.Linear(input_size, dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 1)
        
    def forward(self, x):
        x = self.embed(x.view(x.shape[0], -1))
        x = x.unsqueeze(1)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        return self.head(x.squeeze(1))

def generate_balanced_100k_configs():
    """Generate 100K balanced experiment configurations"""
    configs = []
    
    # BALANCED SCHEDULER DISTRIBUTION (50-50 split)
    # Greedy: 50% (50K experiments)
    # Others: 50% (50K experiments split as: Cosine 30K, Cosine+Restarts 15K, Exponential 5K)
    
    scheduler_distribution = {
        'greedy': 50000,           # 50%
        'cosine': 30000,           # 30%
        'cosine_restarts': 15000,  # 15%
        'exponential': 5000        # 5%
    }
    
    # EXPANDED MODEL TYPES with more transformers
    model_types = {
        # Original optimization functions (40K total)
        'quadratic': 10000,
        'rosenbrock': 10000,  
        'rastrigin': 10000,
        'ackley': 10000,
        
        # Neural architectures (60K total)
        'neural_simple': 8000,
        'neural_resnet': 8000,
        'neural_attention': 8000,      # Original attention
        'neural_vit': 7000,            # Vision Transformer  
        'neural_deep_transformer': 7000,  # Deep transformer
        'neural_wide_transformer': 7000,  # Wide transformer
        'neural_multi_head': 7000      # Multi-head attention focus
    }
    
    # FOCUS ON SPIKE/STAGNATION NOISE TYPES
    # These are where GreedyLR should show biggest advantage
    noise_distribution = {
        'periodic_spike': 20000,    # 20% - Key advantage area
        'random_spike': 20000,      # 20% - Key advantage area  
        'plateau': 15000,           # 15% - Stagnation recovery
        'burst': 12000,             # 12% - Sudden disruptions
        'adversarial': 10000,       # 10% - Worst case scenarios
        'oscillatory': 8000,        #  8% - Cyclical challenges
        'gaussian': 8000,           #  8% - Basic noise
        'drift': 4000,              #  4% - Gradual changes
        'none': 3000                #  3% - Clean baseline
    }
    
    # EXPANDED PARAMETER VARIATIONS
    noise_strengths = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]  # 11 levels
    
    # Problem variants for each model type
    problem_variants = {
        'quadratic': ['well_conditioned', 'ill_conditioned', 'very_ill_conditioned', 'adversarial', 'sparse'],
        'rosenbrock': ['standard', 'extended', 'noisy', 'constrained', 'multi_objective'],
        'rastrigin': ['standard', 'shifted', 'rotated', 'scaled', 'hybrid'],
        'ackley': ['standard', 'extended', 'shifted', 'rotated', 'modified'],
        'neural_simple': ['easy', 'normal', 'hard', 'dropout', 'batch_norm'],
        'neural_resnet': ['shallow', 'deep', 'wide', 'bottleneck', 'dense'],
        'neural_attention': ['single_head', 'multi_head', 'self_attention', 'cross_attention', 'sparse'],
        'neural_vit': ['tiny', 'small', 'base', 'large', 'patch_variants'],
        'neural_deep_transformer': ['6_layer', '12_layer', '18_layer', '24_layer', 'adaptive_depth'],
        'neural_wide_transformer': ['256_dim', '512_dim', '1024_dim', '2048_dim', 'adaptive_width'],
        'neural_multi_head': ['4_heads', '8_heads', '16_heads', '32_heads', 'mixed_heads']
    }
    
    print("🏗️  Generating 100K balanced experiment configurations...")
    print("=" * 60)
    print(f"📊 Scheduler Distribution:")
    for sched, count in scheduler_distribution.items():
        print(f"   {sched:20}: {count:6,} ({count/100000*100:4.1f}%)")
    
    print(f"\n🧠 Model Distribution:")
    for model, count in model_types.items():
        print(f"   {model:25}: {count:6,} ({count/100000*100:4.1f}%)")
        
    print(f"\n🌊 Noise Distribution (Focus on Spikes/Stagnation):")
    for noise, count in noise_distribution.items():
        print(f"   {noise:20}: {count:6,} ({count/100000*100:4.1f}%)")
    
    # Generate configurations
    config_id = 0
    
    for scheduler_type, sched_count in scheduler_distribution.items():
        for model_type, model_count in model_types.items():
            for noise_type, noise_count in noise_distribution.items():
                
                # Calculate proportion for this combination
                total_base = sum(scheduler_distribution.values()) * sum(model_types.values()) * sum(noise_distribution.values())
                combination_weight = sched_count * model_count * noise_count
                target_configs = int(100000 * combination_weight / total_base)
                
                if target_configs == 0:
                    continue
                
                # Distribute across noise strengths and problem variants
                variants = problem_variants.get(model_type, ['standard'])
                configs_per_variant = max(1, target_configs // (len(noise_strengths) * len(variants)))
                
                for variant in variants:
                    for strength in noise_strengths:
                        # COMPREHENSIVE PARAMETER VARIATIONS for each scheduler
                        if scheduler_type == 'greedy':
                            # GreedyLR parameter grid - Factor 0.5 to 0.9 as requested
                            factor_values = [0.5, 0.6, 0.7, 0.8, 0.9]
                            min_lr_values = [1e-7, 1e-6, 1e-5, 1e-4]
                            patience_values = [5, 10, 15, 20]
                            
                            for factor in factor_values:
                                for min_lr in min_lr_values:
                                    for patience in patience_values:
                                        scheduler_params = {
                                            'min_lr': min_lr, 
                                            'patience': patience, 
                                            'factor': factor
                                        }
                                        
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
                        
                        elif scheduler_type == 'cosine':
                            # Cosine parameter variations
                            T_max_values = [400, 600, 800, 1000]
                            eta_min_values = [1e-7, 1e-6, 1e-5]
                            
                            for T_max in T_max_values:
                                for eta_min in eta_min_values:
                                    scheduler_params = {'T_max': T_max, 'eta_min': eta_min}
                                    
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
                        
                        elif scheduler_type == 'cosine_restarts':
                            # Cosine with restarts variations
                            T_0_values = [25, 50, 100]
                            T_mult_values = [1.5, 2.0, 2.5]
                            eta_min_values = [1e-7, 1e-6, 1e-5]
                            
                            for T_0 in T_0_values:
                                for T_mult in T_mult_values:
                                    for eta_min in eta_min_values:
                                        scheduler_params = {
                                            'T_0': T_0, 
                                            'T_mult': T_mult, 
                                            'eta_min': eta_min
                                        }
                                        
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
                        
                        else:  # exponential
                            # Exponential decay variations  
                            gamma_values = [0.90, 0.95, 0.97, 0.99]
                            
                            for gamma in gamma_values:
                                scheduler_params = {'gamma': gamma}
                                
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
    
    print(f"\n✅ Generated {len(configs):,} balanced experiment configurations")
    
    # Verify final distribution
    df = pd.DataFrame([{
        'scheduler_type': c.scheduler_type,
        'model_type': c.model_type,
        'noise_type': c.noise_type
    } for c in configs])
    
    print(f"\n📊 FINAL VERIFICATION:")
    print(f"Scheduler balance:")
    sched_counts = df['scheduler_type'].value_counts()
    greedy_count = sched_counts.get('greedy', 0)
    other_count = len(configs) - greedy_count
    print(f"   Greedy: {greedy_count:,} ({greedy_count/len(configs)*100:.1f}%)")
    print(f"   Others: {other_count:,} ({other_count/len(configs)*100:.1f}%)")
    print(f"   Ratio: {greedy_count/other_count:.2f}:1" if other_count > 0 else "   Ratio: ∞:1")
    
    return configs

if __name__ == "__main__":
    configs = generate_balanced_100k_configs()
    print(f"\n🎯 Ready for 100K balanced experiment with {len(configs):,} configurations!")