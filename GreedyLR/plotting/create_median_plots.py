#!/usr/bin/env python3
"""
Create median-based plots and update analysis
"""
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

def create_median_based_plots():
    """Create plots using median instead of mean"""
    print("📊 Creating median-based performance plots...")
    
    with open('robust_results.json', 'r') as f:
        results = json.load(f)
    
    processed_data = []
    for result in results:
        if isinstance(result, dict):
            metrics = result.get('metrics', {})
            if isinstance(metrics, dict):
                model_type = result.get('model_type', 'unknown')
                
                # Categorize architectures
                if 'neural' in model_type.lower():
                    category = 'neural'
                elif model_type in ['quadratic', 'rosenbrock', 'rastrigin', 'ackley']:
                    category = 'analytical'
                else:
                    category = 'unknown'
                
                processed_data.append({
                    'model_type': model_type,
                    'category': category,
                    'scheduler_type': result.get('scheduler_type', 'unknown'),
                    'noise_type': result.get('noise_type', 'unknown'),
                    'final_loss': metrics.get('final_loss', float('inf'))
                })
    
    df = pd.DataFrame(processed_data)
    df = df[df['final_loss'] != float('inf')]
    df = df[df['scheduler_type'] != 'unknown']
    df = df[df['category'] != 'unknown']
    
    # Create overall performance plot (median-based)
    fig, ax = plt.subplots(figsize=(8, 5))
    
    performance = df.groupby('scheduler_type')['final_loss'].agg(['median', 'count']).sort_values('median')
    
    colors = ['#2E8B57' if 'greedy' in s else '#708090' for s in performance.index]
    bars = ax.bar(range(len(performance)), performance['median'], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.set_xticks(range(len(performance)))
    ax.set_xticklabels([s.replace('_', ' ').title() for s in performance.index])
    ax.set_ylabel('Median Final Loss')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Add sample size annotations
    for i, (bar, count) in enumerate(zip(bars, performance['count'])):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() * 1.2,
                f'n={count}', ha='center', va='bottom', fontsize=9, color='gray')
    
    plt.tight_layout()
    plt.savefig('final_plots/figure_1_median_performance.png', dpi=300, bbox_inches='tight')
    plt.savefig('final_plots/figure_1_median_performance.pdf', bbox_inches='tight')
    plt.close()
    
    # Create analytical vs neural comparison (median-based)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Analytical functions
    analytical_data = df[df['category'] == 'analytical']
    analytical_perf = analytical_data.groupby('scheduler_type')['final_loss'].median().sort_values()
    
    colors = ['#2E8B57' if 'greedy' in s else '#708090' for s in analytical_perf.index]
    bars1 = ax1.bar(range(len(analytical_perf)), analytical_perf.values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax1.set_xticks(range(len(analytical_perf)))
    ax1.set_xticklabels([s.replace('_', ' ').title() for s in analytical_perf.index], rotation=45, ha='right')
    ax1.set_ylabel('Median Final Loss')
    ax1.set_title('(A) Analytical Functions', fontweight='bold')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Neural networks
    neural_data = df[df['category'] == 'neural']
    neural_perf = neural_data.groupby('scheduler_type')['final_loss'].median().sort_values()
    
    colors = ['#2E8B57' if 'greedy' in s else '#708090' for s in neural_perf.index]
    bars2 = ax2.bar(range(len(neural_perf)), neural_perf.values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax2.set_xticks(range(len(neural_perf)))
    ax2.set_xticklabels([s.replace('_', ' ').title() for s in neural_perf.index], rotation=45, ha='right')
    ax2.set_ylabel('Median Final Loss')
    ax2.set_title('(B) Neural Networks', fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('final_plots/figure_2_median_analytical_neural.png', dpi=300, bbox_inches='tight')
    plt.savefig('final_plots/figure_2_median_analytical_neural.pdf', bbox_inches='tight')
    plt.close()
    
    print("✅ Median-based plots created")
    
    # Return performance data for README
    return performance, analytical_perf, neural_perf

def main():
    Path('final_plots').mkdir(exist_ok=True)
    performance_data = create_median_based_plots()
    return performance_data

if __name__ == "__main__":
    main()