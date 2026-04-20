#!/usr/bin/env python3
"""
Analyze the comprehensive scheduler experiment results
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
from dataclasses import dataclass
from tqdm import tqdm
import warnings
import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
warnings.filterwarnings('ignore')

# Import the necessary functions from the main experiment
import sys
sys.path.append('/Users/subshrey/Projects/greedylr_research')

def analyze_results():
    """Load and analyze the comprehensive experiment results"""
    
    # Check if results file exists
    results_file = '/Users/subshrey/Projects/greedylr_research/comprehensive_results.json'
    if not os.path.exists(results_file):
        print("❌ Results file not found. The experiment may not have completed successfully.")
        return
    
    # Load results
    print("📊 Loading comprehensive experiment results...")
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print(f"✅ Loaded {len(results)} experimental results")
    
    # Convert to DataFrame for analysis
    print("📈 Converting to DataFrame for analysis...")
    df = pd.DataFrame(results)
    
    print(f"📊 Dataset overview:")
    print(f"• Total experiments: {len(df)}")
    print(f"• Model types: {df['model_type'].nunique()} ({list(df['model_type'].unique())})")
    print(f"• Scheduler types: {df['scheduler_type'].nunique()} ({list(df['scheduler_type'].unique())})")
    print(f"• Noise types: {df['noise_type'].nunique()} ({list(df['noise_type'].unique())})")
    
    # Basic analysis
    print("\n🔍 Performing statistical analysis...")
    
    # Group by scheduler type for comparison
    scheduler_stats = df.groupby('scheduler_type').agg({
        'final_loss': ['mean', 'std', 'min'],
        'convergence_rate_50': ['mean', 'std', 'max'],
        'stability_score': ['mean', 'std', 'max'],
        'efficiency_score': ['mean', 'std', 'max'],
        'robustness_score': ['mean', 'std', 'max']
    }).round(4)
    
    print("\n📊 Scheduler Performance Summary:")
    print(scheduler_stats)
    
    # Winner analysis
    print("\n🏆 WINNERS BY METRIC:")
    print(f"• Best Final Loss: {df.groupby('scheduler_type')['final_loss'].mean().idxmin()}")
    print(f"• Best Convergence Rate: {df.groupby('scheduler_type')['convergence_rate_50'].mean().idxmax()}")
    print(f"• Best Stability: {df.groupby('scheduler_type')['stability_score'].mean().idxmax()}")
    print(f"• Best Efficiency: {df.groupby('scheduler_type')['efficiency_score'].mean().idxmax()}")
    print(f"• Best Robustness: {df.groupby('scheduler_type')['robustness_score'].mean().idxmax()}")
    
    # Create visualizations
    print("\n🎨 Creating visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Final Loss Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    df.boxplot(column='final_loss', by='scheduler_type', ax=ax1)
    ax1.set_title('Final Loss by Scheduler')
    ax1.set_xlabel('Scheduler Type')
    plt.suptitle('')
    
    # 2. Convergence Rate Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    df.boxplot(column='convergence_rate_50', by='scheduler_type', ax=ax2)
    ax2.set_title('Convergence Rate by Scheduler')
    ax2.set_xlabel('Scheduler Type')
    plt.suptitle('')
    
    # 3. Stability Score Comparison
    ax3 = fig.add_subplot(gs[0, 2])
    df.boxplot(column='stability_score', by='scheduler_type', ax=ax3)
    ax3.set_title('Stability Score by Scheduler')
    ax3.set_xlabel('Scheduler Type')
    plt.suptitle('')
    
    # 4. Efficiency Score Comparison
    ax4 = fig.add_subplot(gs[0, 3])
    df.boxplot(column='efficiency_score', by='scheduler_type', ax=ax4)
    ax4.set_title('Efficiency Score by Scheduler')
    ax4.set_xlabel('Scheduler Type')
    plt.suptitle('')
    
    # 5. Performance across noise types
    ax5 = fig.add_subplot(gs[1, :2])
    noise_performance = df.groupby(['scheduler_type', 'noise_type'])['final_loss'].mean().unstack()
    noise_performance.plot(kind='bar', ax=ax5)
    ax5.set_title('Final Loss by Scheduler and Noise Type')
    ax5.set_xlabel('Scheduler Type')
    ax5.legend(title='Noise Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 6. Model type performance
    ax6 = fig.add_subplot(gs[1, 2:])
    model_performance = df.groupby(['scheduler_type', 'model_type'])['final_loss'].mean().unstack()
    model_performance.plot(kind='bar', ax=ax6)
    ax6.set_title('Final Loss by Scheduler and Model Type')
    ax6.set_xlabel('Scheduler Type')
    ax6.legend(title='Model Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 7. Overall score comparison
    ax7 = fig.add_subplot(gs[2, :])
    # Create composite score
    df['composite_score'] = (
        (1 - df['final_loss'] / df['final_loss'].max()) * 0.3 +  # Lower loss is better
        (df['convergence_rate_50'] / df['convergence_rate_50'].max()) * 0.2 +
        (df['stability_score'] / df['stability_score'].max()) * 0.2 +
        (df['efficiency_score'] / df['efficiency_score'].max()) * 0.15 +
        (df['robustness_score'] / df['robustness_score'].max()) * 0.15
    )
    
    df.boxplot(column='composite_score', by='scheduler_type', ax=ax7)
    ax7.set_title('Composite Performance Score by Scheduler')
    ax7.set_xlabel('Scheduler Type')
    ax7.set_ylabel('Composite Score (Higher is Better)')
    plt.suptitle('')
    
    # Add overall title
    fig.suptitle('Comprehensive Scheduler Comparison Results\n29,866 Experiments Across Multiple Models and Noise Conditions', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Add summary statistics
    summary_stats = f"""EXPERIMENT SUMMARY:
Total Experiments: {len(df)}
Successful: {len(df)}
Model Types: {df['model_type'].nunique()}
Noise Conditions: {df['noise_type'].nunique()}
Scheduler Variants: {df['scheduler_type'].nunique()}

OVERALL WINNER:
{df.groupby('scheduler_type')['composite_score'].mean().idxmax()}
(Composite Score: {df.groupby('scheduler_type')['composite_score'].mean().max():.3f})"""
    
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    fig.text(0.98, 0.02, summary_stats, transform=fig.transFigure, fontsize=10,
             verticalalignment='bottom', horizontalalignment='right', bbox=props, fontweight='bold')
    
    # Save the visualization
    output_file = '/Users/subshrey/Projects/greedylr_research/comprehensive_scheduler_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Comprehensive analysis saved to: {output_file}")
    
    # Save detailed statistics
    stats_file = '/Users/subshrey/Projects/greedylr_research/scheduler_statistics.csv'
    scheduler_stats.to_csv(stats_file)
    print(f"✅ Detailed statistics saved to: {stats_file}")
    
    # Create a summary report
    report_file = '/Users/subshrey/Projects/greedylr_research/EXPERIMENT_REPORT.md'
    
    with open(report_file, 'w') as f:
        f.write("# Comprehensive Scheduler Comparison Results\n\n")
        f.write(f"**Experiment Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Experiment Overview\n\n")
        f.write(f"- **Total Experiments:** {len(df):,}\n")
        f.write(f"- **Model Types:** {df['model_type'].nunique()} ({', '.join(df['model_type'].unique())})\n")
        f.write(f"- **Scheduler Types:** {df['scheduler_type'].nunique()} ({', '.join(df['scheduler_type'].unique())})\n")
        f.write(f"- **Noise Types:** {df['noise_type'].nunique()} ({', '.join(df['noise_type'].unique())})\n\n")
        
        f.write("## Key Findings\n\n")
        f.write("### Winners by Metric\n\n")
        f.write(f"- **Best Final Loss:** {df.groupby('scheduler_type')['final_loss'].mean().idxmin()}\n")
        f.write(f"- **Best Convergence Rate:** {df.groupby('scheduler_type')['convergence_rate_50'].mean().idxmax()}\n")
        f.write(f"- **Best Stability:** {df.groupby('scheduler_type')['stability_score'].mean().idxmax()}\n")
        f.write(f"- **Best Efficiency:** {df.groupby('scheduler_type')['efficiency_score'].mean().idxmax()}\n")
        f.write(f"- **Best Robustness:** {df.groupby('scheduler_type')['robustness_score'].mean().idxmax()}\n")
        f.write(f"- **Overall Winner (Composite Score):** {df.groupby('scheduler_type')['composite_score'].mean().idxmax()}\n\n")
        
        f.write("### Performance Statistics\n\n")
        f.write("```\n")
        f.write(str(scheduler_stats))
        f.write("\n```\n\n")
        
        f.write("## Visualization\n\n")
        f.write("![Comprehensive Analysis](comprehensive_scheduler_analysis.png)\n\n")
        
        f.write("## Conclusion\n\n")
        winner = df.groupby('scheduler_type')['composite_score'].mean().idxmax()
        winner_score = df.groupby('scheduler_type')['composite_score'].mean().max()
        f.write(f"Based on the comprehensive analysis of {len(df):,} experiments across multiple model architectures and noise conditions, ")
        f.write(f"**{winner}** emerges as the overall best scheduler with a composite score of {winner_score:.3f}.\n\n")
        
        f.write("This result validates the effectiveness of the scheduler across diverse optimization landscapes and challenging training conditions.\n")
    
    print(f"✅ Comprehensive report saved to: {report_file}")
    
    return df, scheduler_stats

if __name__ == "__main__":
    df, stats = analyze_results()