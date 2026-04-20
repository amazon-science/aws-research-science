#!/usr/bin/env python3
"""
Clean Professional Plots for Academic Publication
- No unprofessional text bubbles or arrows
- Clean formatting suitable for journals
- Accurate data analysis separating analytical vs neural
"""
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set professional academic style
plt.style.use('default')
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'figure.figsize': (8, 5),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.linewidth': 1,
    'lines.linewidth': 2,
    'axes.grid': True,
    'grid.alpha': 0.3
})

def load_and_analyze_data():
    """Load data and perform careful analysis"""
    print("📊 Loading and carefully analyzing experimental data...")
    
    with open('robust_results.json', 'r') as f:
        results = json.load(f)
    
    processed_data = []
    for result in results:
        if isinstance(result, dict):
            metrics = result.get('metrics', {})
            if isinstance(metrics, dict):
                model_type = result.get('model_type', 'unknown')
                
                # Careful categorization
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
                    'final_loss': metrics.get('final_loss', float('inf')),
                    'lr_changes': metrics.get('lr_changes', 0),
                    'losses': result.get('losses', [])
                })
    
    df = pd.DataFrame(processed_data)
    
    # Remove invalid data
    df = df[df['final_loss'] != float('inf')]
    df = df[df['scheduler_type'] != 'unknown']
    df = df[df['category'] != 'unknown']
    
    print(f"✅ Analyzed {len(df)} valid experiments")
    print(f"📊 Analytical functions: {len(df[df['category'] == 'analytical'])}")
    print(f"🧠 Neural networks: {len(df[df['category'] == 'neural'])}")
    
    # Print detailed breakdown to understand the data
    print("\nDetailed Analysis:")
    print("Overall performance by scheduler:")
    overall_stats = df.groupby('scheduler_type')['final_loss'].agg(['mean', 'median', 'count'])
    print(overall_stats)
    
    print("\nAnalytical functions performance:")
    analytical_stats = df[df['category'] == 'analytical'].groupby('scheduler_type')['final_loss'].agg(['mean', 'median', 'count'])
    print(analytical_stats)
    
    print("\nNeural networks performance:")
    neural_stats = df[df['category'] == 'neural'].groupby('scheduler_type')['final_loss'].agg(['mean', 'median', 'count'])
    print(neural_stats)
    
    return df

def create_figure_1_overall_clean():
    """Figure 1: Clean Overall Performance Comparison"""
    print("📊 Creating Figure 1: Clean Overall Performance")
    
    df = load_and_analyze_data()
    
    # Calculate performance metrics
    performance = df.groupby('scheduler_type')['final_loss'].agg(['mean', 'std', 'count'])
    performance = performance.sort_values('mean')
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Simple, clean bars
    colors = ['#2E8B57' if 'greedy' in s else '#708090' for s in performance.index]
    bars = ax.bar(range(len(performance)), performance['mean'], 
                 yerr=performance['std'], capsize=5,
                 color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Clean formatting
    ax.set_xticks(range(len(performance)))
    ax.set_xticklabels([s.replace('_', ' ').title() for s in performance.index])
    ax.set_ylabel('Mean Final Loss')
    ax.set_yscale('log')
    
    # Add sample size annotations (professional)
    for i, (bar, count) in enumerate(zip(bars, performance['count'])):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() * 1.2,
                f'n={count}', ha='center', va='bottom', fontsize=9, color='gray')
    
    plt.tight_layout()
    plt.savefig('final_plots/figure_1_overall_performance.png', dpi=300, bbox_inches='tight')
    plt.savefig('final_plots/figure_1_overall_performance.pdf', bbox_inches='tight')
    plt.close()
    print("✅ Figure 1 saved")

def create_figure_2_analytical_vs_neural():
    """Figure 2: Analytical Functions vs Neural Networks - Separate Analysis"""
    print("📊 Creating Figure 2: Analytical vs Neural Comparison")
    
    df = load_and_analyze_data()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Analytical Functions
    analytical_data = df[df['category'] == 'analytical']
    analytical_perf = analytical_data.groupby('scheduler_type')['final_loss'].agg(['mean', 'std'])
    analytical_perf = analytical_perf.sort_values('mean')
    
    colors = ['#2E8B57' if 'greedy' in s else '#708090' for s in analytical_perf.index]
    bars1 = ax1.bar(range(len(analytical_perf)), analytical_perf['mean'], 
                   yerr=analytical_perf['std'], capsize=4,
                   color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax1.set_xticks(range(len(analytical_perf)))
    ax1.set_xticklabels([s.replace('_', ' ').title() for s in analytical_perf.index], 
                       rotation=45, ha='right')
    ax1.set_ylabel('Mean Final Loss')
    ax1.set_title('(A) Analytical Functions', fontweight='bold')
    ax1.set_yscale('log')
    
    # Right: Neural Networks
    neural_data = df[df['category'] == 'neural']
    neural_perf = neural_data.groupby('scheduler_type')['final_loss'].agg(['mean', 'std'])
    neural_perf = neural_perf.sort_values('mean')
    
    colors = ['#2E8B57' if 'greedy' in s else '#708090' for s in neural_perf.index]
    bars2 = ax2.bar(range(len(neural_perf)), neural_perf['mean'], 
                   yerr=neural_perf['std'], capsize=4,
                   color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax2.set_xticks(range(len(neural_perf)))
    ax2.set_xticklabels([s.replace('_', ' ').title() for s in neural_perf.index], 
                       rotation=45, ha='right')
    ax2.set_ylabel('Mean Final Loss')
    ax2.set_title('(B) Neural Networks', fontweight='bold')
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('final_plots/figure_2_analytical_vs_neural.png', dpi=300, bbox_inches='tight')
    plt.savefig('final_plots/figure_2_analytical_vs_neural.pdf', bbox_inches='tight')
    plt.close()
    print("✅ Figure 2 saved")

def create_figure_3_noise_analysis():
    """Figure 3: Noise Condition Analysis"""
    print("📊 Creating Figure 3: Noise Analysis")
    
    df = load_and_analyze_data()
    
    # Create noise performance matrix
    noise_perf = df.groupby(['noise_type', 'scheduler_type'])['final_loss'].mean().unstack()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Clean heatmap
    im = ax.imshow(np.log10(noise_perf.values), cmap='RdYlBu_r', aspect='auto')
    
    # Set labels
    ax.set_xticks(np.arange(len(noise_perf.columns)))
    ax.set_yticks(np.arange(len(noise_perf.index)))
    ax.set_xticklabels([col.replace('_', ' ').title() for col in noise_perf.columns])
    ax.set_yticklabels([idx.replace('_', ' ').title() for idx in noise_perf.index])
    
    # Rotate labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Log₁₀(Final Loss)', rotation=270, labelpad=20)
    
    ax.set_xlabel('Learning Rate Scheduler')
    ax.set_ylabel('Noise Condition')
    
    plt.tight_layout()
    plt.savefig('final_plots/figure_3_noise_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('final_plots/figure_3_noise_analysis.pdf', bbox_inches='tight')
    plt.close()
    print("✅ Figure 3 saved")

def create_performance_summary_tables():
    """Create detailed performance tables"""
    print("📊 Creating Performance Summary Tables")
    
    df = load_and_analyze_data()
    
    # Overall summary
    overall_summary = df.groupby('scheduler_type').agg({
        'final_loss': ['mean', 'median', 'std', 'count']
    }).round(4)
    
    # Analytical vs Neural breakdown
    analytical_summary = df[df['category'] == 'analytical'].groupby('scheduler_type').agg({
        'final_loss': ['mean', 'median', 'std', 'count']
    }).round(4)
    
    neural_summary = df[df['category'] == 'neural'].groupby('scheduler_type').agg({
        'final_loss': ['mean', 'median', 'std', 'count']
    }).round(4)
    
    # Clean vs Noisy conditions
    clean_summary = df[df['noise_type'] == 'none'].groupby('scheduler_type').agg({
        'final_loss': ['mean', 'median', 'std', 'count']
    }).round(4)
    
    noisy_summary = df[df['noise_type'] != 'none'].groupby('scheduler_type').agg({
        'final_loss': ['mean', 'median', 'std', 'count']
    }).round(4)
    
    # Save comprehensive tables
    with open('final_plots/comprehensive_performance_analysis.txt', 'w') as f:
        f.write("COMPREHENSIVE PERFORMANCE ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. OVERALL PERFORMANCE ACROSS ALL EXPERIMENTS\n")
        f.write("-" * 50 + "\n")
        f.write(overall_summary.to_string())
        f.write("\n\n")
        
        f.write("2. ANALYTICAL FUNCTIONS PERFORMANCE\n")
        f.write("-" * 50 + "\n")
        f.write(analytical_summary.to_string())
        f.write("\n\n")
        
        f.write("3. NEURAL NETWORKS PERFORMANCE\n")
        f.write("-" * 50 + "\n")
        f.write(neural_summary.to_string())
        f.write("\n\n")
        
        f.write("4. CLEAN CONDITIONS (NO NOISE)\n")
        f.write("-" * 50 + "\n")
        f.write(clean_summary.to_string())
        f.write("\n\n")
        
        f.write("5. NOISY CONDITIONS\n")
        f.write("-" * 50 + "\n")
        f.write(noisy_summary.to_string())
        f.write("\n\n")
        
        # Add analysis notes
        f.write("ANALYSIS NOTES:\n")
        f.write("-" * 20 + "\n")
        f.write("- Mean: Average final loss across experiments\n")
        f.write("- Median: Median final loss (robust to outliers)\n")
        f.write("- Std: Standard deviation of final loss\n")
        f.write("- Count: Number of experiments in each category\n")
    
    print("✅ Comprehensive analysis tables saved")
    
    return overall_summary, analytical_summary, neural_summary, clean_summary, noisy_summary

def main():
    """Generate clean, professional plots and analysis"""
    Path('final_plots').mkdir(exist_ok=True)
    
    print("🚀 Generating clean professional materials...")
    print("=" * 70)
    
    create_figure_1_overall_clean()
    create_figure_2_analytical_vs_neural()
    create_figure_3_noise_analysis()
    summaries = create_performance_summary_tables()
    
    print("=" * 70)
    print("🎉 Clean professional materials complete!")
    print("📁 Saved to: final_plots/")
    print("✅ No unprofessional annotations - ready for academic publication!")

if __name__ == "__main__":
    main()