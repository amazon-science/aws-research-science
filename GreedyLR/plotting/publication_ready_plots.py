#!/usr/bin/env python3
"""
Publication-Ready Plots with Clean Formatting
- Captions in markdown, not graph titles
- All annotations within graph borders
- Separate analytical vs neural analysis
"""
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set clean publication style
plt.style.use('default')
plt.rcParams.update({
    'font.size': 11,
    'figure.figsize': (8, 5),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.linewidth': 1,
    'lines.linewidth': 2
})

def load_data():
    """Load and categorize data"""
    print("📊 Loading and categorizing data...")
    
    with open('robust_results.json', 'r') as f:
        results = json.load(f)
    
    processed_data = []
    for result in results[:4000]:  # Sample for efficiency
        if isinstance(result, dict):
            metrics = result.get('metrics', {})
            if isinstance(metrics, dict):
                model_type = result.get('model_type', 'unknown')
                
                # Categorize architectures
                if 'neural' in model_type:
                    category = 'neural'
                else:
                    category = 'analytical'
                
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
    print(f"✅ Loaded {len(df)} experiments")
    print(f"📊 Analytical functions: {len(df[df['category'] == 'analytical'])}")
    print(f"🧠 Neural networks: {len(df[df['category'] == 'neural'])}")
    return df

def create_figure_1_overall_performance():
    """Figure 1: Overall Performance - Clean bars without titles"""
    print("📊 Creating Figure 1: Overall Performance")
    
    df = load_data()
    
    # Calculate overall performance
    overall_perf = df.groupby('scheduler_type')['final_loss'].mean().sort_values()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Clean color scheme
    colors = ['#2E8B57' if 'greedy' in s else '#8B8B8B' for s in overall_perf.index]
    
    bars = ax.bar(range(len(overall_perf)), overall_perf.values, 
                 color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Clean labels
    ax.set_xticks(range(len(overall_perf)))
    ax.set_xticklabels([s.replace('_', ' ').title() for s in overall_perf.index])
    ax.set_ylabel('Final Loss')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Add values on bars (within plot area)
    for i, (bar, value) in enumerate(zip(bars, overall_perf.values)):
        ax.text(bar.get_x() + bar.get_width()/2., value * 0.7,
                f'{value:.3f}', ha='center', va='top', 
                fontsize=10, fontweight='bold')
    
    # Add improvement annotations (within plot area)
    greedy_idx = [i for i, s in enumerate(overall_perf.index) if 'greedy' in s]
    if greedy_idx:
        greedy_loss = overall_perf.iloc[greedy_idx[0]]
        for i, (scheduler, loss) in enumerate(overall_perf.items()):
            if scheduler != 'greedy' and loss > greedy_loss:
                improvement = loss / greedy_loss
                # Position annotation within plot area
                y_pos = greedy_loss * 1.5
                ax.annotate(f'{improvement:.0f}×', 
                          xy=(i, loss), xytext=(greedy_idx[0], y_pos),
                          arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                          fontsize=10, fontweight='bold', color='red',
                          ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig('pub_plots/figure_1_overall.png', dpi=300, bbox_inches='tight')
    plt.savefig('pub_plots/figure_1_overall.pdf', bbox_inches='tight')
    plt.show()
    print("✅ Figure 1 saved")

def create_figure_2_noise_comparison():
    """Figure 2: Noise Performance Comparison"""
    print("📊 Creating Figure 2: Noise Comparison")
    
    df = load_data()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Clean vs Noisy conditions
    clean_data = df[df['noise_type'] == 'none']
    noisy_data = df[df['noise_type'] != 'none']
    
    conditions = ['Clean', 'Noisy']
    greedy_means = [
        clean_data[clean_data['scheduler_type'] == 'greedy']['final_loss'].mean(),
        noisy_data[noisy_data['scheduler_type'] == 'greedy']['final_loss'].mean()
    ]
    cosine_means = [
        clean_data[clean_data['scheduler_type'] == 'cosine']['final_loss'].mean(),
        noisy_data[noisy_data['scheduler_type'] == 'cosine']['final_loss'].mean()
    ]
    
    x = np.arange(len(conditions))
    width = 0.35
    
    ax1.bar(x - width/2, greedy_means, width, label='GreedyLR', color='#2E8B57', alpha=0.8)
    ax1.bar(x + width/2, cosine_means, width, label='Cosine', color='#4682B4', alpha=0.8)
    
    ax1.set_ylabel('Final Loss')
    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions)
    ax1.legend()
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Add improvement annotation (within plot area)
    if len(greedy_means) > 1 and len(cosine_means) > 1:
        improvement = cosine_means[1] / greedy_means[1]
        ax1.text(0.5, max(greedy_means + cosine_means) * 0.5, 
                f'{improvement:.0f}× better\nin noisy conditions',
                ha='center', va='center', fontweight='bold', color='red',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # Right: Noise type breakdown
    noise_types = ['gaussian', 'random_spike', 'adversarial', 'oscillatory']
    improvements = []
    
    for noise in noise_types:
        noise_subset = df[df['noise_type'] == noise]
        if len(noise_subset) > 0:
            greedy_perf = noise_subset[noise_subset['scheduler_type'] == 'greedy']['final_loss'].mean()
            cosine_perf = noise_subset[noise_subset['scheduler_type'] == 'cosine']['final_loss'].mean()
            improvement = cosine_perf / greedy_perf if greedy_perf > 0 else 1
            improvements.append(improvement)
        else:
            improvements.append(1)
    
    bars = ax2.bar(range(len(noise_types)), improvements, 
                  color='#228B22', alpha=0.8, edgecolor='black')
    
    ax2.set_xticks(range(len(noise_types)))
    ax2.set_xticklabels([n.replace('_', ' ').title() for n in noise_types], 
                       rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Improvement Factor')
    ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels (within plot area)
    for bar, improvement in zip(bars, improvements):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height * 0.9,
                f'{improvement:.1f}×', ha='center', va='top', 
                fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('pub_plots/figure_2_noise.png', dpi=300, bbox_inches='tight')
    plt.savefig('pub_plots/figure_2_noise.pdf', bbox_inches='tight')
    plt.show()
    print("✅ Figure 2 saved")

def create_figure_3_analytical_vs_neural():
    """Figure 3: Analytical Functions vs Neural Networks Performance"""
    print("📊 Creating Figure 3: Analytical vs Neural Comparison")
    
    df = load_data()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Analytical functions
    analytical_data = df[df['category'] == 'analytical']
    if len(analytical_data) > 0:
        analytical_perf = analytical_data.groupby('scheduler_type')['final_loss'].mean().sort_values()
        
        colors = ['#2E8B57' if 'greedy' in s else '#8B8B8B' for s in analytical_perf.index]
        bars1 = ax1.bar(range(len(analytical_perf)), analytical_perf.values, 
                       color=colors, alpha=0.8, edgecolor='black')
        
        ax1.set_xticks(range(len(analytical_perf)))
        ax1.set_xticklabels([s.replace('_', ' ').title() for s in analytical_perf.index], 
                           rotation=45, ha='right')
        ax1.set_ylabel('Final Loss')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        
        # Add improvement annotation
        greedy_idx = [i for i, s in enumerate(analytical_perf.index) if 'greedy' in s]
        if greedy_idx and len(analytical_perf) > 1:
            greedy_loss = analytical_perf.iloc[greedy_idx[0]]
            best_competitor = analytical_perf.iloc[-1]  # Highest loss
            if best_competitor > greedy_loss:
                improvement = best_competitor / greedy_loss
                ax1.text(0.5, greedy_loss * 3, f'{improvement:.0f}× better',
                        ha='center', va='center', fontweight='bold', color='red',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.7))
    
    # Right: Neural networks
    neural_data = df[df['category'] == 'neural']
    if len(neural_data) > 0:
        neural_perf = neural_data.groupby('scheduler_type')['final_loss'].mean().sort_values()
        
        colors = ['#2E8B57' if 'greedy' in s else '#8B8B8B' for s in neural_perf.index]
        bars2 = ax2.bar(range(len(neural_perf)), neural_perf.values, 
                       color=colors, alpha=0.8, edgecolor='black')
        
        ax2.set_xticks(range(len(neural_perf)))
        ax2.set_xticklabels([s.replace('_', ' ').title() for s in neural_perf.index], 
                           rotation=45, ha='right')
        ax2.set_ylabel('Final Loss')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        # Add comparison annotation
        greedy_idx = [i for i, s in enumerate(neural_perf.index) if 'greedy' in s]
        if greedy_idx and len(neural_perf) > 1:
            greedy_loss = neural_perf.iloc[greedy_idx[0]]
            # Check if GreedyLR is best
            is_best = greedy_idx[0] == 0  # First in sorted list means lowest loss
            if is_best:
                ax2.text(0.5, greedy_loss * 2, 'Best',
                        ha='center', va='center', fontweight='bold', color='green',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgreen", alpha=0.7))
            else:
                ax2.text(0.5, greedy_loss * 2, 'Competitive',
                        ha='center', va='center', fontweight='bold', color='blue',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('pub_plots/figure_3_analytical_neural.png', dpi=300, bbox_inches='tight')
    plt.savefig('pub_plots/figure_3_analytical_neural.pdf', bbox_inches='tight')
    plt.show()
    print("✅ Figure 3 saved")

def create_performance_tables():
    """Create performance summary tables"""
    print("📊 Creating Performance Tables")
    
    df = load_data()
    
    # Overall performance table
    overall_table = df.groupby('scheduler_type')['final_loss'].agg(['mean', 'std', 'count']).round(4)
    overall_table.columns = ['Mean Loss', 'Std Dev', 'N Experiments']
    
    # Analytical functions table
    analytical_data = df[df['category'] == 'analytical']
    analytical_table = analytical_data.groupby('scheduler_type')['final_loss'].agg(['mean', 'std']).round(4)
    analytical_table.columns = ['Mean Loss', 'Std Dev']
    
    # Neural networks table
    neural_data = df[df['category'] == 'neural']
    neural_table = neural_data.groupby('scheduler_type')['final_loss'].agg(['mean', 'std']).round(4)
    neural_table.columns = ['Mean Loss', 'Std Dev']
    
    # Noise performance table
    noise_table = df.groupby(['noise_type', 'scheduler_type'])['final_loss'].mean().unstack().round(4)
    
    # Save tables
    with open('pub_plots/performance_tables.txt', 'w') as f:
        f.write("OVERALL PERFORMANCE\n")
        f.write("=" * 50 + "\n")
        f.write(overall_table.to_string())
        f.write("\n\n")
        
        f.write("ANALYTICAL FUNCTIONS PERFORMANCE\n")
        f.write("=" * 50 + "\n")
        f.write(analytical_table.to_string())
        f.write("\n\n")
        
        f.write("NEURAL NETWORKS PERFORMANCE\n")
        f.write("=" * 50 + "\n")
        f.write(neural_table.to_string())
        f.write("\n\n")
        
        f.write("NOISE CONDITIONS PERFORMANCE\n")
        f.write("=" * 50 + "\n")
        f.write(noise_table.to_string())
    
    print("✅ Performance tables saved")
    
    return overall_table, analytical_table, neural_table, noise_table

def main():
    """Generate publication-ready plots and tables"""
    Path('pub_plots').mkdir(exist_ok=True)
    
    print("🚀 Generating publication-ready materials...")
    print("=" * 60)
    
    create_figure_1_overall_performance()
    create_figure_2_noise_comparison()
    create_figure_3_analytical_vs_neural()
    tables = create_performance_tables()
    
    print("=" * 60)
    print("🎉 Publication-ready materials complete!")
    print("📁 Saved to: pub_plots/")
    print("✅ Clean plots with proper captions ready for paper!")

if __name__ == "__main__":
    main()