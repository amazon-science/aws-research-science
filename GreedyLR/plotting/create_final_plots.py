#!/usr/bin/env python3
"""
Create final essential plots for research paper
"""
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set professional style
plt.style.use('default')
plt.rcParams.update({
    'font.size': 12,
    'font.weight': 'bold',
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300
})

def load_data():
    """Load data efficiently"""
    print("📊 Loading experimental data...")
    
    with open('robust_results.json', 'r') as f:
        results = json.load(f)
    
    processed_data = []
    for result in results[:3000]:  # Sample for efficiency
        if isinstance(result, dict):
            metrics = result.get('metrics', {})
            if isinstance(metrics, dict):
                processed_data.append({
                    'model_type': result.get('model_type', 'unknown'),
                    'scheduler_type': result.get('scheduler_type', 'unknown'),
                    'noise_type': result.get('noise_type', 'unknown'),
                    'final_loss': metrics.get('final_loss', float('inf')),
                    'lr_changes': metrics.get('lr_changes', 0),
                    'losses': result.get('losses', [])
                })
    
    df = pd.DataFrame(processed_data)
    print(f"✅ Loaded {len(df)} experiments")
    return df

def plot_3_recovery_demo():
    """Figure 3: GreedyLR Recovery Demonstration"""
    print("📊 Creating Figure 3: Recovery Demonstration")
    
    df = load_data()
    
    # Find samples with recovery patterns
    spike_data = df[df['noise_type'].isin(['periodic_spike', 'random_spike'])]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('GreedyLR Recovery Mechanism: How It Adapts to Disruptions', 
                 fontsize=18, fontweight='bold')
    
    # Left: Sample trajectory comparison
    greedy_sample = spike_data[
        (spike_data['scheduler_type'] == 'greedy') & 
        (spike_data['losses'].apply(len) > 50)
    ]
    cosine_sample = spike_data[
        (spike_data['scheduler_type'] == 'cosine') & 
        (spike_data['losses'].apply(len) > 50)
    ]
    
    if len(greedy_sample) > 0 and len(cosine_sample) > 0:
        greedy_losses = greedy_sample.iloc[0]['losses'][:100]
        cosine_losses = cosine_sample.iloc[0]['losses'][:100]
        
        steps = range(len(greedy_losses))
        ax1.plot(steps, greedy_losses, color='#2E8B57', 
                linewidth=4, label='GreedyLR: Adapts & Recovers', alpha=0.9)
        ax1.plot(steps[:len(cosine_losses)], cosine_losses, color='#4682B4', 
                linewidth=3, label='Cosine: Fixed Schedule', alpha=0.7, linestyle='--')
        
        ax1.set_xlabel('Training Step', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Loss Value', fontsize=14, fontweight='bold')
        ax1.set_title('A. Loss Recovery After Noise Spikes', fontweight='bold', fontsize=16)
        ax1.legend(fontsize=12)
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        
        # Highlight final improvement
        final_greedy = greedy_losses[-1] if greedy_losses else float('inf')
        final_cosine = cosine_losses[-1] if cosine_losses else float('inf')
        
        if final_cosine > final_greedy:
            improvement = final_cosine / final_greedy
            ax1.annotate(f'Final: {improvement:.0f}× Better Recovery', 
                       xy=(len(greedy_losses)-1, final_greedy), 
                       xytext=(len(greedy_losses)*0.7, final_greedy * 5),
                       arrowprops=dict(arrowstyle='->', color='red', lw=3),
                       fontsize=14, fontweight='bold', color='red',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow"))
    
    # Right: Adaptation frequency by condition
    adaptation_freq = df[df['scheduler_type'] == 'greedy'].groupby('noise_type')['lr_changes'].mean()
    
    bars = ax2.bar(range(len(adaptation_freq)), adaptation_freq.values,
                  color='#2E8B57', alpha=0.8, edgecolor='black', linewidth=2)
    ax2.set_xticks(range(len(adaptation_freq)))
    ax2.set_xticklabels([noise.replace('_', ' ').title() for noise in adaptation_freq.index], 
                       rotation=45, ha='right', fontsize=10)
    ax2.set_ylabel('Average LR Adaptations', fontweight='bold', fontsize=14)
    ax2.set_title('B. GreedyLR Adaptation Frequency', fontweight='bold', fontsize=16)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, freq in zip(bars, adaptation_freq.values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{freq:.1f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('clear_plots/figure_3_recovery_demo.png', dpi=300, bbox_inches='tight')
    plt.savefig('clear_plots/figure_3_recovery_demo.pdf', bbox_inches='tight')
    plt.show()
    print("✅ Figure 3 saved")

def plot_4_statistical_summary():
    """Figure 4: Statistical Summary"""
    print("📊 Creating Figure 4: Statistical Summary")
    
    df = load_data()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Statistical Validation: GreedyLR Performance Evidence', 
                 fontsize=18, fontweight='bold')
    
    # Left: Sample sizes
    sample_sizes = df['scheduler_type'].value_counts()
    colors = ['#2E8B57' if 'greedy' in s else '#D3D3D3' for s in sample_sizes.index]
    
    bars = ax1.bar(range(len(sample_sizes)), sample_sizes.values, 
                  color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax1.set_xticks(range(len(sample_sizes)))
    ax1.set_xticklabels([s.replace('_', ' ').title() for s in sample_sizes.index], 
                       fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Experiments', fontweight='bold', fontsize=14)
    ax1.set_title('A. Statistical Power (Sample Sizes)', fontweight='bold', fontsize=16)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, size in zip(bars, sample_sizes.values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 20,
                f'{size:,}', ha='center', va='bottom', fontweight='bold')
    
    # Right: Performance comparison boxplot
    data_to_plot = []
    labels_to_plot = []
    colors_to_plot = []
    
    for scheduler in ['greedy', 'cosine', 'exponential']:
        if scheduler in df['scheduler_type'].values:
            scheduler_data = df[df['scheduler_type'] == scheduler]['final_loss']
            # Apply log transform for better visualization
            log_data = np.log10(scheduler_data + 1e-10)
            data_to_plot.append(log_data)
            labels_to_plot.append(scheduler.replace('_', ' ').title())
            colors_to_plot.append('#2E8B57' if scheduler == 'greedy' else '#D3D3D3')
    
    bp = ax2.boxplot(data_to_plot, labels=labels_to_plot, patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors_to_plot):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Highlight GreedyLR box
    if labels_to_plot and 'Greedy' in labels_to_plot:
        greedy_idx = labels_to_plot.index('Greedy')
        bp['boxes'][greedy_idx].set_edgecolor('black')
        bp['boxes'][greedy_idx].set_linewidth(3)
    
    ax2.set_ylabel('Log₁₀(Final Loss)', fontweight='bold', fontsize=14)
    ax2.set_title('B. Loss Distribution Comparison', fontweight='bold', fontsize=16)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('clear_plots/figure_4_statistical_summary.png', dpi=300, bbox_inches='tight')
    plt.savefig('clear_plots/figure_4_statistical_summary.pdf', bbox_inches='tight')
    plt.show()
    print("✅ Figure 4 saved")

def main():
    """Generate additional key plots"""
    Path('clear_plots').mkdir(exist_ok=True)
    
    print("🚀 Generating additional clear plots...")
    print("="*50)
    
    plot_3_recovery_demo()
    plot_4_statistical_summary()
    
    print("="*50)
    print("🎉 Additional plots complete!")
    print("📁 All plots saved to: clear_plots/")

if __name__ == "__main__":
    main()