#!/usr/bin/env python3
"""
Fix all visualization issues:
1. Clean dashed lines for percentile bands (one line per scheduler)
2. Add distribution visualization to README
3. Fix label positioning issues
4. Create more individual comparison examples from neural networks
5. Add learning rate trajectory analysis section
"""
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

def create_clean_percentile_bands():
    """Create clean percentile bands with single dashed lines per scheduler"""
    print("📊 Creating clean percentile band visualization...")
    
    with open('robust_results.json', 'r') as f:
        results = json.load(f)
    
    # Collect spike data
    spike_data = {
        'greedy': [],
        'cosine': [],
        'cosine_restarts': [],
        'exponential': []
    }
    
    for result in results:
        if isinstance(result, dict):
            scheduler = result.get('scheduler_type', '')
            noise = result.get('noise_type', '')
            losses = result.get('losses', [])
            
            if ('spike' in noise and 
                scheduler in spike_data and 
                len(losses) >= 100):
                
                trajectory = np.array(losses[:100])
                if len(trajectory) == 100:
                    spike_data[scheduler].append(trajectory)
    
    colors = {
        'greedy': '#2E8B57',
        'cosine': '#4682B4', 
        'cosine_restarts': '#FF6347',
        'exponential': '#9370DB'
    }
    
    # Clean percentile bands plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for scheduler, trajectories in spike_data.items():
        if len(trajectories) > 0:
            trajectories_array = np.array(trajectories)
            
            # Calculate percentiles
            median_traj = np.median(trajectories_array, axis=0)
            p10_traj = np.percentile(trajectories_array, 10, axis=0)
            p90_traj = np.percentile(trajectories_array, 90, axis=0)
            
            steps = np.arange(len(median_traj))
            
            # Single dashed line showing 10-90th percentile range (fill between)
            ax.fill_between(steps, p10_traj, p90_traj, 
                          color=colors[scheduler], alpha=0.2, 
                          label=f'{scheduler.replace("_", " ").title()} 10-90th percentile')
            
            # Dashed boundary lines
            ax.plot(p10_traj, color=colors[scheduler], linestyle='--', linewidth=1.5, alpha=0.7)
            ax.plot(p90_traj, color=colors[scheduler], linestyle='--', linewidth=1.5, alpha=0.7)
            
            # Bold median line
            ax.plot(median_traj, color=colors[scheduler], linewidth=3, 
                   label=f'{scheduler.replace("_", " ").title()} median (n={len(trajectories)})')
    
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Loss Value', fontsize=12)
    ax.set_yscale('log')
    ax.legend(fontsize=10, framealpha=0.9, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_title('Recovery Trajectories with Clean Percentile Bands\n(Dashed lines show 10-90th percentile bounds, solid lines show medians)', 
                 fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig('final_plots/recovery_trajectories_clean_bands.png', dpi=300, bbox_inches='tight')
    plt.savefig('final_plots/recovery_trajectories_clean_bands.pdf', bbox_inches='tight')
    plt.close()
    
    print("✅ Created clean percentile bands plot")

def create_neural_network_comparisons():
    """Create individual comparison plots for neural network experiments"""
    print("📊 Creating individual neural network comparison examples...")
    
    with open('robust_results.json', 'r') as f:
        results = json.load(f)
    
    # Find neural network experiments with all 4 schedulers
    neural_groups = {}
    
    for i, result in enumerate(results):
        if isinstance(result, dict):
            model_type = result.get('model_type', 'unknown')
            noise_type = result.get('noise_type', 'unknown')
            scheduler_type = result.get('scheduler_type', 'unknown')
            losses = result.get('losses', [])
            lrs = result.get('lrs', [])
            
            # Focus on neural networks
            if ('neural' in model_type.lower() and 
                len(losses) >= 100 and len(lrs) >= 100 and 
                scheduler_type in ['greedy', 'cosine', 'cosine_restarts', 'exponential']):
                
                condition_key = f"{model_type}_{noise_type}"
                
                if condition_key not in neural_groups:
                    neural_groups[condition_key] = {}
                
                neural_groups[condition_key][scheduler_type] = {
                    'losses': np.array(losses[:100]),
                    'lrs': np.array(lrs[:100]),
                    'model_type': model_type,
                    'noise_type': noise_type
                }
    
    # Find complete groups (all 4 schedulers)
    complete_neural_groups = {}
    for condition, schedulers in neural_groups.items():
        if len(schedulers) == 4:
            complete_neural_groups[condition] = schedulers
    
    print(f"Found {len(complete_neural_groups)} complete neural network condition groups")
    
    # Create individual plots for top examples
    colors = {
        'greedy': '#2E8B57',
        'cosine': '#4682B4', 
        'cosine_restarts': '#FF6347',
        'exponential': '#9370DB'
    }
    
    examples_created = 0
    for condition, schedulers in list(complete_neural_groups.items())[:5]:  # Top 5 examples
        
        # Calculate recovery metrics to find good examples
        greedy_losses = schedulers['greedy']['losses']
        max_loss = greedy_losses.max()
        final_loss = greedy_losses[-10:].mean()
        recovery_ratio = max_loss / final_loss if final_loss > 0 else 0
        
        if recovery_ratio > 5:  # Good recovery example
            examples_created += 1
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            model_type = schedulers['greedy']['model_type']
            noise_type = schedulers['greedy']['noise_type']
            
            # Top left: Loss trajectories
            for scheduler_name, data in schedulers.items():
                losses = data['losses']
                ax1.plot(losses, color=colors[scheduler_name], linewidth=2.5, 
                        label=f"{scheduler_name.replace('_', ' ').title()}", alpha=0.9)
            
            ax1.set_xlabel('Training Step')
            ax1.set_ylabel('Loss Value')
            ax1.set_yscale('log')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_title('Loss Trajectories')
            
            # Top right: GreedyLR learning rate
            greedy_lrs = schedulers['greedy']['lrs']
            ax2.plot(greedy_lrs, color=colors['greedy'], linewidth=2.5, 
                    label='GreedyLR', alpha=0.9)
            ax2.set_xlabel('Training Step')
            ax2.set_ylabel('Learning Rate')
            ax2.set_yscale('log')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_title('GreedyLR Learning Rate Adaptation')
            
            # Bottom left: Final loss comparison
            scheduler_names = []
            final_losses = []
            for scheduler_name, data in schedulers.items():
                scheduler_names.append(scheduler_name.replace('_', ' ').title())
                final_losses.append(data['losses'][-10:].mean())
            
            bars = ax3.bar(scheduler_names, final_losses, 
                          color=[colors[s.lower().replace(' ', '_')] for s in scheduler_names],
                          alpha=0.7, edgecolor='black')
            
            # Add values on bars with better positioning
            for bar, loss in zip(bars, final_losses):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height * 2,
                        f'{loss:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax3.set_ylabel('Final Loss Value')
            ax3.set_title('Final Performance Comparison')
            ax3.set_yscale('log')
            ax3.grid(True, alpha=0.3)
            
            # Bottom right: Recovery ratios
            recovery_ratios = []
            for scheduler_name, data in schedulers.items():
                losses = data['losses']
                max_l = losses.max()
                final_l = losses[-10:].mean()
                recovery_ratios.append(max_l / final_l if final_l > 0 else 0)
            
            bars = ax4.bar(scheduler_names, recovery_ratios,
                          color=[colors[s.lower().replace(' ', '_')] for s in scheduler_names],
                          alpha=0.7, edgecolor='black')
            
            # Add values on bars with better positioning
            for bar, ratio in zip(bars, recovery_ratios):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height * 1.5,
                        f'{ratio:.1f}×', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax4.set_ylabel('Recovery Ratio (Max Loss / Final Loss)')
            ax4.set_title('Recovery Performance')
            ax4.set_yscale('log')
            ax4.grid(True, alpha=0.3)
            
            # Clean title
            plt.suptitle(f'Neural Network Comparison {examples_created}\n'
                        f'{model_type.replace("_", " ").title()} | {noise_type.replace("_", " ").title()} Noise',
                        fontsize=14, y=0.98)
            
            plt.tight_layout()
            plt.savefig(f'final_plots/neural_comparison_{examples_created}.png', dpi=300, bbox_inches='tight')
            plt.savefig(f'final_plots/neural_comparison_{examples_created}.pdf', bbox_inches='tight')
            plt.close()
            
            print(f"✅ Created neural comparison {examples_created}: {model_type} + {noise_type}")
    
    return examples_created

def create_learning_rate_trajectory_analysis():
    """Create learning rate trajectory analysis for each scheduler"""
    print("📊 Creating learning rate trajectory analysis...")
    
    with open('robust_results.json', 'r') as f:
        results = json.load(f)
    
    # Collect learning rate data for each scheduler
    lr_data = {
        'greedy': [],
        'cosine': [],
        'cosine_restarts': [],
        'exponential': []
    }
    
    for result in results:
        if isinstance(result, dict):
            scheduler = result.get('scheduler_type', '')
            lrs = result.get('lrs', [])
            
            if (scheduler in lr_data and len(lrs) >= 100):
                lr_trajectory = np.array(lrs[:100])
                if len(lr_trajectory) == 100 and np.all(lr_trajectory > 0):  # Valid LR values
                    lr_data[scheduler].append(lr_trajectory)
    
    colors = {
        'greedy': '#2E8B57',
        'cosine': '#4682B4', 
        'cosine_restarts': '#FF6347',
        'exponential': '#9370DB'
    }
    
    # Create 2x2 subplot for each scheduler
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, (scheduler, trajectories) in enumerate(lr_data.items()):
        ax = axes[idx]
        
        if len(trajectories) > 0:
            trajectories_array = np.array(trajectories)
            
            # Plot individual trajectories (faint)
            sample_size = min(50, len(trajectories))  # Show up to 50 trajectories
            sample_indices = np.random.choice(len(trajectories), sample_size, replace=False)
            
            for i in sample_indices:
                ax.plot(trajectories[i], color=colors[scheduler], alpha=0.15, linewidth=0.5)
            
            # Plot median trajectory (bold)
            median_traj = np.median(trajectories_array, axis=0)
            ax.plot(median_traj, color=colors[scheduler], linewidth=4, 
                   label=f'Median (n={len(trajectories)})')
            
            # Plot percentile bounds
            p10_traj = np.percentile(trajectories_array, 10, axis=0)
            p90_traj = np.percentile(trajectories_array, 90, axis=0)
            ax.plot(p10_traj, color=colors[scheduler], linestyle='--', linewidth=2, alpha=0.7)
            ax.plot(p90_traj, color=colors[scheduler], linestyle='--', linewidth=2, alpha=0.7)
        
        ax.set_title(f'{scheduler.replace("_", " ").title()} Learning Rate Patterns\n(n={len(trajectories)} experiments)', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Learning Rate')
        ax.set_yscale('log')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Learning Rate Trajectory Analysis by Scheduler\n(Faint lines: individual experiments, Bold line: median, Dashed: 10-90th percentiles)', 
                 fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig('final_plots/learning_rate_trajectories.png', dpi=300, bbox_inches='tight')
    plt.savefig('final_plots/learning_rate_trajectories.pdf', bbox_inches='tight')
    plt.close()
    
    print("✅ Created learning rate trajectory analysis")

def main():
    """Fix all visualization issues"""
    Path('final_plots').mkdir(exist_ok=True)
    
    print("🔧 Fixing All Visualization Issues")
    print("=" * 60)
    
    # 1. Create clean percentile bands
    create_clean_percentile_bands()
    
    # 2. Create individual neural network comparisons
    neural_examples = create_neural_network_comparisons()
    
    # 3. Create learning rate trajectory analysis
    create_learning_rate_trajectory_analysis()
    
    print("=" * 60)
    print("✅ All visualization issues fixed!")
    print(f"📊 Created clean percentile bands")
    print(f"🧠 Created {neural_examples} individual neural network comparisons")
    print(f"📈 Created learning rate trajectory analysis")
    
    return neural_examples

if __name__ == "__main__":
    main()