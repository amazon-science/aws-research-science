#!/usr/bin/env python3
"""
Create improved recovery trajectory plots with better visibility
"""
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

def create_improved_recovery_plots():
    """Create multiple versions of recovery plots for better comparison"""
    print("📊 Creating improved recovery trajectory plots...")
    
    with open('robust_results.json', 'r') as f:
        results = json.load(f)
    
    # Collect spike data for trajectory analysis
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
                
                # Normalize length and store
                trajectory = np.array(losses[:100])
                if len(trajectory) == 100:
                    spike_data[scheduler].append(trajectory)
    
    print("Trajectory counts:")
    for scheduler, trajectories in spike_data.items():
        print(f"  {scheduler}: {len(trajectories)} trajectories")
    
    # Version 1: Darker individual trajectories with clear legend
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = {
        'greedy': '#2E8B57',
        'cosine': '#4682B4', 
        'cosine_restarts': '#FF6347',
        'exponential': '#9370DB'
    }
    
    # Plot individual trajectories (darker)
    for scheduler, trajectories in spike_data.items():
        if len(trajectories) > 0:
            trajectories_array = np.array(trajectories)
            
            # Plot individual trajectories (much darker)
            for i, traj in enumerate(trajectories[:30]):  # Show more trajectories
                alpha = 0.3 if i < 20 else 0.15  # Vary opacity
                ax.plot(traj, color=colors[scheduler], alpha=alpha, linewidth=0.8)
            
            # Plot median trajectory (very bold)
            median_traj = np.median(trajectories_array, axis=0)
            ax.plot(median_traj, color=colors[scheduler], alpha=1.0, linewidth=4, 
                   label=f'{scheduler.replace("_", " ").title()} (median, n={len(trajectories)})')
    
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Loss Value', fontsize=12)
    ax.set_yscale('log')
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_title('Recovery Trajectories: Spike Noise Conditions\n(Individual trajectories shown with median highlighted)', 
                 fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig('final_plots/recovery_trajectories_darker.png', dpi=300, bbox_inches='tight')
    plt.savefig('final_plots/recovery_trajectories_darker.pdf', bbox_inches='tight')
    plt.close()
    
    # Version 2: Confidence bands (percentiles)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for scheduler, trajectories in spike_data.items():
        if len(trajectories) > 0:
            trajectories_array = np.array(trajectories)
            
            # Calculate percentiles
            median_traj = np.median(trajectories_array, axis=0)
            p25_traj = np.percentile(trajectories_array, 25, axis=0)
            p75_traj = np.percentile(trajectories_array, 75, axis=0)
            p10_traj = np.percentile(trajectories_array, 10, axis=0)
            p90_traj = np.percentile(trajectories_array, 90, axis=0)
            
            steps = np.arange(len(median_traj))
            
            # Plot confidence bands
            ax.fill_between(steps, p10_traj, p90_traj, 
                          color=colors[scheduler], alpha=0.15, 
                          label=f'{scheduler.replace("_", " ").title()} 10-90th percentile')
            ax.fill_between(steps, p25_traj, p75_traj, 
                          color=colors[scheduler], alpha=0.25)
            
            # Plot median
            ax.plot(median_traj, color=colors[scheduler], linewidth=3, 
                   label=f'{scheduler.replace("_", " ").title()} median (n={len(trajectories)})')
    
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Loss Value', fontsize=12)
    ax.set_yscale('log')
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_title('Recovery Trajectories with Confidence Bands\n(Shaded regions show 10-90th and 25-75th percentiles)', 
                 fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig('final_plots/recovery_trajectories_bands.png', dpi=300, bbox_inches='tight')
    plt.savefig('final_plots/recovery_trajectories_bands.pdf', bbox_inches='tight')
    plt.close()
    
    # Version 3: Separate subplots for each scheduler
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    scheduler_names = list(spike_data.keys())
    
    for i, (scheduler, trajectories) in enumerate(spike_data.items()):
        ax = axes[i]
        
        if len(trajectories) > 0:
            trajectories_array = np.array(trajectories)
            
            # Plot individual trajectories (more visible)
            for traj in trajectories[:50]:  # Show more trajectories
                ax.plot(traj, color=colors[scheduler], alpha=0.4, linewidth=0.7)
            
            # Calculate and plot statistics
            median_traj = np.median(trajectories_array, axis=0)
            mean_traj = np.mean(trajectories_array, axis=0)
            
            ax.plot(median_traj, color='black', linewidth=3, 
                   label=f'Median (n={len(trajectories)})')
            ax.plot(mean_traj, color='red', linewidth=2, linestyle='--',
                   label=f'Mean')
            
            # Add percentile bands
            p25_traj = np.percentile(trajectories_array, 25, axis=0)
            p75_traj = np.percentile(trajectories_array, 75, axis=0)
            steps = np.arange(len(median_traj))
            ax.fill_between(steps, p25_traj, p75_traj, 
                          color=colors[scheduler], alpha=0.2, 
                          label='25-75th percentile')
        
        ax.set_title(f'{scheduler.replace("_", " ").title()} Scheduler', fontsize=12, fontweight='bold')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss Value')
        ax.set_yscale('log')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Recovery Trajectories by Scheduler: Spike Noise Conditions', 
                 fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig('final_plots/recovery_trajectories_separate.png', dpi=300, bbox_inches='tight')
    plt.savefig('final_plots/recovery_trajectories_separate.pdf', bbox_inches='tight')
    plt.close()
    
    # Version 4: Direct comparison with summary statistics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Median trajectories only (very clear)
    for scheduler, trajectories in spike_data.items():
        if len(trajectories) > 0:
            trajectories_array = np.array(trajectories)
            median_traj = np.median(trajectories_array, axis=0)
            
            ax1.plot(median_traj, color=colors[scheduler], linewidth=4, 
                    label=f'{scheduler.replace("_", " ").title()} (n={len(trajectories)})')
    
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('Loss Value', fontsize=12)
    ax1.set_yscale('log')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Median Recovery Trajectories', fontsize=14)
    
    # Right: Recovery statistics
    recovery_stats = []
    for scheduler, trajectories in spike_data.items():
        if len(trajectories) > 0:
            trajectories_array = np.array(trajectories)
            
            # Calculate recovery metrics
            initial_losses = trajectories_array[:, 0]
            final_losses = trajectories_array[:, -10:].mean(axis=1)  # Last 10 steps
            max_losses = trajectories_array.max(axis=1)
            
            recovery_ratios = max_losses / final_losses
            improvement_ratios = initial_losses / final_losses
            
            recovery_stats.append({
                'scheduler': scheduler.replace('_', ' ').title(),
                'median_recovery': np.median(recovery_ratios),
                'mean_recovery': np.mean(recovery_ratios),
                'best_recovery': np.max(recovery_ratios),
                'median_improvement': np.median(improvement_ratios),
                'count': len(trajectories)
            })
    
    # Plot recovery statistics
    schedulers = [s['scheduler'] for s in recovery_stats]
    median_recoveries = [s['median_recovery'] for s in recovery_stats]
    
    bars = ax2.bar(schedulers, median_recoveries, 
                   color=[colors[s.lower().replace(' ', '_')] for s in schedulers],
                   alpha=0.7, edgecolor='black')
    
    # Add values on bars
    for bar, stats in zip(bars, recovery_stats):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                f'{height:.1f}×\n(n={stats["count"]})',
                ha='center', va='bottom', fontsize=10)
    
    ax2.set_ylabel('Median Recovery Ratio (Max Loss / Final Loss)', fontsize=12)
    ax2.set_title('Recovery Performance Comparison', fontsize=14)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('final_plots/recovery_comparison_clear.png', dpi=300, bbox_inches='tight')
    plt.savefig('final_plots/recovery_comparison_clear.pdf', bbox_inches='tight')
    plt.close()
    
    print("✅ Created improved recovery trajectory plots:")
    print("  - recovery_trajectories_darker.png (darker individual trajectories)")
    print("  - recovery_trajectories_bands.png (confidence bands)")
    print("  - recovery_trajectories_separate.png (separate subplots per scheduler)")
    print("  - recovery_comparison_clear.png (median comparison + statistics)")
    
    return recovery_stats

def main():
    """Run all improved plot creation"""
    Path('final_plots').mkdir(exist_ok=True)
    
    print("📊 Creating improved recovery trajectory visualizations...")
    print("=" * 60)
    
    recovery_stats = create_improved_recovery_plots()
    
    print("\n📈 Recovery Statistics Summary:")
    for stats in recovery_stats:
        print(f"  {stats['scheduler']:15} | Median Recovery: {stats['median_recovery']:8.1f}× | Best: {stats['best_recovery']:8.1f}× | n={stats['count']}")
    
    print("=" * 60)
    print("✅ All improved recovery plots created!")

if __name__ == "__main__":
    main()