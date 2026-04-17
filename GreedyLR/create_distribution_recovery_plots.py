#!/usr/bin/env python3
"""
Create recovery plots showing trajectory distributions with dotted bounds for GreedyLR
and multiple approaches to visualize distribution density
"""
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

def create_distribution_recovery_plots():
    """Create recovery plots with better distribution visualization"""
    print("📊 Creating distribution-aware recovery plots...")
    
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
    
    colors = {
        'greedy': '#2E8B57',
        'cosine': '#4682B4', 
        'cosine_restarts': '#FF6347',
        'exponential': '#9370DB'
    }
    
    # Version 1: Dotted bounds for GreedyLR, solid bands for others
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for scheduler, trajectories in spike_data.items():
        if len(trajectories) > 0:
            trajectories_array = np.array(trajectories)
            
            # Calculate percentiles
            median_traj = np.median(trajectories_array, axis=0)
            p10_traj = np.percentile(trajectories_array, 10, axis=0)
            p90_traj = np.percentile(trajectories_array, 90, axis=0)
            p25_traj = np.percentile(trajectories_array, 25, axis=0)
            p75_traj = np.percentile(trajectories_array, 75, axis=0)
            
            steps = np.arange(len(median_traj))
            
            if scheduler == 'greedy':
                # GreedyLR: Dotted lines for bounds, no fill
                ax.plot(p10_traj, color=colors[scheduler], linestyle=':', linewidth=2, alpha=0.7,
                       label=f'{scheduler.replace("_", " ").title()} 10th percentile')
                ax.plot(p90_traj, color=colors[scheduler], linestyle=':', linewidth=2, alpha=0.7,
                       label=f'{scheduler.replace("_", " ").title()} 90th percentile')
                ax.plot(p25_traj, color=colors[scheduler], linestyle='--', linewidth=1.5, alpha=0.6)
                ax.plot(p75_traj, color=colors[scheduler], linestyle='--', linewidth=1.5, alpha=0.6)
                
                # Median line (solid, bold)
                ax.plot(median_traj, color=colors[scheduler], linewidth=4, 
                       label=f'{scheduler.replace("_", " ").title()} median (n={len(trajectories)})')
            else:
                # Other schedulers: Solid bands
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
    ax.set_title('Recovery Trajectories: GreedyLR with Dotted Bounds\n(Dotted/dashed lines show GreedyLR distribution, shaded areas show competitors)', 
                 fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig('final_plots/recovery_trajectories_dotted_bounds.png', dpi=300, bbox_inches='tight')
    plt.savefig('final_plots/recovery_trajectories_dotted_bounds.pdf', bbox_inches='tight')
    plt.close()
    
    # Version 2: Multiple plots showing distribution density
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, (scheduler, trajectories) in enumerate(spike_data.items()):
        ax = axes[idx]
        
        if len(trajectories) > 0:
            trajectories_array = np.array(trajectories)
            
            # Method 1: Show multiple percentile lines to indicate density
            percentiles = [5, 10, 25, 50, 75, 90, 95]
            alphas = [0.3, 0.4, 0.6, 1.0, 0.6, 0.4, 0.3]  # More opacity for median
            linewidths = [1, 1.5, 2, 4, 2, 1.5, 1]  # Thicker for median
            
            for p, alpha, lw in zip(percentiles, alphas, linewidths):
                p_traj = np.percentile(trajectories_array, p, axis=0)
                linestyle = '-' if p == 50 else '--'
                label = f'{p}th percentile' if p in [10, 50, 90] else None
                ax.plot(p_traj, color=colors[scheduler], alpha=alpha, linewidth=lw,
                       linestyle=linestyle, label=label)
            
            # Add sample trajectories for reference
            sample_indices = np.random.choice(len(trajectories), min(10, len(trajectories)), replace=False)
            for i in sample_indices:
                ax.plot(trajectories[i], color=colors[scheduler], alpha=0.1, linewidth=0.5)
        
        ax.set_title(f'{scheduler.replace("_", " ").title()} Distribution\n(n={len(trajectories)} trajectories)', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss Value')
        ax.set_yscale('log')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Recovery Trajectory Distributions by Scheduler\n(Multiple percentile lines show distribution density)', 
                 fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig('final_plots/recovery_distributions_detailed.png', dpi=300, bbox_inches='tight')
    plt.savefig('final_plots/recovery_distributions_detailed.pdf', bbox_inches='tight')
    plt.close()
    
    # Version 3: Violin plots showing distribution at key time points
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    time_points = [10, 50, 90]  # Early, middle, late training
    
    for t_idx, time_point in enumerate(time_points):
        ax = axes[t_idx]
        
        # Collect loss values at this time point for each scheduler
        plot_data = []
        labels = []
        colors_list = []
        
        for scheduler, trajectories in spike_data.items():
            if len(trajectories) > 0:
                trajectories_array = np.array(trajectories)
                if time_point < trajectories_array.shape[1]:
                    losses_at_time = trajectories_array[:, time_point]
                    plot_data.append(losses_at_time)
                    labels.append(scheduler.replace('_', ' ').title())
                    colors_list.append(colors[scheduler])
        
        # Create violin plot
        parts = ax.violinplot(plot_data, positions=range(len(plot_data)), widths=0.6, showmeans=True)
        
        # Color the violin plots
        for pc, color in zip(parts['bodies'], colors_list):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        # Customize violin plot
        if 'cmeans' in parts:
            parts['cmeans'].set_color('black')
            parts['cmeans'].set_linewidth(2)
        elif 'means' in parts:
            parts['means'].set_color('black')
            parts['means'].set_linewidth(2)
        
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Loss Value')
        ax.set_yscale('log')
        ax.set_title(f'Distribution at Step {time_point}')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Loss Distribution Comparison at Key Training Points\n(Violin plots show full distribution shape)', 
                 fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('final_plots/recovery_distribution_violins.png', dpi=300, bbox_inches='tight')
    plt.savefig('final_plots/recovery_distribution_violins.pdf', bbox_inches='tight')
    plt.close()
    
    # Version 4: Density heatmap showing trajectory concentration
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, (scheduler, trajectories) in enumerate(spike_data.items()):
        ax = axes[idx]
        
        if len(trajectories) > 0:
            trajectories_array = np.array(trajectories)
            
            # Create 2D histogram (heatmap) of trajectories
            # X-axis: training step, Y-axis: log loss value
            steps = np.arange(trajectories_array.shape[1])
            
            # Flatten trajectories for histogram
            all_steps = []
            all_losses = []
            for traj in trajectories_array:
                for step, loss in enumerate(traj):
                    if loss > 0:  # Valid loss values only
                        all_steps.append(step)
                        all_losses.append(np.log10(loss))
            
            # Create 2D histogram
            if len(all_steps) > 0:
                h, xedges, yedges = np.histogram2d(all_steps, all_losses, bins=[50, 50])
                
                # Plot heatmap
                im = ax.imshow(h.T, origin='lower', aspect='auto', cmap='Blues', alpha=0.8,
                              extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
                
                # Overlay median trajectory
                median_traj = np.median(trajectories_array, axis=0)
                median_log = np.log10(np.maximum(median_traj, 1e-10))  # Avoid log(0)
                ax.plot(steps, median_log, color=colors[scheduler], linewidth=4, 
                       label=f'Median trajectory')
                
                # Add colorbar
                plt.colorbar(im, ax=ax, alpha=0.8, shrink=0.8)
        
        ax.set_title(f'{scheduler.replace("_", " ").title()} Density\n(n={len(trajectories)} trajectories)', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Log₁₀(Loss Value)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Recovery Trajectory Density Maps\n(Heat intensity shows trajectory concentration)', 
                 fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig('final_plots/recovery_density_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.savefig('final_plots/recovery_density_heatmaps.pdf', bbox_inches='tight')
    plt.close()
    
    print("✅ Created distribution-aware recovery plots:")
    print("  - recovery_trajectories_dotted_bounds.png (dotted bounds for GreedyLR)")
    print("  - recovery_distributions_detailed.png (multiple percentile lines)")
    print("  - recovery_distribution_violins.png (violin plots at key time points)")
    print("  - recovery_density_heatmaps.png (density heatmaps)")

def main():
    """Run distribution recovery plot creation"""
    Path('final_plots').mkdir(exist_ok=True)
    
    print("📊 Creating Distribution-Aware Recovery Visualizations")
    print("=" * 60)
    
    create_distribution_recovery_plots()
    
    print("=" * 60)
    print("✅ All distribution recovery plots created!")

if __name__ == "__main__":
    main()