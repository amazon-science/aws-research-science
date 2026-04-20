#!/usr/bin/env python3
"""
Investigate data patterns and find good recovery examples
"""
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set non-interactive backend to avoid window freezing
import matplotlib
matplotlib.use('Agg')

def investigate_sample_sizes():
    """Investigate why sample sizes are different"""
    print("🔍 Investigating sample size differences...")
    
    with open('robust_results.json', 'r') as f:
        results = json.load(f)
    
    # Count by scheduler type
    scheduler_counts = {}
    noise_counts = {}
    architecture_counts = {}
    
    for result in results:
        if isinstance(result, dict):
            scheduler = result.get('scheduler_type', 'unknown')
            noise = result.get('noise_type', 'unknown')
            arch = result.get('model_type', 'unknown')
            
            scheduler_counts[scheduler] = scheduler_counts.get(scheduler, 0) + 1
            noise_counts[noise] = noise_counts.get(noise, 0) + 1
            architecture_counts[arch] = architecture_counts.get(arch, 0) + 1
    
    print("Sample sizes by scheduler:")
    for scheduler, count in sorted(scheduler_counts.items()):
        print(f"  {scheduler}: {count}")
    
    print(f"\nTotal experiments: {len(results)}")
    print(f"Expected per scheduler (if equal): {len(results) // len(scheduler_counts)}")
    
    print("\nNoise condition counts:")
    for noise, count in sorted(noise_counts.items()):
        print(f"  {noise}: {count}")
    
    print("\nArchitecture counts:")
    for arch, count in sorted(architecture_counts.items()):
        print(f"  {arch}: {count}")

def find_recovery_examples():
    """Find good examples of GreedyLR recovery"""
    print("🔍 Finding recovery examples...")
    
    with open('robust_results.json', 'r') as f:
        results = json.load(f)
    
    recovery_candidates = []
    
    for i, result in enumerate(results):
        if isinstance(result, dict):
            scheduler = result.get('scheduler_type', '')
            noise = result.get('noise_type', '')
            losses = result.get('losses', [])
            lrs = result.get('lrs', [])
            
            if (scheduler == 'greedy' and 
                'spike' in noise and 
                len(losses) > 100 and 
                len(lrs) > 100):
                
                # Check for actual recovery patterns
                losses_array = np.array(losses[:100])
                
                # Look for spike followed by recovery
                if len(losses_array) > 50:
                    max_loss = np.max(losses_array)
                    min_loss = np.min(losses_array)
                    final_loss = losses_array[-10:].mean()
                    
                    # Good recovery: big spike but low final loss
                    if max_loss > min_loss * 10 and final_loss < max_loss * 0.1:
                        recovery_candidates.append({
                            'index': i,
                            'scheduler': scheduler,
                            'noise': noise,
                            'max_loss': max_loss,
                            'min_loss': min_loss,
                            'final_loss': final_loss,
                            'recovery_ratio': max_loss / final_loss,
                            'losses': losses_array,
                            'lrs': np.array(lrs[:100])
                        })
    
    print(f"Found {len(recovery_candidates)} recovery candidates")
    
    # Sort by recovery ratio (best recoveries first)
    recovery_candidates.sort(key=lambda x: x['recovery_ratio'], reverse=True)
    
    # Show top candidates
    print("\nTop recovery examples:")
    for i, candidate in enumerate(recovery_candidates[:5]):
        print(f"  {i+1}. {candidate['noise']} - Recovery: {candidate['recovery_ratio']:.1f}x")
        print(f"      Max loss: {candidate['max_loss']:.3f}, Final: {candidate['final_loss']:.3f}")
    
    return recovery_candidates

def create_recovery_trajectory_plots():
    """Create recovery trajectory plots"""
    print("📊 Creating recovery trajectory plots...")
    
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
    
    # Create overall trajectory plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {
        'greedy': '#2E8B57',
        'cosine': '#4682B4', 
        'cosine_restarts': '#FF6347',
        'exponential': '#9370DB'
    }
    
    for scheduler, trajectories in spike_data.items():
        if len(trajectories) > 0:
            trajectories_array = np.array(trajectories)
            
            # Plot individual trajectories (faint)
            for traj in trajectories[:20]:  # Limit to avoid clutter
                ax.plot(traj, color=colors[scheduler], alpha=0.1, linewidth=0.5)
            
            # Plot median trajectory (bold)
            median_traj = np.median(trajectories_array, axis=0)
            ax.plot(median_traj, color=colors[scheduler], alpha=0.9, linewidth=3, 
                   label=f'{scheduler.title()} (median, n={len(trajectories)})')
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss Value')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Recovery Trajectories: Spike Noise Conditions')
    
    plt.tight_layout()
    plt.savefig('final_plots/recovery_trajectories_overall.png', dpi=300, bbox_inches='tight')
    plt.savefig('final_plots/recovery_trajectories_overall.pdf', bbox_inches='tight')
    plt.close()
    
    # Find and plot best individual recovery example
    recovery_candidates = find_recovery_examples()
    
    if recovery_candidates:
        best_recovery = recovery_candidates[0]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Top: Loss trajectory
        ax1.plot(best_recovery['losses'], color='#2E8B57', linewidth=2)
        ax1.set_ylabel('Loss Value')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f'Individual Recovery Example: {best_recovery["noise"].title()}')
        
        # Bottom: Learning rate adaptation
        ax2.plot(best_recovery['lrs'], color='#FF6B35', linewidth=2)
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Learning Rate')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Corresponding Learning Rate Adaptation')
        
        plt.tight_layout()
        plt.savefig('final_plots/best_individual_recovery.png', dpi=300, bbox_inches='tight')
        plt.savefig('final_plots/best_individual_recovery.pdf', bbox_inches='tight')
        plt.close()
        
        print(f"✅ Created individual recovery plot: {best_recovery['recovery_ratio']:.1f}x recovery")
    
    print("✅ Recovery trajectory plots saved")

def main():
    """Run all investigations"""
    Path('final_plots').mkdir(exist_ok=True)
    
    print("🔍 Investigating data patterns...")
    print("=" * 50)
    
    investigate_sample_sizes()
    print()
    create_recovery_trajectory_plots()
    
    print("=" * 50)
    print("✅ Data investigation complete!")

if __name__ == "__main__":
    main()