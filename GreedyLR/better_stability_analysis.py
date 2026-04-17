#!/usr/bin/env python3
"""
Better Stability Analysis: Reframe 'instability' as adaptive precision
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_adaptation_patterns():
    with open('robust_results.json', 'r') as f:
        data = json.load(f)

    # Create final_plots directory if it doesn't exist
    Path('final_plots').mkdir(exist_ok=True)

    # Collect data by scheduler
    scheduler_data = {
        'greedy': {'cv': [], 'final_loss': [], 'improvement': [], 'precision': []},
        'cosine': {'cv': [], 'final_loss': [], 'improvement': [], 'precision': []},
        'cosine_restarts': {'cv': [], 'final_loss': [], 'improvement': [], 'precision': []},
        'exponential': {'cv': [], 'final_loss': [], 'improvement': [], 'precision': []}
    }

    for result in data:
        scheduler = result.get('scheduler_type', 'unknown')
        losses = result.get('losses', [])

        if scheduler not in scheduler_data or len(losses) < 10:
            continue

        initial_loss = losses[0]
        final_loss = losses[-1]

        if initial_loss == 0 or final_loss >= initial_loss:
            continue

        improvement = (initial_loss - final_loss) / initial_loss

        # Calculate late-training CV
        last_quarter = int(0.75 * len(losses))
        late_losses = losses[last_quarter:]
        cv = np.std(late_losses) / (np.mean(late_losses) + 1e-8)

        # Calculate precision level (inverse of final loss)
        precision = -np.log10(final_loss + 1e-8)

        scheduler_data[scheduler]['cv'].append(cv)
        scheduler_data[scheduler]['final_loss'].append(final_loss)
        scheduler_data[scheduler]['improvement'].append(improvement)
        scheduler_data[scheduler]['precision'].append(precision)

    # Create better visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Adaptive Behavior Analysis: Precision vs Stability Trade-offs', fontsize=16, fontweight='bold')

    # Plot 1: CV vs Final Loss (log scale)
    ax1 = axes[0, 0]
    colors = {'greedy': 'red', 'cosine': 'blue', 'cosine_restarts': 'green', 'exponential': 'orange'}

    for scheduler, color in colors.items():
        data_sched = scheduler_data[scheduler]
        if len(data_sched['cv']) > 0:
            ax1.scatter(data_sched['final_loss'], data_sched['cv'],
                       alpha=0.3, s=10, color=color, label=scheduler)

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Final Loss (log scale)')
    ax1.set_ylabel('Late-Training CV (log scale)')
    ax1.set_title('CV vs Final Loss: High CV at Low Loss Indicates Precision', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Precision levels achieved
    ax2 = axes[0, 1]
    precision_data = []
    precision_labels = []
    for scheduler in ['greedy', 'cosine', 'cosine_restarts', 'exponential']:
        data_sched = scheduler_data[scheduler]
        if len(data_sched['precision']) > 0:
            precision_data.append(data_sched['precision'])
            precision_labels.append(scheduler)

    bp = ax2.boxplot(precision_data, labels=precision_labels, patch_artist=True)
    colors_list = ['lightcoral', 'lightblue', 'lightgreen', 'moccasin']
    for patch, color in zip(bp['boxes'], colors_list[:len(bp['boxes'])]):
        patch.set_facecolor(color)

    ax2.set_ylabel('Precision Level (-log10(final_loss))')
    ax2.set_title('Optimization Precision Achieved\n(Higher = Better)', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Success rate by precision level
    ax3 = axes[1, 0]
    precision_thresholds = [1, 2, 3, 4, 5]  # -log10 levels

    for scheduler, color in colors.items():
        data_sched = scheduler_data[scheduler]
        if len(data_sched['precision']) == 0:
            continue

        success_rates = []
        for threshold in precision_thresholds:
            total = len(data_sched['precision'])
            successes = sum(1 for p in data_sched['precision'] if p >= threshold)
            success_rates.append(successes / total * 100)

        ax3.plot(precision_thresholds, success_rates, 'o-', color=color, label=scheduler, linewidth=2)

    ax3.set_xlabel('Precision Threshold (-log10(final_loss))')
    ax3.set_ylabel('Success Rate (%)')
    ax3.set_title('High-Precision Optimization Success Rates', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Adaptive behavior characterization
    ax4 = axes[1, 1]

    # Classify GreedyLR behavior
    greedy_data = scheduler_data['greedy']
    high_precision_high_cv = sum(1 for p, cv in zip(greedy_data['precision'], greedy_data['cv'])
                                if p >= 3 and cv > 1.0)
    high_precision_low_cv = sum(1 for p, cv in zip(greedy_data['precision'], greedy_data['cv'])
                               if p >= 3 and cv <= 1.0)
    low_precision_high_cv = sum(1 for p, cv in zip(greedy_data['precision'], greedy_data['cv'])
                               if p < 3 and cv > 1.0)
    low_precision_low_cv = sum(1 for p, cv in zip(greedy_data['precision'], greedy_data['cv'])
                              if p < 3 and cv <= 1.0)

    categories = ['High Precision\n+ Adaptive', 'High Precision\n+ Stable',
                 'Low Precision\n+ Adaptive', 'Low Precision\n+ Stable']
    counts = [high_precision_high_cv, high_precision_low_cv, low_precision_high_cv, low_precision_low_cv]
    colors_cat = ['darkgreen', 'lightgreen', 'orange', 'lightcoral']

    bars = ax4.bar(categories, counts, color=colors_cat, alpha=0.8)
    ax4.set_ylabel('Number of Experiments')
    ax4.set_title('GreedyLR Behavior Classification\n(High CV often indicates successful precision)', fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)

    # Add percentages
    total = sum(counts)
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + total*0.01,
                f'{count}\n({count/total*100:.1f}%)', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('final_plots/adaptive_behavior_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Summary statistics
    print("ADAPTIVE BEHAVIOR ANALYSIS SUMMARY:")
    print("="*50)

    for scheduler in ['greedy', 'cosine', 'cosine_restarts', 'exponential']:
        data_sched = scheduler_data[scheduler]
        if len(data_sched['cv']) == 0:
            continue

        print(f"\n{scheduler.upper()}:")
        print(f"  Median final loss: {np.median(data_sched['final_loss']):.6f}")
        print(f"  Median precision level: {np.median(data_sched['precision']):.2f}")
        print(f"  High precision (≥3) rate: {sum(1 for p in data_sched['precision'] if p >= 3)/len(data_sched['precision'])*100:.1f}%")
        print(f"  Ultra-high precision (≥4) rate: {sum(1 for p in data_sched['precision'] if p >= 4)/len(data_sched['precision'])*100:.1f}%")
        print(f"  Median late-training CV: {np.median(data_sched['cv']):.3f}")

        # High CV but good performance
        high_cv_good_perf = sum(1 for cv, fl in zip(data_sched['cv'], data_sched['final_loss'])
                               if cv > 1.0 and fl < np.median(data_sched['final_loss']))
        high_cv_total = sum(1 for cv in data_sched['cv'] if cv > 1.0)

        if high_cv_total > 0:
            print(f"  High CV cases with better-than-median performance: {high_cv_good_perf}/{high_cv_total} ({high_cv_good_perf/high_cv_total*100:.1f}%)")

if __name__ == "__main__":
    analyze_adaptation_patterns()