#!/usr/bin/env python3
"""
Fix image layouts to remove excessive whitespace and improve readability
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def create_compact_plots():
    with open('robust_results.json', 'r') as f:
        data = json.load(f)

    Path('final_plots').mkdir(exist_ok=True)

    # Load convergence analysis data
    convergence_metrics = {
        'scheduler': [],
        'time_to_90_percent': [],
        'time_to_95_percent': [],
        'time_to_99_percent': [],
        'convergence_stability': [],
        'early_stopping_performance': [],
        'final_loss': []
    }

    for result in data:
        scheduler = result.get('scheduler_type', 'unknown')
        losses = result.get('losses', [])

        if len(losses) < 10:
            continue

        initial_loss = losses[0]
        final_loss = losses[-1]

        if final_loss >= initial_loss or initial_loss == 0:
            continue

        # Calculate convergence targets
        improvement_90 = initial_loss - 0.9 * (initial_loss - final_loss)
        improvement_95 = initial_loss - 0.95 * (initial_loss - final_loss)
        improvement_99 = initial_loss - 0.99 * (initial_loss - final_loss)

        # Find time to reach targets
        time_90 = len(losses)
        time_95 = len(losses)
        time_99 = len(losses)

        for i, loss in enumerate(losses):
            if loss <= improvement_90 and time_90 == len(losses):
                time_90 = i
            if loss <= improvement_95 and time_95 == len(losses):
                time_95 = i
            if loss <= improvement_99 and time_99 == len(losses):
                time_99 = i

        # Calculate stability
        last_20_percent = int(0.8 * len(losses))
        if last_20_percent < len(losses) - 5:
            late_losses = losses[last_20_percent:]
            stability = np.std(late_losses) / (np.mean(late_losses) + 1e-8)
        else:
            stability = np.nan

        # Early stopping performance
        early_stop_idx = len(losses) // 2
        early_performance = losses[early_stop_idx] if early_stop_idx < len(losses) else final_loss

        convergence_metrics['scheduler'].append(scheduler)
        convergence_metrics['time_to_90_percent'].append(time_90)
        convergence_metrics['time_to_95_percent'].append(time_95)
        convergence_metrics['time_to_99_percent'].append(time_99)
        convergence_metrics['convergence_stability'].append(stability)
        convergence_metrics['early_stopping_performance'].append(early_performance)
        convergence_metrics['final_loss'].append(final_loss)

    import pandas as pd
    df = pd.DataFrame(convergence_metrics).dropna()

    # 1. COMPACT Convergence Rate Analysis
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Convergence Rate Analysis', fontsize=14, fontweight='bold')

    metrics = ['time_to_90_percent', 'time_to_95_percent', 'time_to_99_percent', 'convergence_stability']
    titles = ['Time to 90%', 'Time to 95%', 'Time to 99%', 'Stability (CV)']

    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i//2, i%2]

        scheduler_data = []
        scheduler_labels = []
        for scheduler in ['greedy', 'cosine', 'cosine_restarts', 'exponential']:
            data_vals = df[df['scheduler'] == scheduler][metric].dropna()
            if len(data_vals) > 0:
                scheduler_data.append(data_vals)
                scheduler_labels.append(scheduler.replace('_', '\n'))

        if scheduler_data:
            bp = ax.boxplot(scheduler_data, labels=scheduler_labels, patch_artist=True)
            colors = ['lightcoral', 'lightblue', 'lightgreen', 'moccasin']
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylabel('Steps' if 'time_to' in metric else 'CV', fontsize=9)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig('final_plots/convergence_rate_analysis_compact.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. COMPACT Early Stopping Analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Early Stopping Performance Analysis', fontsize=14, fontweight='bold')

    # Early vs final correlation
    schedulers = ['greedy', 'cosine', 'cosine_restarts', 'exponential']
    colors = ['red', 'blue', 'green', 'orange']

    for scheduler, color in zip(schedulers, colors):
        scheduler_df = df[df['scheduler'] == scheduler]
        if len(scheduler_df) > 0:
            ax1.scatter(scheduler_df['early_stopping_performance'], scheduler_df['final_loss'],
                       alpha=0.4, s=15, color=color, label=scheduler.replace('_', ' '))

    ax1.set_xlabel('Mid-Training Performance', fontsize=10)
    ax1.set_ylabel('Final Performance', fontsize=10)
    ax1.set_title('Early vs Final Performance', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    # Early stopping ratios
    early_ratios = []
    scheduler_names = []
    for scheduler in schedulers:
        scheduler_df = df[df['scheduler'] == scheduler]
        if len(scheduler_df) > 0:
            ratio = scheduler_df['final_loss'].median() / scheduler_df['early_stopping_performance'].median()
            early_ratios.append(ratio)
            scheduler_names.append(scheduler.replace('_', '\n'))

    bars = ax2.bar(scheduler_names, early_ratios, color=colors[:len(scheduler_names)], alpha=0.7)
    ax2.set_ylabel('Final/Early Ratio', fontsize=10)
    ax2.set_title('Early Stopping Efficiency\n(Lower = Better)', fontsize=11, fontweight='bold')
    ax2.tick_params(axis='x', labelsize=8)
    ax2.tick_params(axis='y', labelsize=8)
    ax2.grid(True, alpha=0.3)

    # Add value labels
    for bar, ratio in zip(bars, early_ratios):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{ratio:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.savefig('final_plots/early_stopping_analysis_compact.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. COMPACT Adaptive Behavior Analysis
    scheduler_data = {
        'greedy': {'cv': [], 'final_loss': [], 'precision': []},
        'cosine': {'cv': [], 'final_loss': [], 'precision': []},
        'cosine_restarts': {'cv': [], 'final_loss': [], 'precision': []},
        'exponential': {'cv': [], 'final_loss': [], 'precision': []}
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

        # Calculate late-training CV
        last_quarter = int(0.75 * len(losses))
        late_losses = losses[last_quarter:]
        cv = np.std(late_losses) / (np.mean(late_losses) + 1e-8)

        # Calculate precision level
        precision = -np.log10(final_loss + 1e-8)

        scheduler_data[scheduler]['cv'].append(cv)
        scheduler_data[scheduler]['final_loss'].append(final_loss)
        scheduler_data[scheduler]['precision'].append(precision)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Adaptive Behavior Analysis: Precision vs Stability', fontsize=14, fontweight='bold')

    # CV vs Final Loss
    ax1 = axes[0, 0]
    colors_dict = {'greedy': 'red', 'cosine': 'blue', 'cosine_restarts': 'green', 'exponential': 'orange'}

    for scheduler, color in colors_dict.items():
        data_sched = scheduler_data[scheduler]
        if len(data_sched['cv']) > 0:
            ax1.scatter(data_sched['final_loss'], data_sched['cv'],
                       alpha=0.3, s=8, color=color, label=scheduler.replace('_', ' '))

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Final Loss', fontsize=9)
    ax1.set_ylabel('Late-Training CV', fontsize=9)
    ax1.set_title('CV at Low Loss = Precision', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Precision levels
    ax2 = axes[0, 1]
    precision_data = []
    precision_labels = []
    for scheduler in ['greedy', 'cosine', 'cosine_restarts', 'exponential']:
        data_sched = scheduler_data[scheduler]
        if len(data_sched['precision']) > 0:
            precision_data.append(data_sched['precision'])
            precision_labels.append(scheduler.replace('_', '\n'))

    bp = ax2.boxplot(precision_data, labels=precision_labels, patch_artist=True)
    colors_list = ['lightcoral', 'lightblue', 'lightgreen', 'moccasin']
    for patch, color in zip(bp['boxes'], colors_list[:len(bp['boxes'])]):
        patch.set_facecolor(color)

    ax2.set_ylabel('Precision Level', fontsize=9)
    ax2.set_title('Optimization Precision', fontsize=11, fontweight='bold')
    ax2.tick_params(axis='x', labelsize=8)
    ax2.grid(True, alpha=0.3)

    # High precision success rates
    ax3 = axes[1, 0]
    precision_thresholds = [1, 2, 3, 4]

    for scheduler, color in colors_dict.items():
        data_sched = scheduler_data[scheduler]
        if len(data_sched['precision']) == 0:
            continue

        success_rates = []
        for threshold in precision_thresholds:
            total = len(data_sched['precision'])
            successes = sum(1 for p in data_sched['precision'] if p >= threshold)
            success_rates.append(successes / total * 100)

        ax3.plot(precision_thresholds, success_rates, 'o-', color=color,
                label=scheduler.replace('_', ' '), linewidth=2, markersize=4)

    ax3.set_xlabel('Precision Threshold', fontsize=9)
    ax3.set_ylabel('Success Rate (%)', fontsize=9)
    ax3.set_title('High-Precision Success', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Behavior classification for GreedyLR
    ax4 = axes[1, 1]
    greedy_data = scheduler_data['greedy']

    high_precision_adaptive = sum(1 for p, cv in zip(greedy_data['precision'], greedy_data['cv'])
                                 if p >= 3 and cv > 1.0)
    high_precision_stable = sum(1 for p, cv in zip(greedy_data['precision'], greedy_data['cv'])
                               if p >= 3 and cv <= 1.0)
    low_precision_adaptive = sum(1 for p, cv in zip(greedy_data['precision'], greedy_data['cv'])
                                if p < 3 and cv > 1.0)
    low_precision_stable = sum(1 for p, cv in zip(greedy_data['precision'], greedy_data['cv'])
                              if p < 3 and cv <= 1.0)

    categories = ['High Prec.\n+ Adaptive', 'High Prec.\n+ Stable', 'Low Prec.\n+ Adaptive', 'Low Prec.\n+ Stable']
    counts = [high_precision_adaptive, high_precision_stable, low_precision_adaptive, low_precision_stable]
    colors_cat = ['darkgreen', 'lightgreen', 'orange', 'lightcoral']

    bars = ax4.bar(categories, counts, color=colors_cat, alpha=0.8)
    ax4.set_ylabel('Count', fontsize=9)
    ax4.set_title('GreedyLR Behavior Types', fontsize=11, fontweight='bold')
    ax4.tick_params(axis='x', labelsize=8)

    # Add percentages
    total = sum(counts)
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + total*0.01,
                f'{count/total*100:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig('final_plots/adaptive_behavior_analysis_compact.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Compact plots created successfully!")

if __name__ == "__main__":
    create_compact_plots()