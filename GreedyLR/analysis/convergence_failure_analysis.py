#!/usr/bin/env python3
"""
Advanced Analysis: Convergence Rate and Failure Mode Analysis for GreedyLR Study
Performs convergence rate analysis and failure mode identification from experimental data.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Try to import seaborn, use fallback if not available
try:
    import seaborn as sns
    sns.set_palette("husl")
except ImportError:
    print("Seaborn not available, using matplotlib defaults")

# Set up plotting style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10

def load_experimental_data():
    """Load and parse experimental data from JSON files"""
    print("Loading experimental data...")

    # Load main results
    with open('robust_results.json', 'r') as f:
        robust_results = json.load(f)

    # Load recovery results if available
    recovery_data = {}
    if Path('test_recovery_results.json').exists():
        with open('test_recovery_results.json', 'r') as f:
            recovery_data = json.load(f)

    print(f"Loaded {len(robust_results)} main experiments")
    if recovery_data:
        print(f"Loaded {len(recovery_data)} recovery experiments")

    return robust_results, recovery_data

def analyze_convergence_rates(data):
    """Analyze convergence rates and time-to-convergence metrics"""
    print("\n=== CONVERGENCE RATE ANALYSIS ===")

    convergence_metrics = {
        'scheduler': [],
        'architecture': [],
        'noise_type': [],
        'time_to_90_percent': [],
        'time_to_95_percent': [],
        'time_to_99_percent': [],
        'final_loss': [],
        'convergence_stability': [],
        'early_stopping_performance': []
    }

    scheduler_convergence = defaultdict(list)

    # Handle both list and dict data formats
    if isinstance(data, list):
        data_items = enumerate(data)
    else:
        data_items = data.items()

    for exp_id, result in data_items:
        if 'loss_history' not in result or not result['loss_history']:
            continue

        loss_history = result['loss_history']
        scheduler = result.get('scheduler', 'unknown')
        architecture = result.get('architecture', 'unknown')
        noise_type = result.get('noise_type', 'none')

        if len(loss_history) < 10:  # Skip very short runs
            continue

        # Calculate final performance targets
        final_loss = loss_history[-1]
        initial_loss = loss_history[0]

        # Skip if no improvement or initial loss is 0
        if final_loss >= initial_loss or initial_loss == 0:
            continue

        # Calculate convergence targets
        improvement_90 = initial_loss - 0.9 * (initial_loss - final_loss)
        improvement_95 = initial_loss - 0.95 * (initial_loss - final_loss)
        improvement_99 = initial_loss - 0.99 * (initial_loss - final_loss)

        # Find time to reach targets
        time_90 = len(loss_history)
        time_95 = len(loss_history)
        time_99 = len(loss_history)

        for i, loss in enumerate(loss_history):
            if loss <= improvement_90 and time_90 == len(loss_history):
                time_90 = i
            if loss <= improvement_95 and time_95 == len(loss_history):
                time_95 = i
            if loss <= improvement_99 and time_99 == len(loss_history):
                time_99 = i

        # Calculate convergence stability (coefficient of variation in last 20% of training)
        last_20_percent = int(0.8 * len(loss_history))
        if last_20_percent < len(loss_history) - 5:
            late_losses = loss_history[last_20_percent:]
            stability = np.std(late_losses) / (np.mean(late_losses) + 1e-8)
        else:
            stability = np.nan

        # Early stopping performance (performance at 50% of training)
        early_stop_idx = len(loss_history) // 2
        early_performance = loss_history[early_stop_idx] if early_stop_idx < len(loss_history) else final_loss

        # Store metrics
        convergence_metrics['scheduler'].append(scheduler)
        convergence_metrics['architecture'].append(architecture)
        convergence_metrics['noise_type'].append(noise_type)
        convergence_metrics['time_to_90_percent'].append(time_90)
        convergence_metrics['time_to_95_percent'].append(time_95)
        convergence_metrics['time_to_99_percent'].append(time_99)
        convergence_metrics['final_loss'].append(final_loss)
        convergence_metrics['convergence_stability'].append(stability)
        convergence_metrics['early_stopping_performance'].append(early_performance)

        scheduler_convergence[scheduler].append({
            'time_90': time_90,
            'time_95': time_95,
            'time_99': time_99,
            'stability': stability,
            'early_perf': early_performance,
            'final_perf': final_loss
        })

    # Convert to DataFrame for analysis
    df = pd.DataFrame(convergence_metrics)
    df = df.dropna()

    if len(df) == 0:
        print("No valid convergence data found")
        return None, None

    print(f"Analyzing {len(df)} valid convergence experiments")

    # Statistical summary
    print("\n--- CONVERGENCE SPEED COMPARISON ---")
    convergence_summary = df.groupby('scheduler').agg({
        'time_to_90_percent': ['median', 'mean', 'std'],
        'time_to_95_percent': ['median', 'mean', 'std'],
        'time_to_99_percent': ['median', 'mean', 'std'],
        'convergence_stability': ['median', 'mean', 'std'],
        'early_stopping_performance': ['median', 'mean', 'std']
    }).round(3)

    print(convergence_summary)

    return df, scheduler_convergence

def create_convergence_plots(df, scheduler_convergence):
    """Create comprehensive convergence analysis plots"""
    if df is None or df.empty:
        print("No data available for convergence plotting")
        return

    print("\nGenerating convergence analysis plots...")

    # Plot 1: Time-to-convergence comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Convergence Rate Analysis: Time-to-Target Performance', fontsize=16, fontweight='bold')

    metrics = ['time_to_90_percent', 'time_to_95_percent', 'time_to_99_percent', 'convergence_stability']
    titles = ['Time to 90% of Final Performance', 'Time to 95% of Final Performance',
              'Time to 99% of Final Performance', 'Convergence Stability (CV)']

    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i//2, i%2]

        # Box plot comparison
        scheduler_data = []
        scheduler_labels = []
        for scheduler in df['scheduler'].unique():
            data = df[df['scheduler'] == scheduler][metric].dropna()
            if len(data) > 0:
                scheduler_data.append(data)
                scheduler_labels.append(scheduler)

        if scheduler_data:
            bp = ax.boxplot(scheduler_data, labels=scheduler_labels, patch_artist=True)
            colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)

        ax.set_title(title, fontweight='bold')
        ax.set_ylabel('Steps' if 'time_to' in metric else 'Coefficient of Variation')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('final_plots/convergence_rate_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Early stopping analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Early Stopping Performance Analysis', fontsize=16, fontweight='bold')

    # Early vs final performance correlation
    schedulers = df['scheduler'].unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(schedulers)))

    for scheduler, color in zip(schedulers, colors):
        scheduler_df = df[df['scheduler'] == scheduler]
        ax1.scatter(scheduler_df['early_stopping_performance'], scheduler_df['final_loss'],
                   alpha=0.6, label=scheduler, color=color, s=30)

    ax1.set_xlabel('Performance at 50% Training (Early Stop)')
    ax1.set_ylabel('Final Performance')
    ax1.set_title('Early vs Final Performance Correlation', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    # Early stopping advantage
    early_ratios = []
    scheduler_names = []
    for scheduler in schedulers:
        scheduler_df = df[df['scheduler'] == scheduler]
        if len(scheduler_df) > 0:
            ratio = scheduler_df['final_loss'].median() / scheduler_df['early_stopping_performance'].median()
            early_ratios.append(ratio)
            scheduler_names.append(scheduler)

    bars = ax2.bar(scheduler_names, early_ratios, color=colors[:len(scheduler_names)], alpha=0.7)
    ax2.set_ylabel('Final/Early Performance Ratio')
    ax2.set_title('Early Stopping Efficiency\n(Lower = Better Early Performance)', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, ratio in zip(bars, early_ratios):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{ratio:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('final_plots/early_stopping_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Convergence analysis plots saved to final_plots/")

def analyze_failure_modes(data):
    """Identify and analyze failure modes for each scheduler"""
    print("\n=== FAILURE MODE ANALYSIS ===")

    failure_analysis = {
        'scheduler': [],
        'architecture': [],
        'noise_type': [],
        'final_loss': [],
        'improvement_ratio': [],
        'convergence_failure': [],
        'instability_score': [],
        'performance_rank': []
    }

    # Group experiments by configuration for fair comparison
    config_groups = defaultdict(list)

    # Handle both list and dict data formats
    if isinstance(data, list):
        data_items = enumerate(data)
    else:
        data_items = data.items()

    for exp_id, result in data_items:
        if 'loss_history' not in result or not result['loss_history']:
            continue

        architecture = result.get('architecture', 'unknown')
        noise_type = result.get('noise_type', 'none')
        scheduler = result.get('scheduler', 'unknown')

        config_key = f"{architecture}_{noise_type}"
        config_groups[config_key].append((scheduler, result))

    # Analyze each configuration
    scheduler_failures = defaultdict(list)

    for config, experiments in config_groups.items():
        if len(experiments) < 2:  # Need at least 2 schedulers for comparison
            continue

        architecture, noise_type = config.split('_', 1)

        # Calculate performance metrics for this configuration
        config_results = []
        for scheduler, result in experiments:
            loss_history = result['loss_history']
            if len(loss_history) < 10:
                continue

            initial_loss = loss_history[0]
            final_loss = loss_history[-1]

            # Skip if no improvement or division by zero
            if initial_loss == 0 or final_loss >= initial_loss:
                continue

            improvement_ratio = (initial_loss - final_loss) / initial_loss

            # Calculate instability (variance in last 25% of training)
            last_quarter = int(0.75 * len(loss_history))
            late_losses = loss_history[last_quarter:]
            instability = np.std(late_losses) / (np.mean(late_losses) + 1e-8)

            # Check for convergence failure (loss increases in final 20%)
            final_20_percent = int(0.8 * len(loss_history))
            if final_20_percent < len(loss_history) - 2:
                trend_slope = np.polyfit(range(len(loss_history[final_20_percent:])),
                                       loss_history[final_20_percent:], 1)[0]
                convergence_failure = trend_slope > 0
            else:
                convergence_failure = False

            config_results.append({
                'scheduler': scheduler,
                'final_loss': final_loss,
                'improvement_ratio': improvement_ratio,
                'instability': instability,
                'convergence_failure': convergence_failure
            })

        if len(config_results) < 2:
            continue

        # Rank performance within this configuration
        config_results.sort(key=lambda x: x['final_loss'])
        for rank, result in enumerate(config_results):
            failure_analysis['scheduler'].append(result['scheduler'])
            failure_analysis['architecture'].append(architecture)
            failure_analysis['noise_type'].append(noise_type)
            failure_analysis['final_loss'].append(result['final_loss'])
            failure_analysis['improvement_ratio'].append(result['improvement_ratio'])
            failure_analysis['convergence_failure'].append(result['convergence_failure'])
            failure_analysis['instability_score'].append(result['instability'])
            failure_analysis['performance_rank'].append(rank + 1)  # 1-indexed

            # Track failures for each scheduler
            if rank == len(config_results) - 1:  # Worst performer
                scheduler_failures[result['scheduler']].append({
                    'config': config,
                    'type': 'worst_performance',
                    'final_loss': result['final_loss'],
                    'details': result
                })

            if result['convergence_failure']:
                scheduler_failures[result['scheduler']].append({
                    'config': config,
                    'type': 'convergence_failure',
                    'final_loss': result['final_loss'],
                    'details': result
                })

            if result['instability'] > 1.0:  # High instability threshold
                scheduler_failures[result['scheduler']].append({
                    'config': config,
                    'type': 'instability',
                    'instability': result['instability'],
                    'details': result
                })

    # Convert to DataFrame
    df = pd.DataFrame(failure_analysis)

    if len(df) == 0:
        print("No valid failure analysis data found")
        return None, scheduler_failures

    print(f"Analyzing {len(df)} experiments for failure modes")

    # Statistical summary of failures
    print("\n--- FAILURE MODE SUMMARY ---")
    for scheduler, failures in scheduler_failures.items():
        print(f"\n{scheduler}:")
        failure_types = defaultdict(int)
        for failure in failures:
            failure_types[failure['type']] += 1

        for failure_type, count in failure_types.items():
            total_experiments = len(df[df['scheduler'] == scheduler])
            percentage = (count / total_experiments) * 100 if total_experiments > 0 else 0
            print(f"  {failure_type}: {count} cases ({percentage:.1f}%)")

    # Rank analysis
    print("\n--- PERFORMANCE RANKING ANALYSIS ---")
    rank_summary = df.groupby('scheduler')['performance_rank'].agg(['mean', 'median', 'std']).round(3)
    print(rank_summary)

    return df, scheduler_failures

def create_failure_mode_plots(df, scheduler_failures):
    """Create failure mode analysis visualizations"""
    if df is None or df.empty:
        print("No data available for failure mode plotting")
        return

    print("\nGenerating failure mode analysis plots...")

    # Plot 1: Failure mode frequency analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Failure Mode Analysis: Scheduler Robustness Comparison', fontsize=16, fontweight='bold')

    # Failure frequency by scheduler
    schedulers = list(scheduler_failures.keys())
    failure_counts = {scheduler: len(failures) for scheduler, failures in scheduler_failures.items()}
    total_experiments = {scheduler: len(df[df['scheduler'] == scheduler]) for scheduler in schedulers}

    failure_rates = [failure_counts.get(s, 0) / max(total_experiments.get(s, 1), 1) * 100
                    for s in schedulers]

    bars1 = axes[0,0].bar(schedulers, failure_rates, alpha=0.7, color=['red', 'orange', 'yellow', 'green'][:len(schedulers)])
    axes[0,0].set_title('Overall Failure Rate by Scheduler', fontweight='bold')
    axes[0,0].set_ylabel('Failure Rate (%)')
    axes[0,0].tick_params(axis='x', rotation=45)
    axes[0,0].grid(True, alpha=0.3)

    for bar, rate in zip(bars1, failure_rates):
        height = bar.get_height()
        axes[0,0].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                      f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')

    # Performance ranking distribution
    rank_data = []
    rank_labels = []
    for scheduler in df['scheduler'].unique():
        ranks = df[df['scheduler'] == scheduler]['performance_rank']
        if len(ranks) > 0:
            rank_data.append(ranks)
            rank_labels.append(scheduler)

    bp = axes[0,1].boxplot(rank_data, labels=rank_labels, patch_artist=True)
    colors = ['lightcoral', 'lightsalmon', 'lightblue', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)

    axes[0,1].set_title('Performance Ranking Distribution\n(Lower = Better)', fontweight='bold')
    axes[0,1].set_ylabel('Rank (1=Best, 4=Worst)')
    axes[0,1].tick_params(axis='x', rotation=45)
    axes[0,1].grid(True, alpha=0.3)

    # Instability analysis
    instability_data = []
    instability_labels = []
    for scheduler in df['scheduler'].unique():
        instability = df[df['scheduler'] == scheduler]['instability_score'].dropna()
        if len(instability) > 0:
            instability_data.append(instability)
            instability_labels.append(scheduler)

    bp2 = axes[1,0].boxplot(instability_data, labels=instability_labels, patch_artist=True)
    for patch, color in zip(bp2['boxes'], colors[:len(bp2['boxes'])]):
        patch.set_facecolor(color)

    axes[1,0].set_title('Training Instability Scores\n(Lower = More Stable)', fontweight='bold')
    axes[1,0].set_ylabel('Instability Score (CV)')
    axes[1,0].tick_params(axis='x', rotation=45)
    axes[1,0].set_yscale('log')
    axes[1,0].grid(True, alpha=0.3)

    # Worst-case scenario analysis
    worst_case_losses = []
    scheduler_names = []
    for scheduler in df['scheduler'].unique():
        scheduler_df = df[df['scheduler'] == scheduler]
        if len(scheduler_df) > 0:
            # 95th percentile of final losses (worst-case scenarios)
            worst_case = scheduler_df['final_loss'].quantile(0.95)
            worst_case_losses.append(worst_case)
            scheduler_names.append(scheduler)

    bars2 = axes[1,1].bar(scheduler_names, worst_case_losses, alpha=0.7,
                         color=['darkred', 'darkorange', 'gold', 'darkgreen'][:len(scheduler_names)])
    axes[1,1].set_title('Worst-Case Performance (95th Percentile)', fontweight='bold')
    axes[1,1].set_ylabel('Final Loss (95th Percentile)')
    axes[1,1].tick_params(axis='x', rotation=45)
    axes[1,1].set_yscale('log')
    axes[1,1].grid(True, alpha=0.3)

    for bar, loss in zip(bars2, worst_case_losses):
        height = bar.get_height()
        axes[1,1].text(bar.get_x() + bar.get_width()/2., height * 1.1,
                      f'{loss:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('final_plots/failure_mode_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Failure mode breakdown by noise type
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Failure Mode Analysis by Condition', fontsize=16, fontweight='bold')

    # Failure rate by noise type
    noise_failure_data = defaultdict(lambda: defaultdict(int))
    noise_total_data = defaultdict(lambda: defaultdict(int))

    for noise_type in df['noise_type'].unique():
        for scheduler in df['scheduler'].unique():
            scheduler_noise_df = df[(df['scheduler'] == scheduler) & (df['noise_type'] == noise_type)]
            total = len(scheduler_noise_df)
            failures = len(scheduler_noise_df[scheduler_noise_df['convergence_failure'] == True])

            noise_failure_data[noise_type][scheduler] = failures
            noise_total_data[noise_type][scheduler] = total

    # Create stacked bar chart for noise types
    noise_types = list(noise_failure_data.keys())[:6]  # Limit to 6 for readability
    schedulers = df['scheduler'].unique()

    x_pos = np.arange(len(noise_types))
    width = 0.2

    for i, scheduler in enumerate(schedulers):
        failure_rates = []
        for noise_type in noise_types:
            total = noise_total_data[noise_type].get(scheduler, 1)
            failures = noise_failure_data[noise_type].get(scheduler, 0)
            rate = (failures / total) * 100 if total > 0 else 0
            failure_rates.append(rate)

        ax1.bar(x_pos + i * width, failure_rates, width, label=scheduler, alpha=0.8)

    ax1.set_xlabel('Noise Type')
    ax1.set_ylabel('Convergence Failure Rate (%)')
    ax1.set_title('Convergence Failures by Noise Condition', fontweight='bold')
    ax1.set_xticks(x_pos + width * 1.5)
    ax1.set_xticklabels(noise_types, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Performance consistency across architectures
    arch_performance = defaultdict(lambda: defaultdict(list))

    for arch in df['architecture'].unique():
        for scheduler in df['scheduler'].unique():
            scheduler_arch_df = df[(df['scheduler'] == scheduler) & (df['architecture'] == arch)]
            if len(scheduler_arch_df) > 0:
                median_rank = scheduler_arch_df['performance_rank'].median()
                arch_performance[arch][scheduler].append(median_rank)

    # Show architecture where each scheduler performs worst
    worst_archs = {}
    for scheduler in schedulers:
        worst_rank = 0
        worst_arch = ""
        for arch, scheduler_ranks in arch_performance.items():
            if scheduler in scheduler_ranks and scheduler_ranks[scheduler]:
                rank = np.mean(scheduler_ranks[scheduler])
                if rank > worst_rank:
                    worst_rank = rank
                    worst_arch = arch
        worst_archs[scheduler] = (worst_arch, worst_rank)

    scheduler_names = list(worst_archs.keys())
    worst_ranks = [worst_archs[s][1] for s in scheduler_names]
    worst_arch_names = [worst_archs[s][0][:15] + "..." if len(worst_archs[s][0]) > 15 else worst_archs[s][0]
                       for s in scheduler_names]

    bars = ax2.bar(scheduler_names, worst_ranks, alpha=0.7,
                   color=['darkred', 'darkorange', 'gold', 'darkgreen'][:len(scheduler_names)])
    ax2.set_ylabel('Worst Average Rank')
    ax2.set_title('Worst-Case Architecture Performance\n(Higher = Worse)', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)

    # Add architecture labels
    for bar, arch_name in zip(bars, worst_arch_names):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                arch_name, ha='center', va='bottom', fontsize=8, rotation=45)

    plt.tight_layout()
    plt.savefig('final_plots/failure_mode_conditions.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Failure mode analysis plots saved to final_plots/")

def main():
    """Main analysis function"""
    print("Starting Convergence Rate and Failure Mode Analysis")
    print("=" * 60)

    # Load data
    robust_results, recovery_data = load_experimental_data()

    if not robust_results:
        print("No experimental data found!")
        return

    # Analyze convergence rates
    convergence_df, scheduler_convergence = analyze_convergence_rates(robust_results)
    if convergence_df is not None:
        create_convergence_plots(convergence_df, scheduler_convergence)

    # Analyze failure modes
    failure_df, scheduler_failures = analyze_failure_modes(robust_results)
    if failure_df is not None:
        create_failure_mode_plots(failure_df, scheduler_failures)

    print("\n" + "=" * 60)
    print("Analysis complete! Check final_plots/ for generated visualizations.")

    return convergence_df, failure_df, scheduler_convergence, scheduler_failures

if __name__ == "__main__":
    convergence_df, failure_df, scheduler_convergence, scheduler_failures = main()