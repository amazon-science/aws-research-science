#!/usr/bin/env python3
"""
Learning Rate Scheduler Comparison Tool

This script analyzes TensorBoard logs from Greedy LR and Cosine LR training runs
to generate detailed comparisons, visualizations, and performance reports.

Key features:
- Comparative analysis of training/evaluation metrics
- Visualization of learning rate schedules
- Training efficiency statistics and recommendations
- Detailed loss progression analysis at different training stages
- Comprehensive markdown report generation

Usage:
  python compare_schedulers.py --greedy-dir PATH --cosine-dir PATH [--output-dir PATH]
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing import event_accumulator
from datetime import datetime

# Configure plotting style
plt.style.use('ggplot')
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.figsize': (12, 8)
})

# Global color scheme
COLORS = {
    'greedy': '#1f77b4',  # Blue
    'cosine': '#ff7f0e',  # Orange
}

def parse_args():
    """
    Parse command-line arguments for the comparison script.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description='Compare Greedy LR and Cosine LR training results')
    
    parser.add_argument(
        '--greedy-dir',
        type=str,
        default='./logs/redpajama/meta-llama/Llama-3.2-1B/run_1/greedy/2025-06-06/tensorboard',
        help='Directory containing Greedy LR TensorBoard logs'
    )
    
    parser.add_argument(
        '--cosine-dir',
        type=str,
        default='./logs/redpajama/meta-llama/Llama-3.2-1B/run_cosine250/cosine/2025-06-09/tensorboard',
        help='Directory containing Cosine LR TensorBoard logs (will search for tensorboard subdirectory)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./comparison_results',
        help='Directory to save comparison results'
    )
    
    return parser.parse_args()

def load_event_data(log_dir):
    """
    Load TensorBoard event data from the specified directory.
    
    This function finds and parses TensorBoard logs, extracting scalar metrics
    including timestamps, step information, and metric values. It also calculates
    total training time from the earliest to latest event.
    
    Args:
        log_dir (str): Path to directory containing TensorBoard logs
        
    Returns:
        dict: Dictionary of metrics data and training time information
    
    Raises:
        FileNotFoundError: If log directory or event files cannot be found
    """
    if not os.path.exists(log_dir):
        raise FileNotFoundError(f"Log directory not found: {log_dir}")
    
    print(f"Loading event data from {log_dir}...")
    
    # Check if this directory contains event files directly
    event_files = [f for f in os.listdir(log_dir) if f.startswith('events.out.tfevents')]
    
    # If no event files found, look for tensorboard subdirectory
    if not event_files:
        # Look for date-based subdirectories that contain tensorboard
        potential_dirs = []
        for item in os.listdir(log_dir):
            item_path = os.path.join(log_dir, item)
            if os.path.isdir(item_path):
                tensorboard_path = os.path.join(item_path, 'tensorboard')
                if os.path.exists(tensorboard_path):
                    potential_dirs.append(tensorboard_path)
        
        if potential_dirs:
            # Use the most recent directory (sort by modification time)
            potential_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            log_dir = potential_dirs[0]
            print(f"Found tensorboard directory: {log_dir}")
            event_files = [f for f in os.listdir(log_dir) if f.startswith('events.out.tfevents')]
    
    if not event_files:
        raise FileNotFoundError(f"No event files found in {log_dir} or its tensorboard subdirectories")
    
    # Sort by timestamp to get the most recent first
    event_files.sort(reverse=True)
    event_file = os.path.join(log_dir, event_files[0])
    
    # Load the events
    ea = event_accumulator.EventAccumulator(
        event_file,
        size_guidance={
            event_accumulator.SCALARS: 0,  # Load all scalars
        }
    )
    ea.Reload()
    
    # Get available tags (metrics)
    tags = ea.Tags()['scalars']
    print(f"Available metrics: {tags}")
    
    # Extract data with wall time information
    data = {}
    training_start_time = None
    training_end_time = None
    
    for tag in tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        wall_times = [e.wall_time for e in events]
        
        data[tag] = pd.DataFrame({
            'step': steps, 
            'value': values,
            'wall_time': wall_times
        })
        
        # Track overall training time from any metric
        if events:
            if training_start_time is None or wall_times[0] < training_start_time:
                training_start_time = wall_times[0]
            if training_end_time is None or wall_times[-1] > training_end_time:
                training_end_time = wall_times[-1]
    
    # Calculate total training time
    if training_start_time and training_end_time:
        total_time_seconds = training_end_time - training_start_time
        data['_training_info'] = {
            'start_time': training_start_time,
            'end_time': training_end_time,
            'total_seconds': total_time_seconds,
            'total_minutes': total_time_seconds / 60,
            'total_hours': total_time_seconds / 3600
        }
    
    return data

def truncate_to_equal_steps(greedy_data, cosine_data):
    """
    Truncate both datasets to have equal number of steps for fair comparison.
    
    This function finds the common step range that exists in both datasets
    and trims both datasets to this range for fair, direct comparison.
    It also recalculates training time metrics based on the truncated data.
    
    Args:
        greedy_data (dict): Data dictionary from Greedy LR training
        cosine_data (dict): Data dictionary from Cosine LR training
        
    Returns:
        tuple: (truncated_greedy_data, truncated_cosine_data)
    """
    # Find the key metrics we're interested in to determine step ranges
    key_metrics = ['train/loss', 'eval/loss', 'train/learning_rate', 'train/grad_norm']
    
    # Find available metrics that we want to compare
    greedy_available = {k: v for k, v in greedy_data.items() if not k.startswith('_') and k in key_metrics}
    cosine_available = {k: v for k, v in cosine_data.items() if not k.startswith('_') and k in key_metrics}
    
    if not greedy_available or not cosine_available:
        print("Warning: No matching key metrics found between datasets")
        return greedy_data, cosine_data
    
    # Find minimum and maximum steps for each dataset
    greedy_min_steps = min([df['step'].min() for df in greedy_available.values()])
    greedy_max_steps = max([df['step'].max() for df in greedy_available.values()])
    cosine_min_steps = min([df['step'].min() for df in cosine_available.values()])
    cosine_max_steps = max([df['step'].max() for df in cosine_available.values()])
    
    # Use the most restrictive range (latest start, earliest end)
    common_start = max(greedy_min_steps, cosine_min_steps)
    common_end = min(greedy_max_steps, cosine_max_steps)
    
    print(f"Truncating to common step range: {common_start} to {common_end}")
    
    # Truncate all metrics to the common range
    truncated_greedy = {}
    truncated_cosine = {}
    
    for key, df in greedy_data.items():
        if key.startswith('_'):  # Preserve metadata but update training time
            if key == '_training_info':
                # Recalculate training time for the truncated range
                updated_training_info = df.copy()
                truncated_greedy[key] = updated_training_info
            else:
                truncated_greedy[key] = df
        else:
            mask = (df['step'] >= common_start) & (df['step'] <= common_end)
            truncated_df = df[mask].reset_index(drop=True)
            truncated_greedy[key] = truncated_df
            
            # Update training time based on truncated data
            if key in key_metrics and '_training_info' in greedy_data and not truncated_df.empty:
                start_time = truncated_df['wall_time'].iloc[0]
                end_time = truncated_df['wall_time'].iloc[-1]
                actual_duration = end_time - start_time
                truncated_greedy['_training_info'] = {
                    'start_time': start_time,
                    'end_time': end_time,
                    'total_seconds': actual_duration,
                    'total_minutes': actual_duration / 60,
                    'total_hours': actual_duration / 3600
                }
    
    for key, df in cosine_data.items():
        if key.startswith('_'):  # Preserve metadata but update training time
            if key == '_training_info':
                # Recalculate training time for the truncated range
                updated_training_info = df.copy()
                truncated_cosine[key] = updated_training_info
            else:
                truncated_cosine[key] = df
        else:
            mask = (df['step'] >= common_start) & (df['step'] <= common_end)
            truncated_df = df[mask].reset_index(drop=True)
            truncated_cosine[key] = truncated_df
            
            # Update training time based on truncated data
            if key in key_metrics and '_training_info' in cosine_data and not truncated_df.empty:
                start_time = truncated_df['wall_time'].iloc[0]
                end_time = truncated_df['wall_time'].iloc[-1]
                actual_duration = end_time - start_time
                truncated_cosine['_training_info'] = {
                    'start_time': start_time,
                    'end_time': end_time,
                    'total_seconds': actual_duration,
                    'total_minutes': actual_duration / 60,
                    'total_hours': actual_duration / 3600
                }
    
    return truncated_greedy, truncated_cosine

def create_comparison_plots(greedy_data, cosine_data, output_dir):
    """
    Create comparison plots between Greedy LR and Cosine LR data.
    
    Generates visualizations comparing:
    - Training loss
    - Evaluation loss
    - Learning rate schedules
    - Gradient norm
    
    Also calculates improvement percentages and generates a loss breakdown
    at different training milestones.
    
    Args:
        greedy_data (dict): Data dictionary from Greedy LR training
        cosine_data (dict): Data dictionary from Cosine LR training
        output_dir (str): Directory to save output plots
        
    Returns:
        tuple: (plots_metadata, loss_breakdown)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Truncate both datasets to equal steps
    greedy_data, cosine_data = truncate_to_equal_steps(greedy_data, cosine_data)
    
    # Store the actual step range used for reporting
    actual_max_steps = 0
    if greedy_data:
        for key, df in greedy_data.items():
            if not key.startswith('_') and not df.empty:
                actual_max_steps = max(actual_max_steps, df['step'].max())
    
    # Define the specific metrics we want to compare
    target_metrics = [
        'train/loss', 
        'eval/loss',
        'train/learning_rate',
        'train/grad_norm'
    ]
    
    # Find common metrics from our target list
    greedy_metrics = set(greedy_data.keys())
    cosine_metrics = set(cosine_data.keys())
    common_metrics = greedy_metrics.intersection(cosine_metrics).intersection(target_metrics)
    
    print(f"Creating comparison plots for filtered metrics: {common_metrics}")
    
    # Sort metrics in desired order
    plots = []
    loss_breakdown = None
    
    for metric in target_metrics:
        if metric not in common_metrics:
            print(f"Skipping {metric} - not available in both datasets")
            continue
            
        print(f"Plotting {metric}...")
        fig, ax = plt.subplots()
        
        # Plot Greedy LR data
        greedy_df = greedy_data[metric]
        ax.plot(greedy_df['step'], greedy_df['value'], 
                color=COLORS['greedy'], label='Greedy LR', 
                marker='o', markersize=3, linestyle='-')
        
        # Plot Cosine LR data
        cosine_df = cosine_data[metric]
        ax.plot(cosine_df['step'], cosine_df['value'], 
                color=COLORS['cosine'], label='Cosine LR', 
                marker='s', markersize=3, linestyle='-')
        
        # Calculate final values and improvement
        final_greedy = greedy_df['value'].iloc[-1]
        final_cosine = cosine_df['value'].iloc[-1]
        
        # Format metric name for display
        display_name = metric.replace('/', ' - ').replace('train/learning_rate', 'train/learning_rate').title()
        if metric == 'train/learning_rate':
            display_name = 'Train - Learning Rate'
        
        # Customize the plot based on the metric type
        if 'loss' in metric.lower():
            improvement = ((final_greedy - final_cosine) / final_greedy) * 100
            improvement_text = f"Lower is better: {'Cosine' if final_cosine < final_greedy else 'Greedy'} by {abs(improvement):.2f}%"
            ax.set_title(f"{display_name} Comparison\n{improvement_text}")
        elif 'learning_rate' in metric.lower():
            ax.set_title(f"{display_name} Schedule Comparison")
            # Use log scale for learning rate
            ax.set_yscale('log')
        else:
            ax.set_title(f"{display_name} Comparison")
        
        # Set labels and legend
        ax.set_xlabel('Training Steps')
        ax.set_ylabel(display_name)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add annotations for final values
        ax.annotate(f'{final_greedy:.4f}', 
                    xy=(greedy_df['step'].iloc[-1], final_greedy),
                    xytext=(10, 10), textcoords='offset points',
                    color=COLORS['greedy'], fontweight='bold')
        
        ax.annotate(f'{final_cosine:.4f}', 
                    xy=(cosine_df['step'].iloc[-1], final_cosine),
                    xytext=(10, -15), textcoords='offset points',
                    color=COLORS['cosine'], fontweight='bold')
        
        # Save the plot
        filename = f"{metric.replace('/', '_')}.png"
        filepath = os.path.join(output_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=150)
        plt.close(fig)
        
        plots.append({
            'metric': metric,
            'display_name': display_name,
            'filename': filename,
            'greedy_final': final_greedy,
            'cosine_final': final_cosine
        })
        
        # Generate detailed loss breakdown for train/loss
        if metric == 'train/loss':
            loss_breakdown = generate_loss_breakdown(greedy_df, cosine_df)
    
    # Create a summary plot with training loss
    if 'train/loss' in common_metrics:
        create_summary_plot(greedy_data, cosine_data, output_dir)
    
    # Create a separate learning rate plot
    if 'train/learning_rate' in common_metrics:
        create_learning_rate_plot(greedy_data, cosine_data, output_dir)
    
    return plots, loss_breakdown

def generate_loss_breakdown(greedy_df, cosine_df):
    """
    Generate detailed breakdown of train loss at different training stages.
    
    Analyzes training loss at key milestones (10%, 50%, 90%, 100%) to show
    how schedulers perform at different phases of the training process.
    
    Args:
        greedy_df (pandas.DataFrame): Training loss data for Greedy LR
        cosine_df (pandas.DataFrame): Training loss data for Cosine LR
        
    Returns:
        list: List of dictionaries with milestone data and comparisons
    """
    # Find the total number of steps
    max_steps = min(greedy_df['step'].max(), cosine_df['step'].max())
    
    # Calculate milestone steps
    milestones = {
        '10%': int(max_steps * 0.1),
        '50%': int(max_steps * 0.5),
        '90%': int(max_steps * 0.9),
        '100%': max_steps
    }
    
    breakdown = []
    
    for milestone, target_step in milestones.items():
        # Find the closest step in each dataset
        greedy_idx = (greedy_df['step'] - target_step).abs().idxmin()
        cosine_idx = (cosine_df['step'] - target_step).abs().idxmin()
        
        greedy_loss = greedy_df.iloc[greedy_idx]['value']
        cosine_loss = cosine_df.iloc[cosine_idx]['value']
        actual_step = greedy_df.iloc[greedy_idx]['step']  # Use greedy step as reference
        
        # Calculate improvement
        if greedy_loss != 0:
            improvement = ((greedy_loss - cosine_loss) / greedy_loss) * 100
        else:
            improvement = 0
            
        better = 'Cosine' if cosine_loss < greedy_loss else 'Greedy'
        
        breakdown.append({
            'milestone': milestone,
            'step': actual_step,
            'greedy_loss': greedy_loss,
            'cosine_loss': cosine_loss,
            'difference': abs(greedy_loss - cosine_loss),
            'improvement_pct': abs(improvement),
            'better': better
        })
    
    return breakdown

def create_summary_plot(greedy_data, cosine_data, output_dir):
    """
    Create a summary plot focusing on training loss comparison.
    
    Generates a high-visibility plot comparing training loss between
    schedulers with clear annotations showing final values and improvement.
    
    Args:
        greedy_data (dict): Data dictionary from Greedy LR training
        cosine_data (dict): Data dictionary from Cosine LR training
        output_dir (str): Directory to save output plot
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot training loss only
    if 'train/loss' in greedy_data and 'train/loss' in cosine_data:
        greedy_loss = greedy_data['train/loss']
        cosine_loss = cosine_data['train/loss']
        
        # Scale for better visibility
        greedy_loss_scaled = greedy_loss.copy()
        greedy_loss_scaled['value'] = np.clip(greedy_loss_scaled['value'], 0, 10)
        cosine_loss_scaled = cosine_loss.copy()
        cosine_loss_scaled['value'] = np.clip(cosine_loss_scaled['value'], 0, 10)
        
        ax.plot(greedy_loss_scaled['step'], greedy_loss_scaled['value'], 
                color=COLORS['greedy'], label='Greedy LR', 
                marker='o', markersize=4, linestyle='-', linewidth=2)
        ax.plot(cosine_loss_scaled['step'], cosine_loss_scaled['value'], 
                color=COLORS['cosine'], label='Cosine LR', 
                marker='s', markersize=4, linestyle='-', linewidth=2)
        
        final_greedy = greedy_loss['value'].iloc[-1]
        final_cosine = cosine_loss['value'].iloc[-1]
        
        improvement = ((final_greedy - final_cosine) / final_greedy) * 100
        winner = 'Cosine' if final_cosine < final_greedy else 'Greedy'
        
        ax.set_title(f"Training Loss Comparison\n{winner} performs better by {abs(improvement):.2f}%", fontsize=18)
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add annotations for final values
        ax.annotate(f'{final_greedy:.4f}', 
                    xy=(greedy_loss['step'].iloc[-1], min(final_greedy, 10)),
                    xytext=(10, 10), textcoords='offset points',
                    color=COLORS['greedy'], fontweight='bold', fontsize=12)
        
        ax.annotate(f'{final_cosine:.4f}', 
                    xy=(cosine_loss['step'].iloc[-1], min(final_cosine, 10)),
                    xytext=(10, -15), textcoords='offset points',
                    color=COLORS['cosine'], fontweight='bold', fontsize=12)
    
    # Save the summary plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_comparison.png'), dpi=150)
    plt.close(fig)

def create_learning_rate_plot(greedy_data, cosine_data, output_dir):
    """
    Create a specialized learning rate comparison plot with proper scaling.
    
    Generates a detailed visualization of both learning rate schedules,
    with annotations for peaks, final values, and appropriate scaling
    (linear or logarithmic) based on the value ranges.
    
    Args:
        greedy_data (dict): Data dictionary from Greedy LR training
        cosine_data (dict): Data dictionary from Cosine LR training
        output_dir (str): Directory to save output plot
    """
    if 'train/learning_rate' not in greedy_data or 'train/learning_rate' not in cosine_data:
        return
        
    fig, ax = plt.subplots(figsize=(12, 8))
    
    greedy_lr = greedy_data['train/learning_rate']
    cosine_lr = cosine_data['train/learning_rate']
    
    # Use smaller markers and thinner lines to show all data points clearly
    ax.plot(greedy_lr['step'], greedy_lr['value'], 
            color=COLORS['greedy'], label='Greedy LR', 
            marker='o', markersize=2, linestyle='-', linewidth=1.5, alpha=0.8)
    ax.plot(cosine_lr['step'], cosine_lr['value'], 
            color=COLORS['cosine'], label='Cosine LR', 
            marker='s', markersize=2, linestyle='-', linewidth=1.5, alpha=0.8)
    
    ax.set_title("Learning Rate Schedule Comparison", fontsize=18)
    ax.set_xlabel('Training Steps', fontsize=14)
    ax.set_ylabel('Learning Rate', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Use linear scale by default to show cosine curve properly
    # Only consider log scale if the range is extremely large AND there are very small values
    min_lr = min(greedy_lr['value'].min(), cosine_lr['value'].min())
    max_lr = max(greedy_lr['value'].max(), cosine_lr['value'].max())
    
    # Force linear scale for cosine schedulers to show the proper curve shape
    # Only use log scale if min value is very small (< 1e-5) AND range is huge (>1000x)
    use_log_scale = False
    if min_lr > 0:
        lr_range = max_lr / min_lr
        if min_lr < 1e-5 and lr_range > 1000:
            use_log_scale = True
    
    if use_log_scale:
        ax.set_yscale('log')
        # For log scale, format y-axis nicely
        from matplotlib.ticker import LogFormatter
        ax.yaxis.set_major_formatter(LogFormatter(base=10, labelOnlyBase=False))
    else:
        # Use linear scale and format nicely
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Add annotations for initial, peak, and final values to show the complete schedule
    initial_greedy_lr = greedy_lr['value'].iloc[0]
    initial_cosine_lr = cosine_lr['value'].iloc[0]
    final_greedy_lr = greedy_lr['value'].iloc[-1]
    final_cosine_lr = cosine_lr['value'].iloc[-1]
    
    # Find peak values for cosine (should be early in training)
    peak_cosine_lr = cosine_lr['value'].max()
    peak_cosine_step = cosine_lr.loc[cosine_lr['value'].idxmax(), 'step']
    
    # Annotate final values
    ax.annotate(f'Final: {final_greedy_lr:.2e}', 
                xy=(greedy_lr['step'].iloc[-1], final_greedy_lr),
                xytext=(10, 10), textcoords='offset points',
                color=COLORS['greedy'], fontweight='bold', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    ax.annotate(f'Final: {final_cosine_lr:.2e}', 
                xy=(cosine_lr['step'].iloc[-1], final_cosine_lr),
                xytext=(10, -25), textcoords='offset points',
                color=COLORS['cosine'], fontweight='bold', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Annotate peak cosine value to show the curve better
    ax.annotate(f'Peak: {peak_cosine_lr:.2e}', 
                xy=(peak_cosine_step, peak_cosine_lr),
                xytext=(10, 10), textcoords='offset points',
                color=COLORS['cosine'], fontweight='bold', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Ensure all data points are visible by adding some margin
    y_min = min(greedy_lr['value'].min(), cosine_lr['value'].min())
    y_max = max(greedy_lr['value'].max(), cosine_lr['value'].max())
    
    if not use_log_scale:
        margin = (y_max - y_min) * 0.1
        ax.set_ylim(y_min - margin, y_max + margin)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_rate_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

def generate_markdown_report(greedy_dir, cosine_dir, plots, loss_breakdown, greedy_data, cosine_data, output_dir):
    """
    Generate a comprehensive markdown report with comparison results.
    
    Creates a detailed markdown report including:
    - Training configuration details
    - Summary metrics table with improvement calculations
    - Detailed loss breakdown across training stages
    - Training efficiency analysis
    - Conclusions and recommendations
    - References to generated plots
    
    Args:
        greedy_dir (str): Path to Greedy LR logs 
        cosine_dir (str): Path to Cosine LR logs
        plots (list): List of generated plot metadata
        loss_breakdown (list): List of loss milestone comparisons
        greedy_data (dict): Data dictionary from Greedy LR training
        cosine_data (dict): Data dictionary from Cosine LR training
        output_dir (str): Directory to save the report
        
    Returns:
        str: Path to the generated report file
    """
    report_path = os.path.join(output_dir, 'comparison_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# LLaMA 3.2 Training: Greedy LR vs Cosine LR Comparison\n\n")
        
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Training Configuration\n\n")
        f.write("| Parameter | Value |\n")
        f.write("|-----------|-------|\n")
        f.write("| Model | LLaMA 3.2 1B |\n")
        
        # Get step information from the loss breakdown which shows actual steps used
        if loss_breakdown:
            # Use the final step from the loss breakdown which is the actual max
            final_step = loss_breakdown[-1]['step']  # 100% milestone
            f.write(f"| Steps | {int(final_step)} |\n")
        elif plots:
            # Fallback: Find max steps from the truncated data used in plots
            max_steps = 0
            for plot in plots:
                if plot['metric'] in greedy_data and not greedy_data[plot['metric']].empty:
                    current_max = greedy_data[plot['metric']]['step'].max()
                    max_steps = max(max_steps, current_max)
            f.write(f"| Steps | {int(max_steps)} |\n")
        else:
            f.write("| Steps | Unknown |\n")
        
        f.write("| Greedy LR Path | `{}` |\n".format(greedy_dir))
        f.write("| Cosine LR Path | `{}` |\n".format(cosine_dir))
        
        # Add training time information
        greedy_time_info = greedy_data.get('_training_info')
        cosine_time_info = cosine_data.get('_training_info')
        
        if greedy_time_info:
            f.write(f"| Greedy LR Training Time | {greedy_time_info['total_minutes']:.1f} minutes ({greedy_time_info['total_hours']:.2f} hours) |\n")
        if cosine_time_info:
            f.write(f"| Cosine LR Training Time | {cosine_time_info['total_minutes']:.1f} minutes ({cosine_time_info['total_hours']:.2f} hours) |\n")
        
        if greedy_time_info and cosine_time_info:
            time_diff = abs(greedy_time_info['total_minutes'] - cosine_time_info['total_minutes'])
            faster = 'Cosine' if cosine_time_info['total_minutes'] < greedy_time_info['total_minutes'] else 'Greedy'
            f.write(f"| Faster Scheduler | {faster} (by {time_diff:.1f} minutes) |\n")
        
        f.write("\n")
        
        # Summary metrics section
        f.write("## Summary Metrics\n\n")
        f.write("| Metric | Greedy LR | Cosine LR | Difference | Better |\n")
        f.write("|--------|-----------|-----------|------------|-------|\n")
        
        for plot in plots:
            metric = plot['display_name']
            greedy_val = plot['greedy_final']
            cosine_val = plot['cosine_final']
            diff = abs(greedy_val - cosine_val)
            
            # Determine which is better
            if 'loss' in plot['metric'].lower():
                better = 'Cosine' if cosine_val < greedy_val else 'Greedy'
                perc_diff = abs((greedy_val - cosine_val) / greedy_val * 100)
                diff_text = f"{diff:.4f} ({perc_diff:.2f}%)"
            else:
                # For metrics where higher might be better, just show the difference
                better = 'N/A'
                diff_text = f"{diff:.4f}"
            
            f.write(f"| {metric} | {greedy_val:.4f} | {cosine_val:.4f} | {diff_text} | {better} |\n")
        
        # Add detailed train loss breakdown
        if loss_breakdown:
            f.write("\n## Train Loss Breakdown by Training Stage\n\n")
            f.write("| Training Stage | Step | Greedy LR Loss | Cosine LR Loss | Difference | Improvement | Better |\n")
            f.write("|----------------|------|----------------|----------------|------------|-------------|--------|\n")
            
            for breakdown in loss_breakdown:
                milestone = breakdown['milestone']
                step = int(breakdown['step'])
                greedy_loss = breakdown['greedy_loss']
                cosine_loss = breakdown['cosine_loss']
                difference = breakdown['difference']
                improvement_pct = breakdown['improvement_pct']
                better = breakdown['better']
                
                f.write(f"| {milestone} | {step} | {greedy_loss:.4f} | {cosine_loss:.4f} | {difference:.4f} | {improvement_pct:.2f}% | {better} |\n")
        
        f.write("\n## Detailed Comparison\n\n")
        
        # Add the summary plot first
        f.write("### Summary Plot\n\n")
        f.write("![Summary Comparison](summary_comparison.png)\n\n")
        
        # Add separate learning rate comparison plot
        f.write("### Learning Rate Schedule Comparison\n\n")
        f.write("![Learning Rate Schedule Comparison](learning_rate_comparison.png)\n\n")
        f.write("This separate plot shows the learning rate schedules with optimized scaling to clearly display the cosine decay pattern.\n\n")
        
        # Add individual metric plots
        for plot in plots:
            metric = plot['display_name']
            filename = plot['filename']
            
            f.write(f"### {metric}\n\n")
            f.write(f"![{metric} Comparison]({filename})\n\n")
            
            # Add metric-specific commentary
            if 'loss' in plot['metric'].lower():
                greedy_val = plot['greedy_final']
                cosine_val = plot['cosine_final']
                
                if cosine_val < greedy_val:
                    perc_improvement = (greedy_val - cosine_val) / greedy_val * 100
                    f.write(f"**Cosine LR performs better** with a {perc_improvement:.2f}% lower loss.\n\n")
                else:
                    perc_improvement = (cosine_val - greedy_val) / cosine_val * 100
                    f.write(f"**Greedy LR performs better** with a {perc_improvement:.2f}% lower loss.\n\n")
        
        f.write("## Training Efficiency Analysis\n\n")
        
        if greedy_time_info and cosine_time_info:
            # Calculate steps per minute for efficiency
            greedy_steps = max([df['step'].max() for df in [greedy_data.get(p['metric'], pd.DataFrame({'step': [1]})) for p in plots] if not df.empty])
            cosine_steps = max([df['step'].max() for df in [cosine_data.get(p['metric'], pd.DataFrame({'step': [1]})) for p in plots] if not df.empty])
            
            greedy_spm = greedy_steps / greedy_time_info['total_minutes']
            cosine_spm = cosine_steps / cosine_time_info['total_minutes']
            
            f.write(f"- **Greedy LR**: {greedy_spm:.2f} steps/minute\n")
            f.write(f"- **Cosine LR**: {cosine_spm:.2f} steps/minute\n")
            
            if greedy_spm > cosine_spm:
                efficiency_diff = ((greedy_spm - cosine_spm) / cosine_spm) * 100
                f.write(f"- **Greedy LR is {efficiency_diff:.1f}% more efficient** in terms of training speed\n\n")
            else:
                efficiency_diff = ((cosine_spm - greedy_spm) / greedy_spm) * 100
                f.write(f"- **Cosine LR is {efficiency_diff:.1f}% more efficient** in terms of training speed\n\n")
        
        f.write("## Conclusion\n\n")
        
        if any('loss' in plot['metric'].lower() for plot in plots):
            # Find the training loss plots
            train_loss_plots = [p for p in plots if p['metric'] == 'train/loss']
            
            if train_loss_plots:
                train_plot = train_loss_plots[0]
                if train_plot['cosine_final'] < train_plot['greedy_final']:
                    perc_improvement = (train_plot['greedy_final'] - train_plot['cosine_final']) / train_plot['greedy_final'] * 100
                    f.write(f"At the final step, the **Cosine LR scheduler** shows better performance with {perc_improvement:.2f}% lower training loss.\n\n")
                else:
                    perc_improvement = (train_plot['cosine_final'] - train_plot['greedy_final']) / train_plot['cosine_final'] * 100
                    f.write(f"At the final step, the **Greedy LR scheduler** shows better performance with {perc_improvement:.2f}% lower training loss.\n\n")
        
        # Add analysis of loss progression if breakdown exists
        if loss_breakdown:
            f.write("### Loss Progression Analysis\n\n")
            early_better = loss_breakdown[0]['better']  # 10%
            mid_better = loss_breakdown[1]['better']    # 50%
            late_better = loss_breakdown[2]['better']   # 90%
            final_better = loss_breakdown[3]['better']  # 100%
            
            if early_better == mid_better == late_better == final_better:
                f.write(f"**{final_better} LR consistently outperforms** throughout the entire training process.\n\n")
            else:
                f.write("**Performance varies by training stage:**\n")
                f.write(f"- Early training (10%): {early_better} LR performs better\n")
                f.write(f"- Mid training (50%): {mid_better} LR performs better\n")
                f.write(f"- Late training (90%): {late_better} LR performs better\n")
                f.write(f"- Final (100%): {final_better} LR performs better\n\n")
        
        f.write("This comparison evaluates performance using equal training steps for fair comparison.\n")
        f.write("Results are based on the actual training data and may vary with different hyperparameters or longer training.\n")
    
    print(f"Report generated: {report_path}")
    return report_path

def main():
    """
    Main execution function for the comparison tool.
    
    Workflow:
    1. Parse command-line arguments
    2. Load data from TensorBoard logs
    3. Generate comparative plots
    4. Create detailed markdown report
    5. Output results location
    """
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    try:
        greedy_data = load_event_data(args.greedy_dir)
        cosine_data = load_event_data(args.cosine_dir)
    except Exception as e:
        print(f"Error loading data: {e}")
        print("\nIf the training hasn't completed yet, please run the training first:")
        print("  cd /home/sagemaker-user/balak4/transformers/examples/greedy-lr")
        print("  conda activate pytorch_p310_greedy_v2")
        print("  ./run_cosine_250steps.sh")
        return
    
    # Create plots and get loss breakdown
    plots, loss_breakdown = create_comparison_plots(greedy_data, cosine_data, args.output_dir)
    
    # Generate report with enhanced features
    report_path = generate_markdown_report(
        args.greedy_dir, 
        args.cosine_dir, 
        plots, 
        loss_breakdown, 
        greedy_data, 
        cosine_data, 
        args.output_dir
    )
    
    print("\nComparison complete!")
    print(f"Results saved to: {os.path.abspath(args.output_dir)}")
    print(f"View the report: {os.path.abspath(report_path)}")

if __name__ == "__main__":
    main()
