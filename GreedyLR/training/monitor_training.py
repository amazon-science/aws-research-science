#!/usr/bin/env python3
"""
LLaMA Training Monitor

Real-time monitoring of training progress by reading TensorBoard logs directly.
Provides a terminal-based dashboard showing key metrics and trends without
requiring a TensorBoard server.

Usage:
  python monitor_training.py [--log-dir DIR] [--refresh SECONDS]
"""

import os
import sys
import time
import glob
from datetime import datetime, timedelta
from collections import defaultdict, deque
import argparse

try:
    from tensorboard.backend.event_processing import event_accumulator
    import numpy as np
except ImportError:
    print("Error: Required packages not found. Please install:")
    print("pip install tensorboard numpy")
    sys.exit(1)

class TrainingMonitor:
    """
    Real-time monitor for LLaMA training that reads TensorBoard logs and displays
    metrics, progress, and trends in a terminal-based dashboard.
    """
    
    def __init__(self, base_log_dir, refresh_interval=30):
        """
        Initialize the training monitor.
        
        Args:
            base_log_dir (str): Directory containing training logs
            refresh_interval (int): Time in seconds between dashboard updates
        """
        self.base_log_dir = base_log_dir
        self.refresh_interval = refresh_interval
        self.start_time = time.time()
        self.last_update = 0
        
        # Tracking variables
        self.current_phase = None
        self.phase_start_time = None
        self.metrics_history = defaultdict(lambda: deque(maxlen=10))
        
        print(f"🔍 Monitoring training logs in: {base_log_dir}")
        print(f"⏱️  Refresh interval: {refresh_interval} seconds")
        print("=" * 70)
    
    def find_tensorboard_logs(self):
        """
        Find all TensorBoard log directories for different schedulers.
        
        Returns:
            dict: Dictionary mapping scheduler names to their log directories
        """
        log_dirs = {}
        
        # Look for greedy and cosine scheduler logs
        for scheduler in ['greedy', 'cosine']:
            pattern = os.path.join(self.base_log_dir, scheduler, '*', 'tensorboard')
            matches = glob.glob(pattern)
            if matches:
                # Use the most recent directory
                matches.sort(key=os.path.getmtime, reverse=True)
                log_dirs[scheduler] = matches[0]
        
        return log_dirs
    
    def load_metrics_from_logs(self, log_dir):
        """
        Load metrics data from TensorBoard event files.
        
        Args:
            log_dir (str): Directory containing TensorBoard event files
            
        Returns:
            dict: Dictionary containing extracted metrics data
        """
        if not os.path.exists(log_dir):
            return {}
        
        # Find event files
        event_files = glob.glob(os.path.join(log_dir, 'events.out.tfevents.*'))
        if not event_files:
            return {}
        
        # Use the most recent event file
        event_file = max(event_files, key=os.path.getmtime)
        
        try:
            ea = event_accumulator.EventAccumulator(
                event_file,
                size_guidance={
                    event_accumulator.SCALARS: 0,  # Load all scalars
                }
            )
            ea.Reload()
            
            # Extract metrics
            metrics = {}
            available_tags = ea.Tags()['scalars']
            
            for tag in available_tags:
                try:
                    events = ea.Scalars(tag)
                    if events:
                        steps = [e.step for e in events]
                        values = [e.value for e in events]
                        wall_times = [e.wall_time for e in events]
                        
                        metrics[tag] = {
                            'steps': steps,
                            'values': values,
                            'wall_times': wall_times,
                            'latest_step': steps[-1] if steps else 0,
                            'latest_value': values[-1] if values else 0,
                            'latest_time': wall_times[-1] if wall_times else 0
                        }
                except Exception as e:
                    print(f"⚠️  Warning: Could not load metric {tag}: {e}")
                    continue
            
            return metrics
        
        except Exception as e:
            print(f"❌ Error loading logs from {log_dir}: {e}")
            return {}
    
    def detect_current_phase(self, log_dirs, metrics_data):
        """
        Determine which training phase (greedy or cosine) is currently active.
        
        Args:
            log_dirs (dict): Mapping of scheduler names to log directories
            metrics_data (dict): Metrics data loaded from TensorBoard logs
            
        Returns:
            str or None: Name of the active phase, or None if no active phase detected
        """
        active_phases = []
        
        for phase, log_dir in log_dirs.items():
            if phase in metrics_data and metrics_data[phase]:
                # Check if this phase has recent activity (within last 5 minutes)
                latest_times = []
                for metric_data in metrics_data[phase].values():
                    if metric_data['wall_times']:
                        latest_times.append(metric_data['latest_time'])
                
                if latest_times:
                    latest_time = max(latest_times)
                    if time.time() - latest_time < 300:  # 5 minutes
                        active_phases.append((phase, latest_time))
        
        if active_phases:
            # Return the phase with the most recent activity
            return max(active_phases, key=lambda x: x[1])[0]
        
        # Fallback: return phase with any data
        for phase in ['greedy', 'cosine']:
            if phase in metrics_data and metrics_data[phase]:
                return phase
        
        return None
    
    def format_time_elapsed(self, seconds):
        """
        Format elapsed time in a human-readable format.
        
        Args:
            seconds (float): Time in seconds
            
        Returns:
            str: Formatted time string (e.g., "5h 30m", "45m 10s")
        """
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            return f"{int(seconds/60)}m {int(seconds%60)}s"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}h {minutes}m"
    
    def get_trend_indicator(self, values):
        """
        Calculate the trend of a metric over recent values.
        
        Args:
            values (list): List of metric values
            
        Returns:
            str: Trend indicator with emoji and description
        """
        if len(values) < 2:
            return "➡️"
        
        recent_avg = np.mean(values[-3:]) if len(values) >= 3 else values[-1]
        older_avg = np.mean(values[-6:-3]) if len(values) >= 6 else values[0]
        
        if recent_avg < older_avg * 0.99:  # Significant improvement
            return "⬇️ (improving)"
        elif recent_avg > older_avg * 1.01:  # Significant degradation
            return "⬆️ (degrading)"
        else:
            return "➡️ (stable)"
    
    def display_progress_bar(self, current, total, width=40):
        """
        Create a text-based progress bar.
        
        Args:
            current (int): Current step
            total (int): Total steps
            width (int): Width of the progress bar in characters
            
        Returns:
            str: Text progress bar (e.g., "[████████░░░░░░]")
        """
        if total == 0:
            return "[" + "?" * width + "]"
        
        progress = min(current / total, 1.0)
        filled = int(width * progress)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}]"
    
    def display_metrics(self, phase_metrics):
        """Display current metrics for a phase."""
        key_metrics = {
            'train/loss': 'Training Loss',
            'eval/loss': 'Eval Loss', 
            'train/learning_rate': 'Learning Rate',
            'train/grad_norm': 'Grad Norm'
        }
        
        print("\n📊 Current Metrics:")
        for metric_key, display_name in key_metrics.items():
            if metric_key in phase_metrics:
                data = phase_metrics[metric_key]
                latest_value = data['latest_value']
                trend = self.get_trend_indicator(data['values'])
                
                if 'learning_rate' in metric_key:
                    print(f"   {display_name}: {latest_value:.2e}")
                else:
                    print(f"   {display_name}: {latest_value:.4f} {trend}")
    
    def display_recent_trend(self, phase_metrics):
        """Display recent loss trend."""
        if 'train/loss' in phase_metrics:
            loss_data = phase_metrics['train/loss']
            steps = loss_data['steps']
            values = loss_data['values']
            
            if len(steps) >= 3:
                print("\n📈 Recent Loss Trend (last 5 steps):")
                recent_count = min(5, len(steps))
                trend_line = " → ".join([
                    f"{steps[-recent_count+i]}: {values[-recent_count+i]:.4f}"
                    for i in range(recent_count)
                ])
                print(f"   {trend_line}")
    
    def estimate_remaining_time(self, current_step, total_steps, elapsed_time):
        """Estimate remaining training time."""
        if current_step == 0:
            return "Calculating..."
        
        # Avoid division by zero for very small elapsed times
        if elapsed_time < 1:  # Less than 1 second
            return "Calculating..."
        
        # Need at least 2 steps and reasonable elapsed time for accurate estimation
        if current_step < 2 or elapsed_time < 30:
            return "Calculating..."
        
        try:
            steps_per_second = current_step / elapsed_time
            remaining_steps = total_steps - current_step
            
            # Avoid division by zero
            if steps_per_second <= 0:
                return "Calculating..."
                
            remaining_seconds = remaining_steps / steps_per_second
            return self.format_time_elapsed(remaining_seconds)
            
        except (ZeroDivisionError, ValueError):
            return "Calculating..."
    
    def clear_screen(self):
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def display_status(self, log_dirs, metrics_data):
        """Display the current training status."""
        self.clear_screen()
        
        # Header
        print("╔" + "=" * 68 + "╗")
        print("║" + " " * 20 + "🦙 LLaMA Training Monitor" + " " * 20 + "║")
        print("╠" + "=" * 68 + "╣")
        
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"║ Last Updated: {current_time}" + " " * (68 - 14 - len(current_time)) + "║")
        
        # Determine current phase
        active_phase = self.detect_current_phase(log_dirs, metrics_data)
        
        if not active_phase:
            print("║" + " " * 25 + "⏳ Waiting for training..." + " " * 16 + "║")
            print("╚" + "=" * 68 + "╝")
            return
        
        phase_metrics = metrics_data[active_phase]
        phase_display = "Greedy LR" if active_phase == "greedy" else "Cosine LR"
        
        print(f"║ Phase: {phase_display} Training" + " " * (68 - 9 - len(phase_display) - 9) + "║")
        
        # Progress information
        if 'train/loss' in phase_metrics:
            loss_data = phase_metrics['train/loss']
            current_step = loss_data['latest_step']
            total_steps = 2000  # As configured in your training
            
            progress_pct = (current_step / total_steps) * 100
            progress_bar = self.display_progress_bar(current_step, total_steps, 30)
            
            print(f"║ Progress: {progress_bar} {current_step}/{total_steps} ({progress_pct:.1f}%)" + 
                  " " * (68 - 11 - len(progress_bar) - len(f"{current_step}/{total_steps}") - len(f"({progress_pct:.1f}%)")) + "║")
            
            # Time estimation
            if loss_data['wall_times']:
                start_time = loss_data['wall_times'][0]
                current_time = loss_data['latest_time']
                elapsed = current_time - start_time
                
                elapsed_str = self.format_time_elapsed(elapsed)
                remaining_str = self.estimate_remaining_time(current_step, total_steps, elapsed)
                
                time_info = f"Time: {elapsed_str} elapsed | Est. remaining: {remaining_str}"
                print(f"║ {time_info}" + " " * (68 - 1 - len(time_info)) + "║")
        
        print("║" + " " * 68 + "║")
        
        # Display metrics in a formatted way
        self.display_metrics(phase_metrics)
        self.display_recent_trend(phase_metrics)
        
        print("║" + " " * 68 + "║")
        print("║ [Press Ctrl+C to exit]" + " " * 45 + "║")
        print("╚" + "=" * 68 + "╝")
    
    def run(self):
        """
        Main monitoring loop that updates the dashboard at the specified interval.
        Runs indefinitely until interrupted by the user.
        """
        print("🚀 Starting training monitor...")
        
        try:
            while True:
                # Find available log directories
                log_dirs = self.find_tensorboard_logs()
                
                if not log_dirs:
                    print(f"⏳ No training logs found in {self.base_log_dir}")
                    print("   Waiting for training to start...")
                    time.sleep(self.refresh_interval)
                    continue
                
                # Load metrics from all available log directories
                metrics_data = {}
                for phase, log_dir in log_dirs.items():
                    metrics_data[phase] = self.load_metrics_from_logs(log_dir)
                
                # Display current status
                self.display_status(log_dirs, metrics_data)
                
                # Wait before next update
                time.sleep(self.refresh_interval)
                
        except KeyboardInterrupt:
            print("\n\n👋 Monitoring stopped by user")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Monitor LLaMA training progress')
    parser.add_argument(
        '--log-dir',
        type=str,
        default='./logs/redpajama/meta-llama/Llama-3.2-1B/run_2',
        help='Base directory containing training logs'
    )
    parser.add_argument(
        '--refresh',
        type=int,
        default=30,
        help='Refresh interval in seconds (default: 30)'
    )
    
    args = parser.parse_args()
    
    # Validate log directory
    if not os.path.exists(args.log_dir):
        print(f"❌ Error: Log directory not found: {args.log_dir}")
        print("   Please check the path and try again.")
        sys.exit(1)
    
    # Create and run monitor
    monitor = TrainingMonitor(args.log_dir, args.refresh)
    monitor.run()

if __name__ == "__main__":
    main()
