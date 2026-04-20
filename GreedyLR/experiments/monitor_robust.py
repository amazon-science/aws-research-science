#!/usr/bin/env python3
"""
Monitor the robust comprehensive experiment
"""

import json
import time
import os
from datetime import datetime

def monitor_robust_experiment():
    progress_file = '/Users/subshrey/Projects/greedylr_research/robust_progress.json'
    results_file = '/Users/subshrey/Projects/greedylr_research/robust_results.json'
    
    print("🔍 Monitoring ROBUST Comprehensive Scheduler Experiment")
    print("=" * 70)
    print("Press Ctrl+C to stop monitoring\n")
    
    try:
        while True:
            # Clear screen
            os.system('clear' if os.name == 'posix' else 'cls')
            
            print("🔍 ROBUST Comprehensive Scheduler Experiment Monitor")
            print("=" * 70)
            
            if os.path.exists(progress_file):
                try:
                    with open(progress_file, 'r') as f:
                        progress = json.load(f)
                    
                    completed = progress['completed']
                    total = progress['total']
                    percentage = progress['percentage']
                    timestamp = progress['timestamp']
                    memory_mb = progress.get('memory_usage_mb', 0)
                    perf_log = progress.get('performance_log', [])
                    
                    # Progress bar
                    bar_length = 50
                    filled_length = int(bar_length * percentage / 100)
                    bar = '█' * filled_length + '░' * (bar_length - filled_length)
                    
                    print(f"📊 Progress: [{bar}] {percentage:.2f}%")
                    print(f"📈 Completed: {completed:,} / {total:,} experiments")
                    print(f"💾 Memory Usage: {memory_mb:.1f} MB")
                    print()
                    
                    # Performance stats
                    if perf_log:
                        recent_durations = [p['duration_seconds'] for p in perf_log if 'duration_seconds' in p]
                        if recent_durations:
                            avg_duration = sum(recent_durations) / len(recent_durations)
                            rate = 1.0 / avg_duration if avg_duration > 0 else 0
                            remaining = total - completed
                            eta_hours = remaining / rate / 3600 if rate > 0 else 0
                            
                            print(f"⚡ Current Rate: {rate:.1f} experiments/second")
                            print(f"⏱️  ETA: {eta_hours:.1f} hours")
                        print()
                    
                    # Milestones
                    milestones = [1000, 5000, 10000, 15000, 20000, 25000, 29866]
                    print("🎯 Milestones:")
                    for milestone in milestones:
                        if completed >= milestone:
                            print(f"   ✅ {milestone:,} experiments")
                        else:
                            remaining_to_milestone = milestone - completed
                            if perf_log and recent_durations:
                                avg_duration = sum(recent_durations) / len(recent_durations)
                                rate = 1.0 / avg_duration if avg_duration > 0 else 0
                                eta_to_milestone = remaining_to_milestone / rate / 3600 if rate > 0 else float('inf')
                                print(f"   ⏳ {milestone:,} experiments (ETA: {eta_to_milestone:.1f}h)")
                            else:
                                print(f"   ⏳ {milestone:,} experiments")
                            break
                                
                    print()
                    
                    # Check results file size
                    if os.path.exists(results_file):
                        size_mb = os.path.getsize(results_file) / 1024 / 1024
                        print(f"💾 Results file size: {size_mb:.1f} MB")
                    
                    print(f"⏰ Last update: {timestamp}")
                    print(f"🔄 Monitoring... (updates every 20 seconds)")
                    
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"⚠️  Error reading progress file: {e}")
            else:
                print("⏳ Waiting for experiment to start...")
                print("   Progress file not found yet...")
            
            time.sleep(20)  # Update every 20 seconds
            
    except KeyboardInterrupt:
        print("\n\n👋 Stopped monitoring")
        print("Experiment continues running in background")

if __name__ == "__main__":
    monitor_robust_experiment()