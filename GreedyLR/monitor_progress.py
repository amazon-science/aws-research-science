#!/usr/bin/env python3
"""
Real-time progress monitor for the comprehensive scheduler experiment
"""

import json
import time
import os
from datetime import datetime

def monitor_progress():
    progress_file = '/Users/subshrey/Projects/greedylr_research/experiment_progress.json'
    
    print("🔍 Monitoring Comprehensive Scheduler Experiment Progress")
    print("=" * 60)
    print("Press Ctrl+C to stop monitoring\n")
    
    last_completed = 0
    last_time = None
    
    try:
        while True:
            if os.path.exists(progress_file):
                try:
                    with open(progress_file, 'r') as f:
                        progress = json.load(f)
                    
                    completed = progress['completed']
                    total = progress['total']
                    percentage = progress['percentage']
                    failed = progress['failed']
                    rate = progress.get('rate_per_second', 0)
                    eta_hours = progress.get('eta_hours', 0)
                    timestamp = progress['timestamp']
                    latest_config = progress['latest_config']
                    
                    # Progress bar with better visualization
                    bar_length = 50
                    filled_length = int(bar_length * percentage / 100)
                    bar = '█' * filled_length + '░' * (bar_length - filled_length)
                    
                    # Clear screen and show updated progress
                    os.system('clear' if os.name == 'posix' else 'cls')
                    
                    print("🔍 Comprehensive Scheduler Experiment Monitor")
                    print("=" * 60)
                    print(f"📊 Progress: [{bar}] {percentage:.2f}%")
                    print(f"📈 Completed: {completed:,} / {total:,} experiments")
                    print(f"⚡ Rate: {rate:.2f} experiments/second")
                    print(f"⏱️  ETA: {eta_hours:.1f} hours")
                    print(f"❌ Failed: {failed} experiments")
                    print()
                    print("🔧 Current Experiment:")
                    print(f"   • Model: {latest_config['model_type']}")
                    print(f"   • Scheduler: {latest_config['scheduler_type']}")  
                    print(f"   • Noise: {latest_config['noise_type']} (strength: {latest_config.get('noise_strength', 'N/A')})")
                    print(f"   • Variant: {latest_config.get('problem_variant', 'N/A')}")
                    print()
                    
                    # Milestones
                    milestones = [1000, 5000, 10000, 15000, 20000, 25000]
                    print("🎯 Milestones:")
                    for milestone in milestones:
                        if completed >= milestone:
                            print(f"   ✅ {milestone:,} experiments")
                        else:
                            remaining_to_milestone = milestone - completed
                            eta_to_milestone = remaining_to_milestone / rate if rate > 0 else float('inf')
                            print(f"   ⏳ {milestone:,} experiments (ETA: {eta_to_milestone/3600:.1f}h)")
                            break
                    
                    print()
                    print(f"⏰ Last update: {timestamp}")
                    print(f"🔄 Monitoring... (updates every 15 seconds)")
                    
                    # Check if milestone reached
                    if progress.get('milestone_reached', False):
                        print("\n🎉 MILESTONE REACHED! 🎉")
                    
                except json.JSONDecodeError:
                    print("⚠️  Progress file corrupted, waiting...")
                except KeyError as e:
                    print(f"⚠️  Missing key in progress file: {e}")
            else:
                print("⏳ Waiting for experiment to start...")
                print("   Progress file not found yet...")
            
            time.sleep(15)  # Update every 15 seconds
            
    except KeyboardInterrupt:
        print("\n\n👋 Stopped monitoring")
        print("Experiment continues running in background")

if __name__ == "__main__":
    monitor_progress()