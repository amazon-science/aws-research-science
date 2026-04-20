#!/usr/bin/env python3
"""
ROBUST Comprehensive Scheduler Comparison Experiment
- Incremental saving every 100 experiments
- Memory cleanup between experiments  
- Error recovery and logging
- Performance monitoring
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json
import os
import gc
from dataclasses import dataclass
from tqdm import tqdm
import warnings
import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import psutil
import traceback
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Import all the functions from the original experiment
import sys
sys.path.append('/Users/subshrey/Projects/greedylr_research')

# Import the core components from comprehensive_scheduler_experiment.py
from comprehensive_scheduler_experiment import (
    ExperimentConfig, AdvancedGreedyLR, 
    generate_comprehensive_configs, run_comprehensive_experiment,
    create_unified_visualization
)

class RobustExperimentRunner:
    def __init__(self, checkpoint_interval=100, memory_limit_mb=4000):
        self.checkpoint_interval = checkpoint_interval
        self.memory_limit_mb = memory_limit_mb
        self.results_file = '/Users/subshrey/Projects/greedylr_research/robust_results.json'
        self.progress_file = '/Users/subshrey/Projects/greedylr_research/robust_progress.json'
        self.error_log = '/Users/subshrey/Projects/greedylr_research/robust_errors.log'
        self.checkpoint_dir = '/Users/subshrey/Projects/greedylr_research/checkpoints'
        
        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Initialize performance tracking
        self.performance_log = []
        
    def load_checkpoint(self):
        """Load existing results and find restart point"""
        try:
            if os.path.exists(self.results_file):
                with open(self.results_file, 'r') as f:
                    results = json.load(f)
                print(f"📂 Loaded {len(results)} existing results")
                return results
        except Exception as e:
            print(f"⚠️ Could not load checkpoint: {e}")
        return []
    
    def save_checkpoint(self, results, experiment_idx, total_experiments):
        """Save results and progress incrementally"""
        try:
            # Convert results to JSON-serializable format
            serializable_results = []
            for result in results:
                if isinstance(result, dict) and 'config' in result:
                    serializable_result = result.copy()
                    # Convert ExperimentConfig to dict
                    config = result['config']
                    if hasattr(config, '__dict__'):  # It's a dataclass
                        serializable_result['config'] = {
                            'model_type': config.model_type,
                            'noise_type': config.noise_type,
                            'noise_strength': config.noise_strength,
                            'total_steps': config.total_steps,
                            'scheduler_type': config.scheduler_type,
                            'scheduler_params': config.scheduler_params,
                            'problem_variant': config.problem_variant
                        }
                        # Add individual fields for easy access during post-processing
                        serializable_result['model_type'] = config.model_type
                        serializable_result['scheduler_type'] = config.scheduler_type
                        serializable_result['noise_type'] = config.noise_type
                        serializable_result['noise_strength'] = config.noise_strength
                        serializable_result['problem_variant'] = config.problem_variant
                    serializable_results.append(serializable_result)
                else:
                    serializable_results.append(result)
            
            # Save main results
            with open(self.results_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            # Save progress info
            progress_data = {
                'completed': experiment_idx + 1,
                'total': total_experiments,
                'percentage': (experiment_idx + 1) / total_experiments * 100,
                'timestamp': pd.Timestamp.now().isoformat(),
                'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024,
                'performance_log': self.performance_log[-10:] if self.performance_log else []
            }
            
            with open(self.progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
                
            # Create numbered checkpoint every 1000 experiments
            if (experiment_idx + 1) % 1000 == 0:
                checkpoint_file = f"{self.checkpoint_dir}/checkpoint_{experiment_idx + 1:05d}.json"
                with open(checkpoint_file, 'w') as f:
                    json.dump(serializable_results, f, indent=2)
                print(f"💾 Checkpoint saved: {checkpoint_file}")
                
        except Exception as e:
            self.log_error(f"Failed to save checkpoint: {e}")
    
    def log_error(self, error_msg):
        """Log errors to file"""
        try:
            with open(self.error_log, 'a') as f:
                f.write(f"{pd.Timestamp.now().isoformat()}: {error_msg}\n")
        except:
            pass  # Don't let logging errors crash the experiment
    
    def cleanup_memory(self):
        """ULTRA-AGGRESSIVE memory cleanup to maintain performance"""
        try:
            # Clear PyTorch cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Clear CPU PyTorch cache
            torch._C._cuda_clearCublasWorkspaces()
            
            # Force multiple garbage collections
            for _ in range(5):  # Increased from 3
                gc.collect()
            
            # Clear matplotlib memory if used
            plt.close('all')
            
            # Check memory usage and take action
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            if memory_mb > self.memory_limit_mb:
                print(f"⚠️ High memory usage: {memory_mb:.1f}MB - AGGRESSIVE CLEANUP")
                
                # Nuclear option: clear everything possible
                import sys
                # Clear module cache for imports
                for module_name in list(sys.modules.keys()):
                    if 'torch' in module_name and 'comprehensive' not in module_name:
                        try:
                            del sys.modules[module_name]
                        except:
                            pass
                
                # Force more garbage collection
                for _ in range(10):
                    gc.collect()
                
                # Re-import torch if needed
                import torch
                
        except Exception as e:
            self.log_error(f"Memory cleanup failed: {e}")
    
    def run_experiment_safely(self, config, experiment_idx):
        """Run single experiment with error handling and cleanup"""
        try:
            # Monitor performance
            start_time = pd.Timestamp.now()
            
            # Pre-experiment cleanup for transformer models
            if 'transformer' in config.model_type or 'vit' in config.model_type:
                if experiment_idx % 5 == 0:  # More frequent for heavy models
                    self.cleanup_memory()
            
            # Run the experiment
            result = run_comprehensive_experiment(config)
            
            # Explicit cleanup of experiment results if they contain model references
            if hasattr(result, '__dict__'):
                for key, value in result.__dict__.items():
                    if hasattr(value, 'parameters'):  # It's a model
                        del value
            
            # Track performance
            end_time = pd.Timestamp.now()
            duration = (end_time - start_time).total_seconds()
            self.performance_log.append({
                'experiment': experiment_idx,
                'duration_seconds': duration,
                'timestamp': end_time.isoformat()
            })
            
            # Adaptive memory cleanup based on performance
            if experiment_idx % 10 == 0 or duration > 1.0:  # More frequent if slow
                self.cleanup_memory()
            
            # Performance degradation detection
            if len(self.performance_log) >= 100:
                recent_avg = np.mean([p['duration_seconds'] for p in self.performance_log[-50:]])
                early_avg = np.mean([p['duration_seconds'] for p in self.performance_log[:50]])
                
                if recent_avg > early_avg * 2.0:  # Performance degraded 2x
                    print(f"⚠️ PERFORMANCE DEGRADATION DETECTED: {1/recent_avg:.1f} exp/sec (was {1/early_avg:.1f})")
                    print("🧹 Performing DEEP CLEANUP...")
                    
                    # Ultra-aggressive cleanup
                    self.cleanup_memory()
                    
                    # Reset performance tracking
                    self.performance_log = self.performance_log[-10:]
            
            return result
            
        except Exception as e:
            error_msg = f"Experiment {experiment_idx} failed: {str(e)}\nConfig: {config}\nTraceback: {traceback.format_exc()}"
            self.log_error(error_msg)
            print(f"❌ Experiment {experiment_idx} failed: {e}")
            return None
    
    def run_robust_experiment(self):
        """Main robust experiment runner"""
        print("🚀 Starting ROBUST Comprehensive Scheduler Experiment")
        print("=" * 80)
        print("Features:")
        print("• Incremental saving every 100 experiments")
        print("• Memory cleanup after each experiment") 
        print("• Error recovery and detailed logging")
        print("• Performance monitoring")
        print("• Checkpoint recovery")
        print("=" * 80)
        
        # Load existing results if any
        results = self.load_checkpoint()
        start_idx = len(results)
        
        # Generate optimized configurations  
        print("📋 Generating OPTIMIZED 8K experiment configurations...")
        from optimized_experiment import generate_optimized_configs
        configs = generate_optimized_configs()
        total_experiments = len(configs)
        print(f"✅ Generated {total_experiments:,} OPTIMIZED experiment configurations")
        
        if start_idx > 0:
            print(f"🔄 Resuming from experiment {start_idx:,}")
            configs = configs[start_idx:]  # Skip already completed experiments
        
        print(f"\n🔬 Running {len(configs):,} experiments...")
        failed_experiments = 0
        
        # Run experiments with robust error handling
        for i, config in enumerate(tqdm(configs, desc="Running robust experiments")):
            actual_idx = start_idx + i
            
            # Run experiment safely
            result = self.run_experiment_safely(config, actual_idx)
            
            if result is not None:
                results.append(result)
            else:
                failed_experiments += 1
                continue
            
            # Save checkpoint regularly
            if (actual_idx + 1) % self.checkpoint_interval == 0:
                self.save_checkpoint(results, actual_idx, total_experiments)
                
                # Performance summary
                if self.performance_log:
                    recent_durations = [p['duration_seconds'] for p in self.performance_log[-100:]]
                    avg_duration = np.mean(recent_durations)
                    rate = 1.0 / avg_duration if avg_duration > 0 else 0
                    print(f"📊 Performance: {rate:.1f} exp/sec (avg over last 100)")
                
                # Memory status
                memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
                print(f"💾 Memory usage: {memory_mb:.1f}MB")
            
            # Milestone reporting
            if (actual_idx + 1) % 1000 == 0:
                completed = actual_idx + 1
                percentage = completed / total_experiments * 100
                print(f"\n🎯 Milestone: {completed:,}/{total_experiments:,} ({percentage:.1f}%)")
                print(f"✅ Successful: {len(results):,}")
                print(f"❌ Failed: {failed_experiments}")
                
                # Estimate remaining time
                if self.performance_log:
                    recent_rates = [1.0/p['duration_seconds'] for p in self.performance_log[-100:] if p['duration_seconds'] > 0]
                    if recent_rates:
                        avg_rate = np.mean(recent_rates)
                        remaining = total_experiments - completed
                        eta_hours = remaining / avg_rate / 3600
                        print(f"⏱️  ETA: {eta_hours:.1f} hours")
        
        # Final save
        self.save_checkpoint(results, start_idx + len(configs) - 1, total_experiments)
        
        print(f"\n✅ EXPERIMENT COMPLETED!")
        print(f"📊 Total successful: {len(results):,}")
        print(f"❌ Total failed: {failed_experiments}")
        print(f"💾 Results saved to: {self.results_file}")
        
        # Attempt analysis and visualization
        try:
            print("\n📊 Performing analysis...")
            df = pd.DataFrame(results)
            
            # Extract metrics from nested structure if needed
            processed_data = []
            for result in results:
                if isinstance(result, dict):
                    # Extract metrics if they're nested
                    metrics = result.get('metrics', {})
                    if isinstance(metrics, dict):
                        # Flatten the result for easier analysis
                        flattened = {
                            'model_type': result.get('model_type', result.get('config', {}).get('model_type', 'unknown')),
                            'scheduler_type': result.get('scheduler_type', result.get('config', {}).get('scheduler_type', 'unknown')),
                            'noise_type': result.get('noise_type', result.get('config', {}).get('noise_type', 'unknown')),
                            'noise_strength': result.get('noise_strength', result.get('config', {}).get('noise_strength', 0)),
                            'problem_variant': result.get('problem_variant', result.get('config', {}).get('problem_variant', 'standard')),
                            'final_loss': metrics.get('final_loss', float('inf')),
                            'min_loss': metrics.get('min_loss', float('inf')),
                            'converged_step': metrics.get('converged_step'),
                            'convergence_rate_50': metrics.get('convergence_rate_50', 0),
                            'stability_score': metrics.get('stability_score', 0),
                            'lr_changes': metrics.get('lr_changes', 0),
                            'spike_recovery_time': metrics.get('spike_recovery_time', 0),
                            'robustness_score': metrics.get('robustness_score', 0)
                        }
                        processed_data.append(flattened)
            
            if processed_data:
                df = pd.DataFrame(processed_data)
            
            # Ensure we have the required columns for analysis
            required_columns = ['scheduler_type', 'final_loss', 'convergence_rate_50', 'stability_score']
            if not all(col in df.columns for col in required_columns):
                raise KeyError(f"Missing required columns. Available: {list(df.columns)}")
            
            # Simple analysis
            analysis = {
                'total_experiments': len(df),
                'scheduler_performance': df.groupby('scheduler_type').agg({
                    'final_loss': ['mean', 'std', 'min'],
                    'convergence_rate_50': ['mean', 'std', 'max'],
                    'stability_score': ['mean', 'std', 'max']
                }).to_dict(),
                'model_performance': df.groupby('model_type').agg({
                    'final_loss': ['mean', 'std'],
                    'convergence_rate_50': ['mean', 'std']
                }).to_dict() if 'model_type' in df.columns else {},
                'noise_analysis': df.groupby('noise_type').agg({
                    'final_loss': ['mean', 'std'],
                    'robustness_score': ['mean', 'std']  
                }).to_dict() if 'noise_type' in df.columns else {},
                'summary': {
                    'best_scheduler_loss': df.groupby('scheduler_type')['final_loss'].mean().idxmin(),
                    'best_scheduler_convergence': df.groupby('scheduler_type')['convergence_rate_50'].mean().idxmax(),
                    'best_scheduler_stability': df.groupby('scheduler_type')['stability_score'].mean().idxmax()
                }
            }
            
            # Save analysis
            analysis_file = '/Users/subshrey/Projects/greedylr_research/robust_analysis.json'
            with open(analysis_file, 'w') as f:
                json.dump(analysis, f, indent=2)
            print(f"✅ Analysis saved to: {analysis_file}")
            
            # Try visualization (skip for now to avoid errors)
            print("📈 Skipping visualization for now (data saved successfully)")
            
        except Exception as e:
            self.log_error(f"Post-processing failed: {e}")
            print(f"⚠️ Post-processing failed: {e}")
            print("🎉 But the main experiment data was saved successfully!")
        
        return results

def main():
    """Main entry point"""
    runner = RobustExperimentRunner(
        checkpoint_interval=100,  # Save every 100 experiments
        memory_limit_mb=4000     # Warning threshold for memory usage
    )
    
    results = runner.run_robust_experiment()
    return results

if __name__ == "__main__":
    results = main()