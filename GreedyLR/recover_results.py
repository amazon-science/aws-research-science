#!/usr/bin/env python3
"""
Attempt to recover the experimental results by re-running post-processing
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path

def recover_results():
    """Try to recover results from all available sources"""
    
    print("🔍 Attempting to recover experimental results...")
    
    # Check if there are any partially saved results
    result_files = [
        'robust_results.json',
        'experiment_progress.json', 
        'scheduler_comparison_analysis.json'
    ]
    
    recovered_data = []
    
    for file_path in result_files:
        if Path(file_path).exists():
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    print(f"✅ Loaded {file_path}: {len(data) if isinstance(data, list) else 'dict'} entries")
                    if isinstance(data, list):
                        recovered_data.extend(data)
                    elif isinstance(data, dict) and 'experiments' in data:
                        recovered_data.extend(data['experiments'])
            except Exception as e:
                print(f"❌ Failed to load {file_path}: {e}")
    
    # Try to parse experiment output log for any embedded results
    print("\n🔍 Scanning experiment output log for embedded results...")
    
    try:
        with open('experiment_output.log', 'r') as f:
            log_content = f.read()
            
        # Look for any JSON-like structures or results
        import re
        
        # Look for any success/failure patterns
        success_pattern = r'✅ Experiment (\d+)'
        success_matches = re.findall(success_pattern, log_content)
        
        failure_pattern = r'❌ Experiment (\d+) failed'
        failure_matches = re.findall(failure_pattern, log_content)
        
        print(f"📊 Found {len(success_matches)} success indicators")
        print(f"📊 Found {len(failure_matches)} failure indicators")
        
        # Look for model types mentioned
        model_patterns = [
            r'quadratic', r'rosenbrock', r'rastrigin', r'ackley',
            r'neural_simple', r'neural_resnet', r'neural_attention', 
            r'neural_vit', r'neural_conv', r'neural_deep_transformer',
            r'neural_wide_transformer', r'neural_multi_head'
        ]
        
        model_counts = {}
        for pattern in model_patterns:
            matches = re.findall(pattern, log_content, re.IGNORECASE)
            if matches:
                model_counts[pattern] = len(matches)
        
        print(f"📊 Model types found in log: {model_counts}")
        
    except Exception as e:
        print(f"❌ Failed to parse log: {e}")
    
    # Check what we actually have
    print(f"\n📊 Total recovered entries: {len(recovered_data)}")
    
    if recovered_data:
        # Try to analyze what we have
        df = pd.DataFrame(recovered_data)
        print(f"📊 DataFrame shape: {df.shape}")
        print(f"📊 Columns: {list(df.columns)}")
        
        if 'model_type' in df.columns:
            print(f"📊 Model types: {df['model_type'].unique()}")
        if 'scheduler_type' in df.columns:
            print(f"📊 Scheduler types: {df['scheduler_type'].unique()}")
        if 'noise_type' in df.columns:
            print(f"📊 Noise types: {df['noise_type'].unique()}")
            
        return df
    else:
        print("❌ No recoverable data found")
        return None

def attempt_rerun_analysis():
    """Try to rerun just the analysis part of the experiment"""
    print("\n🔄 Attempting to rerun analysis from scratch...")
    
    # Import the main experiment functions
    try:
        import sys
        sys.path.append('.')
        from comprehensive_scheduler_experiment import generate_comprehensive_configs, run_comprehensive_experiment
        
        print("✅ Successfully imported experiment functions")
        
        # Generate a small subset of configs to test
        print("🧪 Testing with a small subset of experiments...")
        
        # Get first 50 configs
        all_configs = generate_comprehensive_configs()
        test_configs = all_configs[:50]  # Just test 50 experiments
        
        print(f"🧪 Running {len(test_configs)} test experiments...")
        
        results = []
        for i, config in enumerate(test_configs):
            try:
                result = run_comprehensive_experiment(config)
                if result:
                    results.append(result)
                    print(f"✅ Test experiment {i+1}/{len(test_configs)} completed")
                else:
                    print(f"❌ Test experiment {i+1}/{len(test_configs)} failed")
            except Exception as e:
                print(f"❌ Test experiment {i+1}/{len(test_configs)} error: {e}")
            
            if i >= 9:  # Just test first 10
                break
        
        if results:
            print(f"✅ Successfully ran {len(results)} test experiments")
            
            # Save test results
            with open('test_recovery_results.json', 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            return results
        else:
            print("❌ No test results obtained")
            return None
            
    except Exception as e:
        print(f"❌ Failed to rerun analysis: {e}")
        return None

if __name__ == "__main__":
    # Try recovery first
    recovered_df = recover_results()
    
    if recovered_df is None or len(recovered_df) < 100:
        print("\n⚠️ Insufficient recovered data, attempting test rerun...")
        test_results = attempt_rerun_analysis()
        if test_results:
            print("✅ Test rerun successful - could rerun full experiment if needed")
        else:
            print("❌ Complete data loss - would need to rerun full experiment")
    else:
        print("✅ Sufficient data recovered for analysis")