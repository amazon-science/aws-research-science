#!/usr/bin/env python3
"""
Focused experiment to recover key results across all 12 architectures
"""
import json
from comprehensive_scheduler_experiment import generate_comprehensive_configs, run_comprehensive_experiment
import pandas as pd
import re

def run_focused_experiment():
    """Run a smaller focused experiment to get results for all architectures"""
    
    print("🎯 Running focused recovery experiment...")
    print("📊 Target: ~50 experiments per architecture across key conditions")
    
    # Generate all configs
    all_configs = generate_comprehensive_configs()
    print(f"📊 Total available configs: {len(all_configs)}")
    
    # Group by model type to ensure we get all architectures
    model_type_configs = {}
    for config in all_configs:
        # Parse model type from config string
        match = re.search(r"model_type='([^']+)'", str(config))
        if match:
            model_type = match.group(1)
            if model_type not in model_type_configs:
                model_type_configs[model_type] = []
            model_type_configs[model_type].append(config)
    
    print(f"📊 Found {len(model_type_configs)} model types:")
    for mt, configs in model_type_configs.items():
        print(f"  {mt}: {len(configs)} configs")
    
    # Select focused subset - ~50 per model type
    focused_configs = []
    for model_type, configs in model_type_configs.items():
        # Take every nth config to get good coverage
        step = max(1, len(configs) // 50)
        selected = configs[::step][:50]  # Max 50 per type
        focused_configs.extend(selected)
        print(f"✅ Selected {len(selected)} configs for {model_type}")
    
    print(f"🎯 Total focused configs: {len(focused_configs)}")
    print(f"⏱️ Estimated time: ~{len(focused_configs) * 2 / 60:.1f} minutes")
    
    # Run the focused experiment
    results = []
    failed = 0
    
    for i, config in enumerate(focused_configs):
        try:
            print(f"⚡ Running experiment {i+1}/{len(focused_configs)}", end='', flush=True)
            result = run_comprehensive_experiment(config)
            if result:
                # Parse the config string to extract key info
                config_str = str(config)
                model_match = re.search(r"model_type='([^']+)'", config_str)
                scheduler_match = re.search(r"scheduler_type='([^']+)'", config_str)
                noise_match = re.search(r"noise_type='([^']+)'", config_str)
                
                # Add parsed fields to result
                result['model_type'] = model_match.group(1) if model_match else 'unknown'
                result['scheduler_type'] = scheduler_match.group(1) if scheduler_match else 'unknown'
                result['noise_type'] = noise_match.group(1) if noise_match else 'unknown'
                
                results.append(result)
                print(" ✅")
            else:
                failed += 1
                print(" ❌")
        except Exception as e:
            failed += 1
            print(f" ❌ Error: {e}")
        
        # Save progress every 50 experiments
        if (i + 1) % 50 == 0:
            with open('focused_recovery_progress.json', 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"💾 Progress saved: {len(results)} successful, {failed} failed")
    
    print(f"\n🎉 Focused experiment completed!")
    print(f"✅ Successful: {len(results)}")
    print(f"❌ Failed: {failed}")
    
    # Save final results
    with open('focused_recovery_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Quick analysis
    if results:
        df = pd.DataFrame(results)
        print(f"\n📊 QUICK ANALYSIS:")
        print(f"Model types: {df['model_type'].unique()}")
        print(f"Schedulers: {df['scheduler_type'].unique()}")
        print(f"Noise types: {df['noise_type'].unique()}")
        
        # Group by scheduler and model type
        summary = df.groupby(['scheduler_type', 'model_type'])['metrics'].apply(
            lambda x: pd.json_normalize(x).mean()
        ).reset_index()
        
        print(f"\n📊 Results by scheduler and model type:")
        for scheduler in df['scheduler_type'].unique():
            scheduler_data = df[df['scheduler_type'] == scheduler]
            avg_loss = pd.json_normalize(scheduler_data['metrics'])['final_loss'].mean()
            print(f"  {scheduler}: {avg_loss:.6f} avg final loss")
    
    return results

if __name__ == "__main__":
    run_focused_experiment()