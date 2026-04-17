#!/usr/bin/env python3
"""
Enhanced analysis to understand architecture-specific and noise-condition-specific 
performance of GreedyLR vs other schedulers
"""
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_architecture_specific_performance(results_file='robust_results.json'):
    """
    Analyze performance across different architectures and noise conditions
    to understand where GreedyLR works best
    """
    
    print("🔍 Architecture-Specific Performance Analysis")
    print("=" * 60)
    
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        if not results:
            print("❌ No results found")
            return
            
        print(f"📊 Loaded {len(results)} experimental results")
        
        # Convert to DataFrame with proper extraction
        processed_data = []
        for result in results:
            if isinstance(result, dict):
                metrics = result.get('metrics', {})
                if isinstance(metrics, dict):
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
        
        if not processed_data:
            print("❌ No valid processed data")
            return
            
        df = pd.DataFrame(processed_data)
        
        print(f"📊 Model types: {list(df['model_type'].unique())}")
        print(f"⚙️  Schedulers: {list(df['scheduler_type'].unique())}")
        print(f"🌊 Noise types: {list(df['noise_type'].unique())}")
        
        # 1. NO-NOISE ANALYSIS - Why does GreedyLR perform poorly?
        print("\n" + "="*60)
        print("🧪 NO-NOISE PERFORMANCE ANALYSIS")
        print("="*60)
        
        no_noise_data = df[df['noise_type'] == 'none']
        if len(no_noise_data) > 0:
            print(f"📊 No-noise experiments: {len(no_noise_data)}")
            
            # Performance by scheduler in no-noise conditions
            no_noise_perf = no_noise_data.groupby('scheduler_type').agg({
                'final_loss': ['mean', 'std', 'count'],
                'convergence_rate_50': ['mean', 'std'],
                'stability_score': ['mean', 'std']
            }).round(6)
            
            print("\n📊 NO-NOISE PERFORMANCE BY SCHEDULER:")
            for scheduler in no_noise_perf.index:
                final_loss_mean = no_noise_perf.loc[scheduler, ('final_loss', 'mean')]
                conv_rate_mean = no_noise_perf.loc[scheduler, ('convergence_rate_50', 'mean')]
                count = no_noise_perf.loc[scheduler, ('final_loss', 'count')]
                print(f"  {scheduler:12}: Loss={final_loss_mean:.6f}, Conv={conv_rate_mean:.3f}, N={count}")
            
            # Performance by architecture in no-noise conditions
            print("\n📊 NO-NOISE PERFORMANCE BY ARCHITECTURE:")
            no_noise_arch = no_noise_data.groupby(['model_type', 'scheduler_type'])['final_loss'].mean().unstack()
            print(no_noise_arch.round(6))
            
            # Identify where GreedyLR struggles in no-noise
            if 'greedy' in no_noise_arch.columns:
                struggling_archs = []
                winning_archs = []
                
                for arch in no_noise_arch.index:
                    greedy_loss = no_noise_arch.loc[arch, 'greedy'] if 'greedy' in no_noise_arch.columns else float('inf')
                    best_competitor = no_noise_arch.loc[arch].drop('greedy', errors='ignore').min()
                    
                    if greedy_loss > best_competitor * 1.1:  # 10% worse
                        struggling_archs.append((arch, greedy_loss, best_competitor))
                    elif greedy_loss < best_competitor * 0.9:  # 10% better  
                        winning_archs.append((arch, greedy_loss, best_competitor))
                
                print(f"\n🔍 ARCHITECTURES WHERE GREEDYLR STRUGGLES (no noise):")
                for arch, greedy_loss, best_loss in struggling_archs:
                    print(f"  {arch:15}: GreedyLR={greedy_loss:.6f}, Best={best_loss:.6f} ({greedy_loss/best_loss:.2f}x worse)")
                
                print(f"\n✅ ARCHITECTURES WHERE GREEDYLR EXCELS (no noise):")
                for arch, greedy_loss, best_loss in winning_archs:
                    print(f"  {arch:15}: GreedyLR={greedy_loss:.6f}, Best={best_loss:.6f} ({best_loss/greedy_loss:.2f}x better)")
        
        # 2. NOISE-CONDITION ANALYSIS
        print("\n" + "="*60)
        print("🌊 NOISE-CONDITION PERFORMANCE ANALYSIS")
        print("="*60)
        
        noise_analysis = df.groupby(['noise_type', 'scheduler_type']).agg({
            'final_loss': ['mean', 'std'],
            'robustness_score': ['mean', 'std'],
            'spike_recovery_time': ['mean', 'std']
        }).round(6)
        
        print("\n📊 PERFORMANCE BY NOISE CONDITION:")
        for noise_type in df['noise_type'].unique():
            print(f"\n🌊 {noise_type.upper()} NOISE:")
            noise_data = df[df['noise_type'] == noise_type]
            perf = noise_data.groupby('scheduler_type')['final_loss'].agg(['mean', 'count'])
            
            for scheduler in perf.index:
                mean_loss = perf.loc[scheduler, 'mean']
                count = perf.loc[scheduler, 'count']
                print(f"  {scheduler:12}: {mean_loss:.6f} (N={count})")
        
        # 3. ARCHITECTURE-SPECIFIC SWEET SPOTS
        print("\n" + "="*60)
        print("🎯 ARCHITECTURE-SPECIFIC SWEET SPOTS")
        print("="*60)
        
        # For each architecture, find best scheduler per noise condition
        sweet_spots = {}
        for arch in df['model_type'].unique():
            arch_data = df[df['model_type'] == arch]
            sweet_spots[arch] = {}
            
            for noise in arch_data['noise_type'].unique():
                noise_arch_data = arch_data[arch_data['noise_type'] == noise]
                if len(noise_arch_data) > 0:
                    best_scheduler = noise_arch_data.groupby('scheduler_type')['final_loss'].mean().idxmin()
                    best_loss = noise_arch_data.groupby('scheduler_type')['final_loss'].mean().min()
                    sweet_spots[arch][noise] = (best_scheduler, best_loss)
        
        print("\n🎯 BEST SCHEDULER BY ARCHITECTURE AND NOISE:")
        for arch, noise_dict in sweet_spots.items():
            print(f"\n🏗️  {arch.upper()}:")
            for noise, (scheduler, loss) in noise_dict.items():
                print(f"    {noise:12}: {scheduler:10} (loss={loss:.6f})")
        
        # 4. GREEDYLR DOMINANCE MAP
        print("\n" + "="*60)
        print("🗺️  GREEDYLR DOMINANCE MAP")
        print("="*60)
        
        dominance_map = {}
        for arch in df['model_type'].unique():
            dominance_map[arch] = {}
            arch_data = df[df['model_type'] == arch]
            
            for noise in arch_data['noise_type'].unique():
                noise_arch_data = arch_data[arch_data['noise_type'] == noise]
                if len(noise_arch_data) > 0:
                    scheduler_performance = noise_arch_data.groupby('scheduler_type')['final_loss'].mean()
                    if 'greedy' in scheduler_performance:
                        greedy_loss = scheduler_performance['greedy']
                        best_competitor_loss = scheduler_performance.drop('greedy').min()
                        dominance_ratio = best_competitor_loss / greedy_loss  # >1 means GreedyLR wins
                        dominance_map[arch][noise] = dominance_ratio
                    else:
                        dominance_map[arch][noise] = 0
        
        print("\n🗺️  GreedyLR Dominance Ratio (>1.0 = GreedyLR wins):")
        for arch, noise_dict in dominance_map.items():
            print(f"\n🏗️  {arch.upper()}:")
            for noise, ratio in noise_dict.items():
                status = "🏆 WINS" if ratio > 1.1 else "⚡ CLOSE" if ratio > 0.9 else "❌ LOSES"
                print(f"    {noise:12}: {ratio:.3f} {status}")
        
        # 5. RECOMMENDATIONS
        print("\n" + "="*60)
        print("💡 RECOMMENDATIONS")
        print("="*60)
        
        # Count wins/losses
        greedylr_wins = sum(1 for arch_dict in dominance_map.values() 
                           for ratio in arch_dict.values() if ratio > 1.1)
        greedylr_losses = sum(1 for arch_dict in dominance_map.values() 
                             for ratio in arch_dict.values() if ratio < 0.9)
        total_conditions = sum(len(arch_dict) for arch_dict in dominance_map.values())
        
        print(f"📊 GreedyLR Performance Summary:")
        print(f"   🏆 Wins: {greedylr_wins}/{total_conditions} ({greedylr_wins/total_conditions*100:.1f}%)")
        print(f"   ❌ Losses: {greedylr_losses}/{total_conditions} ({greedylr_losses/total_conditions*100:.1f}%)")
        
        # Save detailed analysis
        analysis_results = {
            'no_noise_analysis': no_noise_perf.to_dict() if len(no_noise_data) > 0 else {},
            'sweet_spots': sweet_spots,
            'dominance_map': dominance_map,
            'summary': {
                'greedylr_wins': greedylr_wins,
                'greedylr_losses': greedylr_losses,
                'total_conditions': total_conditions,
                'win_rate': greedylr_wins / total_conditions if total_conditions > 0 else 0
            }
        }
        
        with open('architecture_specific_analysis.json', 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        print(f"\n💾 Detailed analysis saved to architecture_specific_analysis.json")
        
        return analysis_results
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    analyze_architecture_specific_performance()