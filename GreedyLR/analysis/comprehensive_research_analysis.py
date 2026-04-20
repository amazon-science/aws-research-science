#!/usr/bin/env python3
"""
Comprehensive Research Analysis and Visualization Suite
Creates publication-quality plots and analysis for GreedyLR research paper
"""
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib.patches as mpatches
from scipy import stats
# from sklearn.metrics import mutual_info_score  # Not needed for this analysis
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class ComprehensiveResearchAnalysis:
    def __init__(self, results_file='robust_results.json'):
        self.results_file = results_file
        self.df = None
        self.output_dir = Path('research_plots')
        self.output_dir.mkdir(exist_ok=True)
        
        # Publication color scheme
        self.colors = {
            'greedy': '#2E8B57',      # Sea Green
            'cosine': '#4169E1',      # Royal Blue  
            'cosine_restarts': '#FF6347',  # Tomato
            'exponential': '#9932CC'   # Dark Orchid
        }
        
        # Load and process data
        self.load_and_process_data()
    
    def load_and_process_data(self):
        """Load and preprocess the experimental data"""
        print("📊 Loading experimental data...")
        
        with open(self.results_file, 'r') as f:
            results = json.load(f)
        
        # Process results into flat structure
        processed_data = []
        for result in results:
            if isinstance(result, dict):
                metrics = result.get('metrics', {})
                if isinstance(metrics, dict):
                    flattened = {
                        'model_type': result.get('model_type', 'unknown'),
                        'scheduler_type': result.get('scheduler_type', 'unknown'),
                        'noise_type': result.get('noise_type', 'unknown'),
                        'noise_strength': result.get('noise_strength', 0),
                        'problem_variant': result.get('problem_variant', 'standard'),
                        'final_loss': metrics.get('final_loss', float('inf')),
                        'min_loss': metrics.get('min_loss', float('inf')),
                        'converged_step': metrics.get('converged_step'),
                        'convergence_rate_50': metrics.get('convergence_rate_50', 0),
                        'stability_score': metrics.get('stability_score', 0),
                        'lr_changes': metrics.get('lr_changes', 0),
                        'spike_recovery_time': metrics.get('spike_recovery_time', 0),
                        'robustness_score': metrics.get('robustness_score', 0),
                        'losses': result.get('losses', []),
                        'lrs': result.get('lrs', [])
                    }
                    processed_data.append(flattened)
        
        self.df = pd.DataFrame(processed_data)
        print(f"✅ Processed {len(self.df)} experiments")
        print(f"📊 Architectures: {len(self.df['model_type'].unique())}")
        print(f"⚙️  Schedulers: {len(self.df['scheduler_type'].unique())}")
        print(f"🌊 Noise types: {len(self.df['noise_type'].unique())}")
    
    def create_recovery_visualization(self):
        """Create detailed recovery analysis plots showing GreedyLR's adaptive behavior"""
        print("📈 Creating recovery visualization plots...")
        
        # Filter for spike-based noise types and get sample trajectories
        spike_data = self.df[self.df['noise_type'].isin(['periodic_spike', 'random_spike', 'burst'])]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('GreedyLR Recovery Mechanisms: Adaptive Learning Rate Response to Training Perturbations', 
                     fontsize=16, fontweight='bold', y=0.95)
        
        # Plot 1: Loss trajectory comparison during spike recovery
        ax1 = axes[0, 0]
        
        # Get representative samples for each scheduler
        for scheduler in ['greedy', 'cosine']:
            scheduler_data = spike_data[spike_data['scheduler_type'] == scheduler]
            if len(scheduler_data) > 0:
                # Get a sample with good loss trajectory data
                sample = scheduler_data[scheduler_data['losses'].apply(len) > 100].iloc[0]
                losses = sample['losses'][:150]  # First 150 steps
                
                color = self.colors.get(scheduler, '#666666')
                ax1.plot(losses, label=f'{scheduler.title()} Scheduler', 
                        color=color, linewidth=2.5, alpha=0.8)
        
        ax1.set_xlabel('Training Step', fontweight='bold')
        ax1.set_ylabel('Loss Value', fontweight='bold')
        ax1.set_title('A. Loss Trajectory Recovery from Noise Spikes', fontweight='bold')
        ax1.legend(frameon=True, fancybox=True)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Plot 2: Learning rate adaptation patterns
        ax2 = axes[0, 1]
        
        # Get GreedyLR sample with learning rate changes
        greedy_sample = spike_data[
            (spike_data['scheduler_type'] == 'greedy') & 
            (spike_data['lr_changes'] > 0) &
            (spike_data['lrs'].apply(len) > 100)
        ]
        
        if len(greedy_sample) > 0:
            sample = greedy_sample.iloc[0]
            lrs = sample['lrs'][:150]
            steps = range(len(lrs))
            
            ax2.plot(steps, lrs, color=self.colors['greedy'], linewidth=2.5, 
                    marker='o', markersize=3, alpha=0.8, label='GreedyLR Adaptation')
            
            # Highlight adaptation points
            lr_changes = []
            for i in range(1, len(lrs)):
                if abs(lrs[i] - lrs[i-1]) > 1e-6:
                    lr_changes.append(i)
            
            if lr_changes:
                ax2.scatter([steps[i] for i in lr_changes], [lrs[i] for i in lr_changes], 
                           color='red', s=50, alpha=0.8, zorder=5, label='LR Adaptations')
        
        ax2.set_xlabel('Training Step', fontweight='bold')
        ax2.set_ylabel('Learning Rate', fontweight='bold')  
        ax2.set_title('B. GreedyLR Adaptive Learning Rate Response', fontweight='bold')
        ax2.legend(frameon=True, fancybox=True)
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # Plot 3: Recovery time distribution
        ax3 = axes[1, 0]
        
        recovery_data = spike_data[spike_data['spike_recovery_time'] > 0]
        schedulers = recovery_data['scheduler_type'].unique()
        
        recovery_times = []
        labels = []
        colors_list = []
        
        for scheduler in schedulers:
            times = recovery_data[recovery_data['scheduler_type'] == scheduler]['spike_recovery_time']
            if len(times) > 0:
                recovery_times.append(times)
                labels.append(scheduler.title())
                colors_list.append(self.colors.get(scheduler, '#666666'))
        
        if recovery_times:
            bp = ax3.boxplot(recovery_times, labels=labels, patch_artist=True, 
                            boxprops=dict(facecolor='lightblue', alpha=0.7),
                            medianprops=dict(color='red', linewidth=2))
            
            for patch, color in zip(bp['boxes'], colors_list):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        ax3.set_ylabel('Recovery Time (Steps)', fontweight='bold')
        ax3.set_title('C. Spike Recovery Time Distribution', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Robustness score by noise type
        ax4 = axes[1, 1]
        
        robustness_pivot = self.df.groupby(['noise_type', 'scheduler_type'])['robustness_score'].mean().unstack()
        
        x = np.arange(len(robustness_pivot.index))
        width = 0.2
        
        for i, scheduler in enumerate(robustness_pivot.columns):
            if scheduler in self.colors:
                values = robustness_pivot[scheduler].values
                ax4.bar(x + i*width, values, width, label=scheduler.title(), 
                       color=self.colors[scheduler], alpha=0.8)
        
        ax4.set_xlabel('Noise Type', fontweight='bold')
        ax4.set_ylabel('Robustness Score', fontweight='bold')
        ax4.set_title('D. Robustness Performance by Noise Condition', fontweight='bold')
        ax4.set_xticks(x + width * 1.5)
        ax4.set_xticklabels(robustness_pivot.index, rotation=45, ha='right')
        ax4.legend(frameon=True, fancybox=True)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'greedylr_recovery_mechanisms.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(self.output_dir / 'greedylr_recovery_mechanisms.pdf', 
                   bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"✅ Recovery visualization saved to {self.output_dir}")
    
    def create_architecture_performance_heatmap(self):
        """Create comprehensive architecture vs scheduler performance heatmap"""
        print("🗺️ Creating architecture performance heatmap...")
        
        # Create performance matrix
        perf_matrix = self.df.groupby(['model_type', 'scheduler_type'])['final_loss'].mean().unstack()
        
        # Calculate relative performance (lower is better, so we'll use 1/loss for coloring)
        log_matrix = np.log10(perf_matrix + 1e-10)  # Add small epsilon to avoid log(0)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create heatmap
        im = ax.imshow(log_matrix.values, cmap='RdYlGn_r', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(perf_matrix.columns)))
        ax.set_yticks(np.arange(len(perf_matrix.index)))
        ax.set_xticklabels([col.title().replace('_', ' ') for col in perf_matrix.columns])
        ax.set_yticklabels([idx.replace('neural_', '').replace('_', ' ').title() for idx in perf_matrix.index])
        
        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        for i in range(len(perf_matrix.index)):
            for j in range(len(perf_matrix.columns)):
                value = perf_matrix.iloc[i, j]
                if not pd.isna(value):
                    text = ax.text(j, i, f'{value:.3f}', ha="center", va="center", 
                                 color="white" if log_matrix.iloc[i, j] > log_matrix.values.mean() else "black",
                                 fontweight='bold', fontsize=9)
        
        ax.set_title('Architecture-Specific Scheduler Performance Matrix\n(Final Loss Values - Lower is Better)', 
                    fontweight='bold', fontsize=14, pad=20)
        ax.set_xlabel('Learning Rate Scheduler', fontweight='bold')
        ax.set_ylabel('Model Architecture', fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Log₁₀(Final Loss)', rotation=270, labelpad=20, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'architecture_performance_heatmap.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(self.output_dir / 'architecture_performance_heatmap.pdf', 
                   bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"✅ Architecture heatmap saved to {self.output_dir}")
    
    def create_dominance_analysis(self):
        """Create GreedyLR dominance analysis with statistical significance"""
        print("🏆 Creating dominance analysis...")
        
        # Calculate dominance ratios
        dominance_data = []
        
        for arch in self.df['model_type'].unique():
            arch_data = self.df[self.df['model_type'] == arch]
            
            for noise in arch_data['noise_type'].unique():
                noise_arch_data = arch_data[arch_data['noise_type'] == noise]
                
                if len(noise_arch_data) > 0:
                    scheduler_performance = noise_arch_data.groupby('scheduler_type')['final_loss'].agg(['mean', 'std', 'count'])
                    
                    if 'greedy' in scheduler_performance.index:
                        greedy_mean = scheduler_performance.loc['greedy', 'mean']
                        greedy_std = scheduler_performance.loc['greedy', 'std']
                        greedy_n = scheduler_performance.loc['greedy', 'count']
                        
                        # Find best competitor
                        competitors = scheduler_performance.drop('greedy')
                        if len(competitors) > 0:
                            best_competitor = competitors['mean'].min()
                            best_competitor_name = competitors['mean'].idxmin()
                            best_std = competitors.loc[best_competitor_name, 'std']
                            best_n = competitors.loc[best_competitor_name, 'count']
                            
                            # Calculate statistical significance using Welch's t-test
                            noise_greedy_data = noise_arch_data[noise_arch_data['scheduler_type'] == 'greedy']['final_loss']
                            noise_best_data = noise_arch_data[noise_arch_data['scheduler_type'] == best_competitor_name]['final_loss']
                            
                            if len(noise_greedy_data) > 1 and len(noise_best_data) > 1:
                                t_stat, p_value = stats.ttest_ind(noise_greedy_data, noise_best_data, equal_var=False)
                                
                                dominance_ratio = best_competitor / greedy_mean if greedy_mean > 0 else 0
                                effect_size = (best_competitor - greedy_mean) / np.sqrt((greedy_std**2 + best_std**2) / 2)
                                
                                dominance_data.append({
                                    'architecture': arch,
                                    'noise_type': noise,
                                    'dominance_ratio': dominance_ratio,
                                    'effect_size': effect_size,
                                    'p_value': p_value,
                                    'greedy_mean': greedy_mean,
                                    'competitor_mean': best_competitor,
                                    'competitor_name': best_competitor_name,
                                    'significant': p_value < 0.05,
                                    'greedylr_wins': dominance_ratio > 1.1 and p_value < 0.05
                                })
        
        dominance_df = pd.DataFrame(dominance_data)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('GreedyLR Dominance Analysis: Statistical Performance Assessment', 
                     fontsize=16, fontweight='bold', y=0.95)
        
        # Plot 1: Dominance ratio heatmap
        ax1 = axes[0, 0]
        dominance_pivot = dominance_df.pivot(index='architecture', columns='noise_type', values='dominance_ratio')
        
        # Create custom colormap (red for losses, green for wins)
        from matplotlib.colors import TwoSlopeNorm
        norm = TwoSlopeNorm(vmin=dominance_pivot.min().min(), vcenter=1.0, vmax=dominance_pivot.max().max())
        
        im1 = ax1.imshow(dominance_pivot.values, cmap='RdYlGn', norm=norm, aspect='auto')
        ax1.set_xticks(np.arange(len(dominance_pivot.columns)))
        ax1.set_yticks(np.arange(len(dominance_pivot.index)))
        ax1.set_xticklabels(dominance_pivot.columns, rotation=45, ha='right')
        ax1.set_yticklabels([idx.replace('neural_', '').title() for idx in dominance_pivot.index])
        ax1.set_title('A. Dominance Ratio Matrix\n(>1.0 = GreedyLR Wins)', fontweight='bold')
        
        # Add significance markers
        for i, arch in enumerate(dominance_pivot.index):
            for j, noise in enumerate(dominance_pivot.columns):
                if not pd.isna(dominance_pivot.iloc[i, j]):
                    row_data = dominance_df[(dominance_df['architecture'] == arch) & 
                                          (dominance_df['noise_type'] == noise)]
                    if len(row_data) > 0 and row_data.iloc[0]['significant']:
                        ax1.text(j, i, '*', ha='center', va='center', 
                               color='white' if abs(dominance_pivot.iloc[i, j] - 1) > 0.5 else 'black',
                               fontsize=16, fontweight='bold')
        
        # Plot 2: Win rate by architecture
        ax2 = axes[0, 1]
        win_rates = dominance_df.groupby('architecture')['greedylr_wins'].mean().sort_values(ascending=True)
        
        bars = ax2.barh(range(len(win_rates)), win_rates.values, 
                       color=[self.colors['greedy'] if x > 0.5 else '#FF6B6B' for x in win_rates.values])
        ax2.set_yticks(range(len(win_rates)))
        ax2.set_yticklabels([idx.replace('neural_', '').title() for idx in win_rates.index])
        ax2.set_xlabel('Win Rate', fontweight='bold')
        ax2.set_title('B. GreedyLR Win Rate by Architecture', fontweight='bold')
        ax2.axvline(x=0.5, color='black', linestyle='--', alpha=0.5)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(win_rates.values):
            ax2.text(v + 0.02, i, f'{v:.1%}', va='center', fontweight='bold')
        
        # Plot 3: Effect size distribution
        ax3 = axes[1, 0]
        
        # Separate wins and losses
        wins = dominance_df[dominance_df['greedylr_wins']]['effect_size']
        losses = dominance_df[~dominance_df['greedylr_wins']]['effect_size']
        
        ax3.hist(losses, bins=20, alpha=0.7, color='#FF6B6B', label=f'Losses (n={len(losses)})', density=True)
        ax3.hist(wins, bins=20, alpha=0.7, color=self.colors['greedy'], label=f'Wins (n={len(wins)})', density=True)
        
        ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Effect Size (Cohen\'s d)', fontweight='bold')
        ax3.set_ylabel('Density', fontweight='bold')
        ax3.set_title('C. Effect Size Distribution', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Statistical summary
        ax4 = axes[1, 1]
        
        # Create summary statistics
        total_comparisons = len(dominance_df)
        significant_wins = len(dominance_df[dominance_df['greedylr_wins']])
        significant_losses = len(dominance_df[(dominance_df['p_value'] < 0.05) & (dominance_df['dominance_ratio'] < 0.9)])
        not_significant = len(dominance_df[dominance_df['p_value'] >= 0.05])
        
        categories = ['Significant\nWins', 'Significant\nLosses', 'Not\nSignificant']
        values = [significant_wins, significant_losses, not_significant]
        colors_bar = [self.colors['greedy'], '#FF6B6B', '#FFA500']
        
        bars = ax4.bar(categories, values, color=colors_bar, alpha=0.8)
        ax4.set_ylabel('Number of Comparisons', fontweight='bold')
        ax4.set_title('D. Statistical Significance Summary', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add percentage labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{value}\n({value/total_comparisons:.1%})',
                    ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'greedylr_dominance_analysis.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(self.output_dir / 'greedylr_dominance_analysis.pdf', 
                   bbox_inches='tight', facecolor='white')
        plt.show()
        
        # Save detailed results
        dominance_df.to_csv(self.output_dir / 'dominance_analysis_detailed.csv', index=False)
        
        print(f"✅ Dominance analysis saved to {self.output_dir}")
        print(f"📊 Summary: {significant_wins}/{total_comparisons} significant wins ({significant_wins/total_comparisons:.1%})")
        
        return dominance_df
    
    def create_noise_robustness_analysis(self):
        """Create detailed noise robustness analysis"""
        print("🌊 Creating noise robustness analysis...")
        
        # Calculate improvement ratios vs no-noise baseline
        noise_analysis = []
        
        for arch in self.df['model_type'].unique():
            arch_data = self.df[self.df['model_type'] == arch]
            
            # Get baseline (no-noise) performance for each scheduler
            baseline_data = arch_data[arch_data['noise_type'] == 'none']
            baseline_performance = baseline_data.groupby('scheduler_type')['final_loss'].mean()
            
            for noise in arch_data['noise_type'].unique():
                if noise != 'none':
                    noise_data = arch_data[arch_data['noise_type'] == noise]
                    noise_performance = noise_data.groupby('scheduler_type')['final_loss'].mean()
                    
                    for scheduler in noise_performance.index:
                        if scheduler in baseline_performance.index:
                            baseline_loss = baseline_performance[scheduler]
                            noise_loss = noise_performance[scheduler]
                            
                            # Calculate degradation (how much worse due to noise)
                            degradation_ratio = noise_loss / baseline_loss if baseline_loss > 0 else float('inf')
                            
                            noise_analysis.append({
                                'architecture': arch,
                                'scheduler': scheduler,
                                'noise_type': noise,
                                'baseline_loss': baseline_loss,
                                'noise_loss': noise_loss,
                                'degradation_ratio': degradation_ratio,
                                'robustness_score': 1 / degradation_ratio if degradation_ratio > 0 else 0
                            })
        
        noise_df = pd.DataFrame(noise_analysis)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Noise Robustness Analysis: Performance Degradation Under Perturbations', 
                     fontsize=16, fontweight='bold', y=0.95)
        
        # Plot 1: Average robustness by scheduler
        ax1 = axes[0, 0]
        
        avg_robustness = noise_df.groupby('scheduler')['robustness_score'].agg(['mean', 'std']).sort_values('mean', ascending=False)
        
        x_pos = np.arange(len(avg_robustness))
        bars = ax1.bar(x_pos, avg_robustness['mean'], 
                      yerr=avg_robustness['std'], 
                      color=[self.colors.get(sched, '#666666') for sched in avg_robustness.index],
                      alpha=0.8, capsize=5)
        
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([sched.title() for sched in avg_robustness.index])
        ax1.set_ylabel('Average Robustness Score', fontweight='bold')
        ax1.set_title('A. Overall Noise Robustness by Scheduler', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, mean_val in zip(bars, avg_robustness['mean']):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Robustness by noise type
        ax2 = axes[0, 1]
        
        noise_robustness = noise_df.groupby(['noise_type', 'scheduler'])['robustness_score'].mean().unstack()
        
        noise_robustness.plot(kind='bar', ax=ax2, 
                             color=[self.colors.get(col, '#666666') for col in noise_robustness.columns],
                             alpha=0.8)
        ax2.set_xlabel('Noise Type', fontweight='bold')
        ax2.set_ylabel('Robustness Score', fontweight='bold')
        ax2.set_title('B. Robustness Score by Noise Type', fontweight='bold')
        ax2.legend(title='Scheduler', title_fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Architecture-specific robustness
        ax3 = axes[1, 0]
        
        arch_robustness = noise_df[noise_df['scheduler'] == 'greedy'].groupby('architecture')['robustness_score'].mean().sort_values(ascending=True)
        
        bars = ax3.barh(range(len(arch_robustness)), arch_robustness.values, 
                       color=self.colors['greedy'], alpha=0.8)
        ax3.set_yticks(range(len(arch_robustness)))
        ax3.set_yticklabels([arch.replace('neural_', '').title() for arch in arch_robustness.index])
        ax3.set_xlabel('GreedyLR Robustness Score', fontweight='bold')
        ax3.set_title('C. GreedyLR Robustness by Architecture', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Relative robustness advantage
        ax4 = axes[1, 1]
        
        # Calculate GreedyLR advantage over best competitor in each condition
        advantage_data = []
        
        for noise in noise_df['noise_type'].unique():
            noise_subset = noise_df[noise_df['noise_type'] == noise]
            
            for arch in noise_subset['architecture'].unique():
                arch_noise_data = noise_subset[noise_subset['architecture'] == arch]
                
                if 'greedy' in arch_noise_data['scheduler'].values:
                    greedy_score = arch_noise_data[arch_noise_data['scheduler'] == 'greedy']['robustness_score'].iloc[0]
                    competitor_scores = arch_noise_data[arch_noise_data['scheduler'] != 'greedy']['robustness_score']
                    
                    if len(competitor_scores) > 0:
                        best_competitor = competitor_scores.max()
                        advantage = greedy_score / best_competitor if best_competitor > 0 else float('inf')
                        
                        advantage_data.append({
                            'noise_type': noise,
                            'architecture': arch,
                            'advantage_ratio': advantage
                        })
        
        advantage_df = pd.DataFrame(advantage_data)
        
        if len(advantage_df) > 0:
            avg_advantage = advantage_df.groupby('noise_type')['advantage_ratio'].mean().sort_values(ascending=False)
            
            bars = ax4.bar(range(len(avg_advantage)), avg_advantage.values, 
                          color='lightcoral', alpha=0.8)
            ax4.set_xticks(range(len(avg_advantage)))
            ax4.set_xticklabels(avg_advantage.index, rotation=45, ha='right')
            ax4.set_ylabel('GreedyLR Advantage Ratio', fontweight='bold')
            ax4.set_title('D. GreedyLR Robustness Advantage by Noise', fontweight='bold')
            ax4.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Parity')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'noise_robustness_analysis.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(self.output_dir / 'noise_robustness_analysis.pdf', 
                   bbox_inches='tight', facecolor='white')
        plt.show()
        
        # Save detailed results
        noise_df.to_csv(self.output_dir / 'noise_robustness_detailed.csv', index=False)
        
        print(f"✅ Noise robustness analysis saved to {self.output_dir}")
        
        return noise_df
    
    def run_comprehensive_analysis(self):
        """Run all analysis components"""
        print("🚀 Running comprehensive research analysis suite...")
        
        # Create all visualizations
        recovery_analysis = self.create_recovery_visualization()
        heatmap_analysis = self.create_architecture_performance_heatmap()
        dominance_analysis = self.create_dominance_analysis()
        robustness_analysis = self.create_noise_robustness_analysis()
        
        # Generate summary statistics
        summary_stats = self.generate_summary_statistics()
        
        print(f"\n🎉 Comprehensive analysis complete!")
        print(f"📁 All plots saved to: {self.output_dir}")
        print(f"📊 Generated {len(list(self.output_dir.glob('*.png')))} publication-quality figures")
        
        return {
            'dominance_analysis': dominance_analysis,
            'robustness_analysis': robustness_analysis,
            'summary_stats': summary_stats
        }
    
    def generate_summary_statistics(self):
        """Generate comprehensive summary statistics for the research paper"""
        print("📊 Generating summary statistics...")
        
        stats = {}
        
        # Overall performance
        overall_perf = self.df.groupby('scheduler_type').agg({
            'final_loss': ['mean', 'median', 'std'],
            'convergence_rate_50': ['mean', 'std'],
            'robustness_score': ['mean', 'std']
        }).round(6)
        
        stats['overall_performance'] = overall_perf
        
        # No-noise specific performance
        no_noise = self.df[self.df['noise_type'] == 'none']
        no_noise_perf = no_noise.groupby('scheduler_type')['final_loss'].agg(['mean', 'std', 'count'])
        stats['no_noise_performance'] = no_noise_perf
        
        # Architecture-specific wins
        arch_wins = {}
        for arch in self.df['model_type'].unique():
            arch_data = self.df[self.df['model_type'] == arch]
            best_scheduler = arch_data.groupby('scheduler_type')['final_loss'].mean().idxmin()
            arch_wins[arch] = best_scheduler
        
        stats['architecture_winners'] = arch_wins
        
        # Statistical significance tests
        greedy_losses = self.df[self.df['scheduler_type'] == 'greedy']['final_loss']
        cosine_losses = self.df[self.df['scheduler_type'] == 'cosine']['final_loss']
        
        if len(greedy_losses) > 0 and len(cosine_losses) > 0:
            t_stat, p_value = stats.ttest_ind(greedy_losses, cosine_losses)
            effect_size = (greedy_losses.mean() - cosine_losses.mean()) / np.sqrt((greedy_losses.var() + cosine_losses.var()) / 2)
            
            stats['statistical_tests'] = {
                'greedy_vs_cosine_t_test': {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'effect_size': effect_size,
                    'significant': p_value < 0.05
                }
            }
        
        # Save statistics
        with open(self.output_dir / 'summary_statistics.json', 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        print(f"✅ Summary statistics saved to {self.output_dir}/summary_statistics.json")
        
        return stats

def main():
    """Main execution function"""
    analyzer = ComprehensiveResearchAnalysis('robust_results.json')
    results = analyzer.run_comprehensive_analysis()
    
    print("\n" + "="*60)
    print("📊 COMPREHENSIVE RESEARCH ANALYSIS COMPLETE")
    print("="*60)
    print(f"🔬 Total experiments analyzed: {len(analyzer.df):,}")
    print(f"🏗️  Architectures tested: {len(analyzer.df['model_type'].unique())}")
    print(f"⚙️  Schedulers compared: {len(analyzer.df['scheduler_type'].unique())}")
    print(f"🌊 Noise conditions: {len(analyzer.df['noise_type'].unique())}")
    print(f"📁 Output directory: {analyzer.output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()