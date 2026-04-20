#!/usr/bin/env python3
"""
Journal-Quality Publication Plots for GreedyLR Research
Addresses format and technical issues for academic publication
"""
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import matplotlib.patches as mpatches

# Set journal-quality style
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.linewidth': 0.8,
    'grid.alpha': 0.3,
    'lines.linewidth': 2.5,
    'patch.linewidth': 0.5,
    'xtick.major.size': 6,
    'xtick.minor.size': 4,
    'ytick.major.size': 6,
    'ytick.minor.size': 4,
    'legend.frameon': True,
    'legend.fancybox': True,
    'legend.shadow': True,
    'figure.figsize': (10, 8),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

class JournalQualityPlots:
    def __init__(self, results_file='robust_results.json'):
        self.results_file = results_file
        self.df = None
        self.output_dir = Path('journal_plots')
        self.output_dir.mkdir(exist_ok=True)
        
        # Professional color scheme - colorblind friendly
        self.colors = {
            'greedy': '#2E8B57',      # Sea Green (GreedyLR - primary)
            'cosine': '#1E90FF',      # Dodger Blue  
            'cosine_restarts': '#FF6347',  # Tomato
            'exponential': '#9370DB',  # Medium Purple
            'improvement': '#228B22',  # Forest Green
            'degradation': '#DC143C',  # Crimson
            'neutral': '#696969'       # Dim Gray
        }
        
        self.load_data()
    
    def load_data(self):
        """Load and preprocess data"""
        print("📊 Loading experimental data for journal plots...")
        
        with open(self.results_file, 'r') as f:
            results = json.load(f)
        
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
        print(f"✅ Loaded {len(self.df)} experiments across {len(self.df['model_type'].unique())} architectures")
    
    def plot_1_overall_performance_comparison(self):
        """Figure 1: Overall Performance Comparison - GreedyLR vs Competitors"""
        print("📊 Creating Figure 1: Overall Performance Comparison")
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Calculate overall performance metrics
        performance_metrics = self.df.groupby('scheduler_type').agg({
            'final_loss': ['mean', 'std'],
            'convergence_rate_50': ['mean', 'std'],
            'robustness_score': ['mean', 'std']
        }).round(4)
        
        # Focus on final loss (primary metric)
        schedulers = performance_metrics.index.tolist()
        means = performance_metrics[('final_loss', 'mean')].values
        stds = performance_metrics[('final_loss', 'std')].values
        
        # Create bars with emphasis on GreedyLR
        colors = [self.colors['greedy'] if s == 'greedy' else self.colors.get(s, self.colors['neutral']) 
                 for s in schedulers]
        alphas = [0.9 if s == 'greedy' else 0.7 for s in schedulers]
        
        bars = ax.bar(range(len(schedulers)), means, yerr=stds, 
                     color=colors, alpha=alphas, capsize=8, capthick=2,
                     edgecolor='black', linewidth=1.2)
        
        # Highlight GreedyLR bar
        greedy_idx = schedulers.index('greedy') if 'greedy' in schedulers else -1
        if greedy_idx != -1:
            bars[greedy_idx].set_edgecolor('#000000')
            bars[greedy_idx].set_linewidth(2.5)
            bars[greedy_idx].set_alpha(1.0)
        
        # Formatting
        ax.set_xticks(range(len(schedulers)))
        ax.set_xticklabels([s.replace('_', ' ').title() for s in schedulers], fontsize=12, fontweight='bold')
        ax.set_ylabel('Final Loss (Lower is Better)', fontsize=14, fontweight='bold')
        ax.set_title('Overall Performance Comparison Across All Experiments\\n8,100 Training Runs, 12 Architectures, 9 Noise Conditions', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_yscale('log')
        
        # Add value labels on bars
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            height = bar.get_height()
            improvement = ""
            if i != greedy_idx and greedy_idx != -1:
                ratio = means[greedy_idx] / mean
                if ratio < 1:
                    improvement = f"\\n({1/ratio:.1f}× better)"
            
            ax.text(bar.get_x() + bar.get_width()/2., height * 1.5,
                   f'{mean:.3f}±{std:.3f}{improvement}',
                   ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Add statistical significance annotation
        if greedy_idx != -1:
            greedy_data = self.df[self.df['scheduler_type'] == 'greedy']['final_loss']
            for i, scheduler in enumerate(schedulers):
                if scheduler != 'greedy':
                    comp_data = self.df[self.df['scheduler_type'] == scheduler]['final_loss']
                    if len(greedy_data) > 1 and len(comp_data) > 1:
                        _, p_value = stats.ttest_ind(greedy_data, comp_data)
                        if p_value < 0.001:
                            ax.text(i, means[i] * 3, '***', ha='center', va='bottom', 
                                   fontsize=16, fontweight='bold', color='red')
                        elif p_value < 0.01:
                            ax.text(i, means[i] * 3, '**', ha='center', va='bottom', 
                                   fontsize=16, fontweight='bold', color='red')
                        elif p_value < 0.05:
                            ax.text(i, means[i] * 3, '*', ha='center', va='bottom', 
                                   fontsize=16, fontweight='bold', color='red')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure_1_overall_performance.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'figure_1_overall_performance.pdf', bbox_inches='tight')
        plt.show()
        print(f"✅ Figure 1 saved to {self.output_dir}")
    
    def plot_2_noise_robustness_showcase(self):
        """Figure 2: GreedyLR's Noise Robustness - Where it Excels"""
        print("📊 Creating Figure 2: Noise Robustness Showcase")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('GreedyLR Noise Robustness: Performance Under Training Perturbations', 
                     fontsize=16, fontweight='bold', y=0.95)
        
        # Subplot A: No-noise vs Noisy conditions comparison
        ax1 = axes[0, 0]
        
        no_noise_data = self.df[self.df['noise_type'] == 'none']
        noisy_data = self.df[self.df['noise_type'] != 'none']
        
        conditions = ['Clean Training', 'Noisy Training']
        greedy_means = [
            no_noise_data[no_noise_data['scheduler_type'] == 'greedy']['final_loss'].mean(),
            noisy_data[noisy_data['scheduler_type'] == 'greedy']['final_loss'].mean()
        ]
        cosine_means = [
            no_noise_data[no_noise_data['scheduler_type'] == 'cosine']['final_loss'].mean(),
            noisy_data[noisy_data['scheduler_type'] == 'cosine']['final_loss'].mean()
        ]
        
        x = np.arange(len(conditions))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, greedy_means, width, label='GreedyLR', 
                       color=self.colors['greedy'], alpha=0.9, edgecolor='black')
        bars2 = ax1.bar(x + width/2, cosine_means, width, label='Cosine Annealing', 
                       color=self.colors['cosine'], alpha=0.7, edgecolor='black')
        
        ax1.set_ylabel('Average Final Loss', fontweight='bold')
        ax1.set_title('A. Clean vs Noisy Training Performance', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(conditions)
        ax1.legend()
        ax1.set_yscale('log')
        
        # Add improvement annotations
        for i, (g, c) in enumerate(zip(greedy_means, cosine_means)):
            if c > g:
                improvement = c / g
                ax1.annotate(f'{improvement:.1f}× better', 
                           xy=(i - width/2, g), xytext=(i, max(g, c) * 2),
                           arrowprops=dict(arrowstyle='->', color='red', lw=2),
                           ha='center', fontweight='bold', color='red')
        
        # Subplot B: Recovery demonstration with sample trajectories
        ax2 = axes[0, 1]
        
        # Get sample recovery trajectories
        spike_data = self.df[self.df['noise_type'].isin(['periodic_spike', 'random_spike'])]
        
        # Find good examples of recovery
        greedy_sample = spike_data[
            (spike_data['scheduler_type'] == 'greedy') & 
            (spike_data['losses'].apply(len) > 100) & 
            (spike_data['lr_changes'] > 3)
        ]
        
        cosine_sample = spike_data[
            (spike_data['scheduler_type'] == 'cosine') & 
            (spike_data['losses'].apply(len) > 100)
        ]
        
        if len(greedy_sample) > 0 and len(cosine_sample) > 0:
            greedy_losses = greedy_sample.iloc[0]['losses'][:100]
            cosine_losses = cosine_sample.iloc[0]['losses'][:100]
            
            steps = range(len(greedy_losses))
            ax2.plot(steps, greedy_losses, color=self.colors['greedy'], 
                    linewidth=3, label='GreedyLR', alpha=0.9)
            ax2.plot(steps[:len(cosine_losses)], cosine_losses, color=self.colors['cosine'], 
                    linewidth=2, label='Cosine Annealing', alpha=0.7, linestyle='--')
            
            ax2.set_xlabel('Training Step')
            ax2.set_ylabel('Loss Value')
            ax2.set_title('B. Sample Recovery Trajectory from Noise Spikes', fontweight='bold')
            ax2.legend()
            ax2.set_yscale('log')
        
        # Subplot C: Architecture-specific GreedyLR advantages
        ax3 = axes[1, 0]
        
        # Calculate GreedyLR advantage for each architecture
        advantages = []
        arch_names = []
        
        for arch in self.df['model_type'].unique():
            arch_data = self.df[self.df['model_type'] == arch]
            perf_by_scheduler = arch_data.groupby('scheduler_type')['final_loss'].mean()
            
            if 'greedy' in perf_by_scheduler.index:
                greedy_loss = perf_by_scheduler['greedy']
                competitors = perf_by_scheduler.drop('greedy')
                if len(competitors) > 0:
                    best_competitor = competitors.min()
                    advantage_ratio = best_competitor / greedy_loss
                    
                    if advantage_ratio > 1.1:  # Only show where GreedyLR wins significantly
                        advantages.append(advantage_ratio)
                        arch_names.append(arch.replace('neural_', '').replace('_', ' ').title())
        
        if advantages:
            # Sort by advantage
            sorted_data = sorted(zip(advantages, arch_names), reverse=True)
            advantages, arch_names = zip(*sorted_data)
            
            bars = ax3.barh(range(len(advantages)), advantages, 
                           color=self.colors['improvement'], alpha=0.8, edgecolor='black')
            ax3.set_yticks(range(len(advantages)))
            ax3.set_yticklabels(arch_names)
            ax3.set_xlabel('Performance Advantage (×)', fontweight='bold')
            ax3.set_title('C. GreedyLR Architectural Advantages', fontweight='bold')
            ax3.axvline(x=1.0, color='black', linestyle='--', alpha=0.5)
            
            # Add value labels
            for i, (bar, advantage) in enumerate(zip(bars, advantages)):
                width = bar.get_width()
                ax3.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                        f'{advantage:.1f}×', ha='left', va='center', fontweight='bold')
        
        # Subplot D: Noise type specific improvements
        ax4 = axes[1, 1]
        
        noise_improvements = []
        noise_names = []
        
        for noise in self.df['noise_type'].unique():
            if noise != 'none':
                noise_data = self.df[self.df['noise_type'] == noise]
                perf_by_scheduler = noise_data.groupby('scheduler_type')['final_loss'].mean()
                
                if 'greedy' in perf_by_scheduler.index and 'cosine' in perf_by_scheduler.index:
                    greedy_loss = perf_by_scheduler['greedy']
                    cosine_loss = perf_by_scheduler['cosine']
                    
                    if cosine_loss > greedy_loss:
                        improvement = cosine_loss / greedy_loss
                        noise_improvements.append(improvement)
                        noise_names.append(noise.replace('_', ' ').title())
        
        if noise_improvements:
            # Sort by improvement
            sorted_data = sorted(zip(noise_improvements, noise_names), reverse=True)
            noise_improvements, noise_names = zip(*sorted_data)
            
            bars = ax4.bar(range(len(noise_improvements)), noise_improvements,
                          color=self.colors['improvement'], alpha=0.8, edgecolor='black')
            ax4.set_xticks(range(len(noise_improvements)))
            ax4.set_xticklabels(noise_names, rotation=45, ha='right')
            ax4.set_ylabel('Improvement over Cosine (×)', fontweight='bold')
            ax4.set_title('D. GreedyLR Improvements by Noise Type', fontweight='bold')
            ax4.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
            
            # Add value labels
            for bar, improvement in zip(bars, noise_improvements):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{improvement:.1f}×', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure_2_noise_robustness.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'figure_2_noise_robustness.pdf', bbox_inches='tight')
        plt.show()
        print(f"✅ Figure 2 saved to {self.output_dir}")
    
    def plot_3_learning_rate_adaptation(self):
        """Figure 3: Learning Rate Adaptation Mechanisms"""
        print("📊 Creating Figure 3: Learning Rate Adaptation Mechanisms")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('GreedyLR Adaptive Learning Rate Mechanisms', 
                     fontsize=16, fontweight='bold', y=0.95)
        
        # Get GreedyLR samples with good adaptation patterns
        greedy_data = self.df[
            (self.df['scheduler_type'] == 'greedy') & 
            (self.df['lr_changes'] > 5) &
            (self.df['lrs'].apply(len) > 50)
        ]
        
        if len(greedy_data) > 0:
            # Subplot A: Learning rate trajectory
            ax1 = axes[0, 0]
            sample = greedy_data.iloc[0]
            lrs = sample['lrs'][:100]
            steps = range(len(lrs))
            
            ax1.plot(steps, lrs, color=self.colors['greedy'], linewidth=3, alpha=0.9)
            ax1.scatter([i for i in range(1, len(lrs)) if abs(lrs[i] - lrs[i-1]) > 1e-6], 
                       [lrs[i] for i in range(1, len(lrs)) if abs(lrs[i] - lrs[i-1]) > 1e-6],
                       color='red', s=50, alpha=0.8, zorder=5, label='Adaptations')
            
            ax1.set_xlabel('Training Step')
            ax1.set_ylabel('Learning Rate')
            ax1.set_title('A. Learning Rate Adaptation Pattern', fontweight='bold')
            ax1.set_yscale('log')
            ax1.legend()
            
            # Subplot B: Loss vs LR correlation
            ax2 = axes[0, 1]
            losses = sample['losses'][:len(lrs)]
            
            # Create scatter plot showing LR changes in response to loss
            colors_scatter = ['red' if i > 0 and abs(lrs[i] - lrs[i-1]) > 1e-6 else 'blue' 
                            for i in range(len(losses))]
            ax2.scatter(losses, lrs, c=colors_scatter, alpha=0.6, s=30)
            
            ax2.set_xlabel('Loss Value')
            ax2.set_ylabel('Learning Rate')
            ax2.set_title('B. Loss-LR Adaptation Relationship', fontweight='bold')
            ax2.set_xscale('log')
            ax2.set_yscale('log')
            
            # Add legend
            red_patch = mpatches.Patch(color='red', label='LR Change Steps')
            blue_patch = mpatches.Patch(color='blue', label='No LR Change')
            ax2.legend(handles=[red_patch, blue_patch])
        
        # Subplot C: Adaptation frequency by noise type
        ax3 = axes[1, 0]
        
        adaptation_freq = self.df[self.df['scheduler_type'] == 'greedy'].groupby('noise_type')['lr_changes'].mean()
        
        bars = ax3.bar(range(len(adaptation_freq)), adaptation_freq.values,
                      color=self.colors['greedy'], alpha=0.8, edgecolor='black')
        ax3.set_xticks(range(len(adaptation_freq)))
        ax3.set_xticklabels([noise.replace('_', ' ').title() for noise in adaptation_freq.index], 
                           rotation=45, ha='right')
        ax3.set_ylabel('Average LR Changes per Training', fontweight='bold')
        ax3.set_title('C. Adaptation Frequency by Noise Type', fontweight='bold')
        
        # Subplot D: Adaptation effectiveness
        ax4 = axes[1, 1]
        
        # Calculate correlation between LR changes and final performance
        effectiveness_data = []
        for _, row in self.df[self.df['scheduler_type'] == 'greedy'].iterrows():
            lr_changes = row['lr_changes']
            final_loss = row['final_loss']
            if lr_changes > 0 and final_loss > 0:
                effectiveness_data.append((lr_changes, final_loss))
        
        if effectiveness_data:
            lr_changes_list, final_losses_list = zip(*effectiveness_data)
            ax4.scatter(lr_changes_list, final_losses_list, 
                       color=self.colors['greedy'], alpha=0.6, s=30)
            
            # Add trendline
            z = np.polyfit(lr_changes_list, np.log(final_losses_list), 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(lr_changes_list), max(lr_changes_list), 100)
            ax4.plot(x_trend, np.exp(p(x_trend)), "r--", alpha=0.8, linewidth=2)
            
            ax4.set_xlabel('Number of LR Adaptations')
            ax4.set_ylabel('Final Loss')
            ax4.set_title('D. Adaptation Count vs Performance', fontweight='bold')
            ax4.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure_3_lr_adaptation.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'figure_3_lr_adaptation.pdf', bbox_inches='tight')
        plt.show()
        print(f"✅ Figure 3 saved to {self.output_dir}")
    
    def plot_4_architecture_heatmap_clean(self):
        """Figure 4: Clean Architecture Performance Heatmap"""
        print("📊 Creating Figure 4: Architecture Performance Heatmap")
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Create performance matrix - focus on log ratios for better interpretation
        perf_matrix = self.df.groupby(['model_type', 'scheduler_type'])['final_loss'].mean().unstack()
        
        # Calculate GreedyLR advantage ratios
        if 'greedy' in perf_matrix.columns:
            advantage_matrix = perf_matrix.div(perf_matrix['greedy'], axis=0)
            
            # Create custom colormap: green for GreedyLR wins, red for losses
            from matplotlib.colors import TwoSlopeNorm
            vmin, vmax = advantage_matrix.min().min(), advantage_matrix.max().max()
            norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
            
            im = ax.imshow(advantage_matrix.values, cmap='RdYlGn_r', norm=norm, aspect='auto')
            
            # Set labels
            ax.set_xticks(np.arange(len(advantage_matrix.columns)))
            ax.set_yticks(np.arange(len(advantage_matrix.index)))
            ax.set_xticklabels([col.replace('_', ' ').title() for col in advantage_matrix.columns], 
                              fontweight='bold')
            ax.set_yticklabels([idx.replace('neural_', '').replace('_', ' ').title() 
                               for idx in advantage_matrix.index], fontweight='bold')
            
            # Add text annotations with performance ratios
            for i in range(len(advantage_matrix.index)):
                for j in range(len(advantage_matrix.columns)):
                    value = advantage_matrix.iloc[i, j]
                    if not pd.isna(value):
                        # Format text based on value
                        if value > 1.1:
                            text_color = 'white'
                            display_text = f'{value:.1f}×\\nworse'
                        elif value < 0.9:
                            text_color = 'black'
                            display_text = f'{1/value:.1f}×\\nbetter'
                        else:
                            text_color = 'black'
                            display_text = f'{value:.2f}\\n≈equal'
                        
                        ax.text(j, i, display_text, ha="center", va="center", 
                               color=text_color, fontweight='bold', fontsize=9)
            
            # Highlight GreedyLR column
            greedy_col = list(advantage_matrix.columns).index('greedy')
            ax.axvline(x=greedy_col - 0.5, color='black', linewidth=3, alpha=0.8)
            ax.axvline(x=greedy_col + 0.5, color='black', linewidth=3, alpha=0.8)
            
            ax.set_title('Architecture-Specific Scheduler Performance\\nRelative to GreedyLR (Green = GreedyLR Wins, Red = GreedyLR Loses)', 
                        fontweight='bold', fontsize=14, pad=20)
            ax.set_xlabel('Learning Rate Scheduler', fontweight='bold', fontsize=12)
            ax.set_ylabel('Model Architecture', fontweight='bold', fontsize=12)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20)
            cbar.set_label('Performance Ratio vs GreedyLR\\n(<1 = GreedyLR Better, >1 = Competitor Better)', 
                          rotation=270, labelpad=25, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure_4_architecture_heatmap.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'figure_4_architecture_heatmap.pdf', bbox_inches='tight')
        plt.show()
        print(f"✅ Figure 4 saved to {self.output_dir}")
    
    def plot_5_statistical_summary(self):
        """Figure 5: Statistical Summary and Significance Tests"""
        print("📊 Creating Figure 5: Statistical Summary")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Statistical Analysis: GreedyLR Performance Validation', 
                     fontsize=16, fontweight='bold', y=0.95)
        
        # Subplot A: Effect sizes
        ax1 = axes[0, 0]
        
        effect_sizes = []
        scheduler_pairs = []
        p_values = []
        
        greedy_data = self.df[self.df['scheduler_type'] == 'greedy']['final_loss']
        
        for scheduler in ['cosine', 'cosine_restarts', 'exponential']:
            if scheduler in self.df['scheduler_type'].values:
                comp_data = self.df[self.df['scheduler_type'] == scheduler]['final_loss']
                
                if len(greedy_data) > 1 and len(comp_data) > 1:
                    # Calculate Cohen's d
                    pooled_std = np.sqrt((greedy_data.var() + comp_data.var()) / 2)
                    cohens_d = (greedy_data.mean() - comp_data.mean()) / pooled_std
                    
                    # Statistical test
                    _, p_val = stats.ttest_ind(greedy_data, comp_data)
                    
                    effect_sizes.append(cohens_d)
                    scheduler_pairs.append(f'GreedyLR vs\\n{scheduler.replace("_", " ").title()}')
                    p_values.append(p_val)
        
        colors_effect = [self.colors['improvement'] if es < 0 else self.colors['degradation'] 
                        for es in effect_sizes]
        
        bars = ax1.bar(range(len(effect_sizes)), effect_sizes, 
                      color=colors_effect, alpha=0.8, edgecolor='black')
        ax1.set_xticks(range(len(effect_sizes)))
        ax1.set_xticklabels(scheduler_pairs)
        ax1.set_ylabel('Effect Size (Cohen\'s d)', fontweight='bold')
        ax1.set_title('A. Effect Sizes for GreedyLR Comparisons', fontweight='bold')
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax1.axhline(y=-0.2, color='green', linestyle='--', alpha=0.5, label='Small Effect')
        ax1.axhline(y=-0.5, color='orange', linestyle='--', alpha=0.5, label='Medium Effect')
        ax1.axhline(y=-0.8, color='red', linestyle='--', alpha=0.5, label='Large Effect')
        ax1.legend()
        
        # Add significance stars
        for i, (bar, p_val) in enumerate(zip(bars, p_values)):
            height = bar.get_height()
            stars = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
            ax1.text(bar.get_x() + bar.get_width()/2., height - 0.1 if height < 0 else height + 0.1,
                    stars, ha='center', va='center' if height < 0 else 'bottom', 
                    fontweight='bold', fontsize=14, color='red')
        
        # Subplot B: Distribution comparison
        ax2 = axes[0, 1]
        
        data_to_plot = []
        labels_to_plot = []
        colors_to_plot = []
        
        for scheduler in ['greedy', 'cosine', 'cosine_restarts', 'exponential']:
            if scheduler in self.df['scheduler_type'].values:
                scheduler_data = self.df[self.df['scheduler_type'] == scheduler]['final_loss']
                # Apply log transform for better visualization
                log_data = np.log10(scheduler_data + 1e-10)
                data_to_plot.append(log_data)
                labels_to_plot.append(scheduler.replace('_', ' ').title())
                colors_to_plot.append(self.colors.get(scheduler, self.colors['neutral']))
        
        bp = ax2.boxplot(data_to_plot, labels=labels_to_plot, patch_artist=True)
        
        for patch, color in zip(bp['boxes'], colors_to_plot):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Highlight GreedyLR box
        if 'greedy' in [s for s in self.df['scheduler_type'].unique()]:
            greedy_idx = labels_to_plot.index('Greedy') if 'Greedy' in labels_to_plot else -1
            if greedy_idx != -1:
                bp['boxes'][greedy_idx].set_edgecolor('black')
                bp['boxes'][greedy_idx].set_linewidth(3)
        
        ax2.set_ylabel('Log₁₀(Final Loss)', fontweight='bold')
        ax2.set_title('B. Loss Distribution Comparison', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        
        # Subplot C: Win rate summary
        ax3 = axes[1, 0]
        
        # Calculate win rates for GreedyLR across different conditions
        conditions = ['Overall', 'No Noise', 'With Noise', 'Neural Networks', 'Analytical Functions']
        win_rates = []
        
        # Overall win rate
        overall_wins = 0
        total_comparisons = 0
        
        for arch in self.df['model_type'].unique():
            for noise in self.df['noise_type'].unique():
                subset = self.df[(self.df['model_type'] == arch) & (self.df['noise_type'] == noise)]
                if len(subset) > 0:
                    perf = subset.groupby('scheduler_type')['final_loss'].mean()
                    if 'greedy' in perf.index and len(perf) > 1:
                        total_comparisons += 1
                        if perf['greedy'] == perf.min():
                            overall_wins += 1
        
        win_rates.append(overall_wins / total_comparisons if total_comparisons > 0 else 0)
        
        # No noise win rate  
        no_noise_subset = self.df[self.df['noise_type'] == 'none']
        no_noise_wins = 0
        no_noise_total = 0
        
        for arch in no_noise_subset['model_type'].unique():
            arch_data = no_noise_subset[no_noise_subset['model_type'] == arch]
            if len(arch_data) > 0:
                perf = arch_data.groupby('scheduler_type')['final_loss'].mean()
                if 'greedy' in perf.index and len(perf) > 1:
                    no_noise_total += 1
                    if perf['greedy'] == perf.min():
                        no_noise_wins += 1
        
        win_rates.append(no_noise_wins / no_noise_total if no_noise_total > 0 else 0)
        
        # With noise win rate
        noisy_subset = self.df[self.df['noise_type'] != 'none']
        noisy_wins = 0
        noisy_total = 0
        
        for arch in noisy_subset['model_type'].unique():
            for noise in noisy_subset['noise_type'].unique():
                subset = noisy_subset[(noisy_subset['model_type'] == arch) & (noisy_subset['noise_type'] == noise)]
                if len(subset) > 0:
                    perf = subset.groupby('scheduler_type')['final_loss'].mean()
                    if 'greedy' in perf.index and len(perf) > 1:
                        noisy_total += 1
                        if perf['greedy'] == perf.min():
                            noisy_wins += 1
        
        win_rates.append(noisy_wins / noisy_total if noisy_total > 0 else 0)
        
        # Neural vs Analytical (simplified)
        neural_archs = [arch for arch in self.df['model_type'].unique() if 'neural' in arch]
        analytical_archs = [arch for arch in self.df['model_type'].unique() if 'neural' not in arch]
        
        # Neural networks win rate
        neural_wins = 0
        neural_total = 0
        for arch in neural_archs:
            arch_data = self.df[self.df['model_type'] == arch]  
            if len(arch_data) > 0:
                perf = arch_data.groupby('scheduler_type')['final_loss'].mean()
                if 'greedy' in perf.index and len(perf) > 1:
                    neural_total += 1
                    if perf['greedy'] == perf.min():
                        neural_wins += 1
        
        win_rates.append(neural_wins / neural_total if neural_total > 0 else 0)
        
        # Analytical functions win rate
        analytical_wins = 0
        analytical_total = 0
        for arch in analytical_archs:
            arch_data = self.df[self.df['model_type'] == arch]
            if len(arch_data) > 0:
                perf = arch_data.groupby('scheduler_type')['final_loss'].mean()
                if 'greedy' in perf.index and len(perf) > 1:
                    analytical_total += 1
                    if perf['greedy'] == perf.min():
                        analytical_wins += 1
        
        win_rates.append(analytical_wins / analytical_total if analytical_total > 0 else 0)
        
        colors_win = [self.colors['improvement'] if wr > 0.5 else self.colors['degradation'] 
                     for wr in win_rates]
        
        bars = ax3.bar(range(len(conditions)), win_rates, 
                      color=colors_win, alpha=0.8, edgecolor='black')
        ax3.set_xticks(range(len(conditions)))
        ax3.set_xticklabels(conditions, rotation=45, ha='right')
        ax3.set_ylabel('GreedyLR Win Rate', fontweight='bold')
        ax3.set_title('C. GreedyLR Win Rate by Condition', fontweight='bold')
        ax3.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Random Performance')
        ax3.set_ylim(0, 1)
        ax3.legend()
        
        # Add percentage labels
        for bar, wr in zip(bars, win_rates):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{wr:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # Subplot D: Sample size and power analysis
        ax4 = axes[1, 1]
        
        sample_sizes = []
        schedulers_sample = []
        
        for scheduler in self.df['scheduler_type'].unique():
            n_samples = len(self.df[self.df['scheduler_type'] == scheduler])
            sample_sizes.append(n_samples)
            schedulers_sample.append(scheduler.replace('_', ' ').title())
        
        bars = ax4.bar(range(len(schedulers_sample)), sample_sizes,
                      color=[self.colors.get(s.lower().replace(' ', '_'), self.colors['neutral']) 
                            for s in schedulers_sample], alpha=0.8, edgecolor='black')
        ax4.set_xticks(range(len(schedulers_sample)))
        ax4.set_xticklabels(schedulers_sample, rotation=45, ha='right')
        ax4.set_ylabel('Number of Experiments', fontweight='bold')
        ax4.set_title('D. Sample Sizes by Scheduler', fontweight='bold')
        
        # Add value labels
        for bar, size in zip(bars, sample_sizes):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 50,
                    f'{size:,}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure_5_statistical_summary.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'figure_5_statistical_summary.pdf', bbox_inches='tight')
        plt.show()
        print(f"✅ Figure 5 saved to {self.output_dir}")
    
    def generate_all_figures(self):
        """Generate all journal-quality figures"""
        print("🚀 Generating all journal-quality figures...")
        print("="*60)
        
        self.plot_1_overall_performance_comparison()
        self.plot_2_noise_robustness_showcase()
        self.plot_3_learning_rate_adaptation()
        self.plot_4_architecture_heatmap_clean()
        self.plot_5_statistical_summary()
        
        print("="*60)
        print(f"🎉 All journal-quality figures generated!")
        print(f"📁 Saved to: {self.output_dir}")
        print(f"📊 Total files: {len(list(self.output_dir.glob('*')))}")
        print("✅ Ready for publication!")

def main():
    plotter = JournalQualityPlots('robust_results.json')
    plotter.generate_all_figures()

if __name__ == "__main__":
    main()