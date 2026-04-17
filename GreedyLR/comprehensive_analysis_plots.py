#!/usr/bin/env python3
"""
Comprehensive Analysis and Plotting System for Scheduler Comparison
- Decoupled from experiment runner
- Rich parameter analysis plots for paper discussion
- LR trajectory comparisons with median overlays
- Multiple plot types for paper figures
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Set plotting style for publication quality
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
    'font.family': 'serif',
    'axes.grid': True,
    'grid.alpha': 0.3
})

class ComprehensiveAnalyzer:
    def __init__(self, results_file: str = 'robust_results.json'):
        self.results_file = results_file
        self.df = None
        self.output_dir = Path('analysis_plots')
        self.output_dir.mkdir(exist_ok=True)
        
    def load_results(self):
        """Load results from JSON file"""
        try:
            with open(self.results_file, 'r') as f:
                results = json.load(f)
            
            self.df = pd.DataFrame(results)
            print(f"✅ Loaded {len(self.df)} experimental results")
            print(f"📊 Scheduler types: {list(self.df['scheduler_type'].unique())}")
            print(f"🧠 Model types: {len(self.df['model_type'].unique())}")
            print(f"🌊 Noise types: {len(self.df['noise_type'].unique())}")
            
            # Extract scheduler parameters into separate columns
            self._extract_scheduler_params()
            
            return True
            
        except FileNotFoundError:
            print(f"❌ Results file {self.results_file} not found")
            return False
        except Exception as e:
            print(f"❌ Error loading results: {e}")
            return False
    
    def _extract_scheduler_params(self):
        """Extract scheduler parameters into separate columns for analysis"""
        if 'scheduler_params' in self.df.columns:
            # Extract GreedyLR parameters
            greedy_mask = self.df['scheduler_type'] == 'greedy'
            if greedy_mask.any():
                self.df.loc[greedy_mask, 'factor'] = self.df.loc[greedy_mask, 'scheduler_params'].apply(
                    lambda x: x.get('factor', np.nan) if isinstance(x, dict) else np.nan)
                self.df.loc[greedy_mask, 'min_lr'] = self.df.loc[greedy_mask, 'scheduler_params'].apply(
                    lambda x: x.get('min_lr', np.nan) if isinstance(x, dict) else np.nan)
                self.df.loc[greedy_mask, 'patience'] = self.df.loc[greedy_mask, 'scheduler_params'].apply(
                    lambda x: x.get('patience', np.nan) if isinstance(x, dict) else np.nan)
            
            # Extract Cosine parameters
            cosine_mask = self.df['scheduler_type'] == 'cosine'
            if cosine_mask.any():
                self.df.loc[cosine_mask, 'T_max'] = self.df.loc[cosine_mask, 'scheduler_params'].apply(
                    lambda x: x.get('T_max', np.nan) if isinstance(x, dict) else np.nan)
                self.df.loc[cosine_mask, 'eta_min'] = self.df.loc[cosine_mask, 'scheduler_params'].apply(
                    lambda x: x.get('eta_min', np.nan) if isinstance(x, dict) else np.nan)
    
    def plot_parameter_effects(self):
        """Plot effects of different parameters for each scheduler type"""
        print("📊 Creating parameter effect plots...")
        
        # GreedyLR Parameter Effects
        self._plot_greedy_parameter_effects()
        
        # Cosine Parameter Effects  
        self._plot_cosine_parameter_effects()
        
        # Cross-scheduler parameter comparison
        self._plot_cross_scheduler_comparison()
    
    def _plot_greedy_parameter_effects(self):
        """Detailed analysis of GreedyLR parameter effects"""
        greedy_df = self.df[self.df['scheduler_type'] == 'greedy'].copy()
        
        if len(greedy_df) == 0:
            print("⚠️ No GreedyLR results found")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('GreedyLR Parameter Effects Analysis', fontsize=20, fontweight='bold')
        
        # Factor effect on final loss
        if 'factor' in greedy_df.columns:
            sns.boxplot(data=greedy_df, x='factor', y='final_loss', ax=axes[0,0])
            axes[0,0].set_title('Factor vs Final Loss')
            axes[0,0].set_ylabel('Final Loss (log scale)')
            axes[0,0].set_yscale('log')
        
        # Min_LR effect on final loss
        if 'min_lr' in greedy_df.columns:
            sns.boxplot(data=greedy_df, x='min_lr', y='final_loss', ax=axes[0,1])
            axes[0,1].set_title('Min LR vs Final Loss')
            axes[0,1].set_ylabel('Final Loss (log scale)')
            axes[0,1].set_yscale('log')
            axes[0,1].tick_params(axis='x', rotation=45)
        
        # Patience effect on convergence rate
        if 'patience' in greedy_df.columns:
            sns.boxplot(data=greedy_df, x='patience', y='convergence_rate_50', ax=axes[0,2])
            axes[0,2].set_title('Patience vs Convergence Rate')
            axes[0,2].set_ylabel('Convergence Rate')
        
        # Factor vs Noise Type interaction
        if 'factor' in greedy_df.columns:
            pivot_data = greedy_df.pivot_table(values='final_loss', index='noise_type', 
                                              columns='factor', aggfunc='median')
            sns.heatmap(pivot_data, annot=True, fmt='.3f', ax=axes[1,0], cmap='viridis_r')
            axes[1,0].set_title('Factor × Noise Type (Median Final Loss)')
        
        # Parameter correlation heatmap
        param_cols = ['factor', 'min_lr', 'patience', 'final_loss', 'convergence_rate_50']
        available_cols = [col for col in param_cols if col in greedy_df.columns]
        if len(available_cols) > 2:
            corr_matrix = greedy_df[available_cols].corr()
            sns.heatmap(corr_matrix, annot=True, center=0, ax=axes[1,1], cmap='RdBu_r')
            axes[1,1].set_title('Parameter Correlations')
        
        # Best parameter combinations
        if all(col in greedy_df.columns for col in ['factor', 'min_lr', 'patience']):
            best_combos = greedy_df.nsmallest(20, 'final_loss')[['factor', 'min_lr', 'patience', 'final_loss']]
            axes[1,2].scatter(best_combos['factor'], best_combos['final_loss'], 
                            s=100, alpha=0.7, c=best_combos['patience'], cmap='plasma')
            axes[1,2].set_title('Best Parameter Combinations')
            axes[1,2].set_xlabel('Factor')
            axes[1,2].set_ylabel('Final Loss (log scale)')
            axes[1,2].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'greedy_parameter_effects.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ GreedyLR parameter effects plot saved")
    
    def _plot_cosine_parameter_effects(self):
        """Analysis of Cosine scheduler parameter effects"""
        cosine_df = self.df[self.df['scheduler_type'].isin(['cosine', 'cosine_restarts'])].copy()
        
        if len(cosine_df) == 0:
            print("⚠️ No Cosine results found")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Cosine Scheduler Parameter Effects', fontsize=18, fontweight='bold')
        
        # T_max effect for regular cosine
        regular_cosine = cosine_df[cosine_df['scheduler_type'] == 'cosine']
        if len(regular_cosine) > 0 and 'T_max' in regular_cosine.columns:
            sns.boxplot(data=regular_cosine, x='T_max', y='final_loss', ax=axes[0,0])
            axes[0,0].set_title('T_max vs Final Loss (Cosine)')
            axes[0,0].set_yscale('log')
        
        # eta_min effect
        if 'eta_min' in cosine_df.columns:
            sns.boxplot(data=cosine_df, x='eta_min', y='final_loss', hue='scheduler_type', ax=axes[0,1])
            axes[0,1].set_title('Eta_min vs Final Loss')
            axes[0,1].set_yscale('log')
            axes[0,1].tick_params(axis='x', rotation=45)
        
        # Cosine vs Cosine with Restarts
        sns.boxplot(data=cosine_df, x='scheduler_type', y='convergence_rate_50', ax=axes[1,0])
        axes[1,0].set_title('Convergence Rate Comparison')
        
        # Performance across noise types
        pivot_noise = cosine_df.pivot_table(values='final_loss', index='noise_type', 
                                           columns='scheduler_type', aggfunc='median')
        sns.heatmap(pivot_noise, annot=True, fmt='.3f', ax=axes[1,1], cmap='viridis_r')
        axes[1,1].set_title('Performance × Noise Type')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cosine_parameter_effects.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Cosine parameter effects plot saved")
    
    def _plot_cross_scheduler_comparison(self):
        """Cross-scheduler parameter comparison plots"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Cross-Scheduler Performance Analysis', fontsize=20, fontweight='bold')
        
        # Overall performance comparison
        sns.boxplot(data=self.df, x='scheduler_type', y='final_loss', ax=axes[0,0])
        axes[0,0].set_title('Final Loss by Scheduler Type')
        axes[0,0].set_yscale('log')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Convergence rate comparison
        sns.boxplot(data=self.df, x='scheduler_type', y='convergence_rate_50', ax=axes[0,1])
        axes[0,1].set_title('Convergence Rate by Scheduler')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Performance across model types
        pivot_model = self.df.pivot_table(values='final_loss', index='model_type', 
                                         columns='scheduler_type', aggfunc='median')
        sns.heatmap(pivot_model, annot=True, fmt='.3f', ax=axes[0,2], cmap='viridis_r')
        axes[0,2].set_title('Performance × Model Type')
        
        # Noise robustness
        pivot_noise = self.df.pivot_table(values='final_loss', index='noise_type', 
                                         columns='scheduler_type', aggfunc='median')
        sns.heatmap(pivot_noise, annot=True, fmt='.3f', ax=axes[1,0], cmap='viridis_r')
        axes[1,0].set_title('Noise Robustness')
        
        # Stability comparison
        if 'stability_score' in self.df.columns:
            sns.boxplot(data=self.df, x='scheduler_type', y='stability_score', ax=axes[1,1])
            axes[1,1].set_title('Stability Score by Scheduler')
            axes[1,1].tick_params(axis='x', rotation=45)
        
        # Success rate by noise strength
        success_rate = self.df.groupby(['scheduler_type', 'noise_strength'])['converged'].mean().reset_index()
        for scheduler in success_rate['scheduler_type'].unique():
            sched_data = success_rate[success_rate['scheduler_type'] == scheduler]
            axes[1,2].plot(sched_data['noise_strength'], sched_data['converged'], 
                          marker='o', label=scheduler, linewidth=2)
        axes[1,2].set_title('Success Rate vs Noise Strength')
        axes[1,2].set_xlabel('Noise Strength')
        axes[1,2].set_ylabel('Success Rate')
        axes[1,2].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cross_scheduler_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Cross-scheduler comparison plot saved")
    
    def plot_lr_trajectories(self):
        """Plot LR trajectories with faint individual traces and bold median"""
        print("📈 Creating LR trajectory plots...")
        
        # Check if lr_history is available
        if 'lr_history' not in self.df.columns:
            print("⚠️ LR history not available in results")
            return
        
        schedulers = self.df['scheduler_type'].unique()
        n_schedulers = len(schedulers)
        
        fig, axes = plt.subplots(2, (n_schedulers + 1) // 2, figsize=(6 * n_schedulers, 10))
        if n_schedulers == 1:
            axes = [axes]
        elif n_schedulers <= 2:
            axes = axes.flatten()
        
        fig.suptitle('Learning Rate Trajectories by Scheduler Type', fontsize=20, fontweight='bold')
        
        for i, scheduler_type in enumerate(schedulers):
            ax = axes[i] if n_schedulers > 1 else axes[0]
            
            scheduler_data = self.df[self.df['scheduler_type'] == scheduler_type]
            
            # Collect all LR histories
            lr_histories = []
            for _, row in scheduler_data.iterrows():
                if 'lr_history' in row and row['lr_history'] is not None:
                    try:
                        lr_hist = row['lr_history'] if isinstance(row['lr_history'], list) else json.loads(row['lr_history'])
                        if len(lr_hist) > 0:
                            lr_histories.append(lr_hist)
                    except:
                        continue
            
            if not lr_histories:
                ax.text(0.5, 0.5, f'No LR history\nfor {scheduler_type}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{scheduler_type.title()} Scheduler')
                continue
            
            # Pad histories to same length
            max_len = max(len(hist) for hist in lr_histories)
            padded_histories = []
            
            for hist in lr_histories:
                if len(hist) < max_len:
                    # Pad with last value
                    padded = hist + [hist[-1]] * (max_len - len(hist))
                else:
                    padded = hist[:max_len]
                padded_histories.append(padded)
            
            # Convert to numpy array
            lr_array = np.array(padded_histories)
            steps = np.arange(max_len)
            
            # Plot individual trajectories in faint color
            color = plt.cm.tab10(i)
            for trajectory in lr_array:
                ax.plot(steps, trajectory, color=color, alpha=0.1, linewidth=0.5)
            
            # Plot median trajectory in bold
            median_trajectory = np.median(lr_array, axis=0)
            ax.plot(steps, median_trajectory, color=color, linewidth=3, 
                   label=f'{scheduler_type.title()} (median)', zorder=10)
            
            # Plot quartiles
            q25 = np.percentile(lr_array, 25, axis=0)
            q75 = np.percentile(lr_array, 75, axis=0)
            ax.fill_between(steps, q25, q75, color=color, alpha=0.2)
            
            ax.set_title(f'{scheduler_type.title()} Scheduler\n({len(lr_histories)} trajectories)')
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Learning Rate')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Hide extra subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'lr_trajectories.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ LR trajectory plots saved")
    
    def generate_comprehensive_report(self):
        """Generate comprehensive analysis report with all plots"""
        print("📝 Generating comprehensive analysis report...")
        
        if not self.load_results():
            return
        
        # Generate all plot types
        self.plot_parameter_effects()
        self.plot_lr_trajectories() 
        self._create_summary_tables()
        
        # Generate markdown report
        self._generate_markdown_report()
        
        print(f"✅ Comprehensive analysis complete! Check {self.output_dir}/")
    
    def _create_summary_tables(self):
        """Create summary tables for paper"""
        print("📊 Creating summary tables...")
        
        # EXPERIMENT COUNT BREAKDOWN TABLE
        experiment_breakdown = self._create_experiment_count_table()
        experiment_breakdown.to_csv(self.output_dir / 'experiment_count_breakdown.csv')
        
        # Overall performance table
        perf_summary = self.df.groupby('scheduler_type').agg({
            'final_loss': ['mean', 'std', 'median', 'min'],
            'convergence_rate_50': ['mean', 'std', 'median', 'max'],
            'stability_score': ['mean', 'std'] if 'stability_score' in self.df.columns else ['count'],
        }).round(4)
        
        perf_summary.to_csv(self.output_dir / 'performance_summary.csv')
        
        # Parameter effect tables for GreedyLR
        greedy_df = self.df[self.df['scheduler_type'] == 'greedy']
        if len(greedy_df) > 0 and 'factor' in greedy_df.columns:
            factor_effect = greedy_df.groupby('factor').agg({
                'final_loss': ['mean', 'std', 'count'],
                'convergence_rate_50': ['mean', 'std']
            }).round(4)
            factor_effect.to_csv(self.output_dir / 'factor_effect_table.csv')
            
            # Detailed parameter combination table
            param_combo_table = self._create_parameter_combination_table()
            param_combo_table.to_csv(self.output_dir / 'parameter_combinations.csv')
        
        print("✅ Summary tables saved")
    
    def _create_experiment_count_table(self):
        """Create detailed experiment count breakdown table"""
        # Total experiments by scheduler
        scheduler_counts = self.df['scheduler_type'].value_counts().sort_index()
        
        # Cross-tabulation of scheduler × model type
        scheduler_model_counts = pd.crosstab(self.df['scheduler_type'], self.df['model_type'], margins=True)
        
        # Cross-tabulation of scheduler × noise type  
        scheduler_noise_counts = pd.crosstab(self.df['scheduler_type'], self.df['noise_type'], margins=True)
        
        # Parameter variation counts for GreedyLR
        greedy_df = self.df[self.df['scheduler_type'] == 'greedy']
        param_counts = {}
        if len(greedy_df) > 0:
            if 'factor' in greedy_df.columns:
                param_counts['factor_variations'] = len(greedy_df['factor'].unique())
            if 'min_lr' in greedy_df.columns:
                param_counts['min_lr_variations'] = len(greedy_df['min_lr'].unique())
            if 'patience' in greedy_df.columns:
                param_counts['patience_variations'] = len(greedy_df['patience'].unique())
        
        # Create summary dataframe
        summary_data = {
            'Total_Experiments': [len(self.df)],
            'Scheduler_Types': [len(self.df['scheduler_type'].unique())],
            'Model_Types': [len(self.df['model_type'].unique())],
            'Noise_Types': [len(self.df['noise_type'].unique())],
            'GreedyLR_Count': [scheduler_counts.get('greedy', 0)],
            'Cosine_Count': [scheduler_counts.get('cosine', 0)], 
            'Cosine_Restarts_Count': [scheduler_counts.get('cosine_restarts', 0)],
            'Exponential_Count': [scheduler_counts.get('exponential', 0)],
            'GreedyLR_Percentage': [scheduler_counts.get('greedy', 0) / len(self.df) * 100],
            'Others_Percentage': [(len(self.df) - scheduler_counts.get('greedy', 0)) / len(self.df) * 100],
        }
        
        # Add parameter variation counts
        for param_name, count in param_counts.items():
            summary_data[f'GreedyLR_{param_name}'] = [count]
        
        return pd.DataFrame(summary_data)
    
    def _create_parameter_combination_table(self):
        """Create detailed parameter combination analysis table"""
        greedy_df = self.df[self.df['scheduler_type'] == 'greedy']
        if len(greedy_df) == 0:
            return pd.DataFrame()
        
        param_cols = ['factor', 'min_lr', 'patience']
        available_params = [col for col in param_cols if col in greedy_df.columns]
        
        if not available_params:
            return pd.DataFrame()
        
        # Group by parameter combinations and calculate statistics
        combo_stats = greedy_df.groupby(available_params).agg({
            'final_loss': ['count', 'mean', 'std', 'min'],
            'convergence_rate_50': ['mean', 'std'],
        }).round(4)
        
        # Flatten column names
        combo_stats.columns = [f'{col[0]}_{col[1]}' for col in combo_stats.columns]
        
        return combo_stats.reset_index()
    
    def _get_experiment_calculation_summary(self):
        """Generate experiment count calculation breakdown"""
        scheduler_counts = self.df['scheduler_type'].value_counts().sort_index()
        total = len(self.df)
        
        greedy_count = scheduler_counts.get('greedy', 0)
        others_count = total - greedy_count
        
        # Parameter variation analysis for GreedyLR
        greedy_df = self.df[self.df['scheduler_type'] == 'greedy']
        param_info = ""
        if len(greedy_df) > 0:
            if 'factor' in greedy_df.columns:
                factors = sorted(greedy_df['factor'].unique())
                param_info += f"- **Factor values tested**: {factors} ({len(factors)} variations)\n"
            if 'min_lr' in greedy_df.columns:
                min_lrs = sorted(greedy_df['min_lr'].unique())
                param_info += f"- **Min LR values tested**: {min_lrs} ({len(min_lrs)} variations)\n"
            if 'patience' in greedy_df.columns:
                patience_vals = sorted(greedy_df['patience'].unique())
                param_info += f"- **Patience values tested**: {patience_vals} ({len(patience_vals)} variations)\n"
        
        summary = f"""
**Total Experiments Executed**: {total:,}

**Scheduler Distribution**:
- GreedyLR: {greedy_count:,} ({greedy_count/total*100:.1f}%)
- Other Schedulers: {others_count:,} ({others_count/total*100:.1f}%)
  - Cosine: {scheduler_counts.get('cosine', 0):,}
  - Cosine with Restarts: {scheduler_counts.get('cosine_restarts', 0):,}
  - Exponential: {scheduler_counts.get('exponential', 0):,}

**GreedyLR Parameter Sweep Details**:
{param_info}

**Model Architecture Coverage**: {len(self.df['model_type'].unique())} types
**Noise Condition Coverage**: {len(self.df['noise_type'].unique())} types
**Problem Variants**: {len(self.df['problem_variant'].unique()) if 'problem_variant' in self.df.columns else 'N/A'}
"""
        return summary
    
    def _generate_markdown_report(self):
        """Generate markdown report for paper discussion"""
        report_content = f"""# Comprehensive Scheduler Analysis Report

## Dataset Overview
- **Total Experiments**: {len(self.df):,}
- **Scheduler Types**: {', '.join(self.df['scheduler_type'].unique())}
- **Model Types**: {len(self.df['model_type'].unique())}
- **Noise Conditions**: {len(self.df['noise_type'].unique())}

## Generated Figures for Paper

### 1. Parameter Effects Analysis
- `greedy_parameter_effects.png` - Detailed GreedyLR parameter analysis
- `cosine_parameter_effects.png` - Cosine scheduler parameter effects
- `cross_scheduler_comparison.png` - Cross-scheduler performance comparison

### 2. Learning Rate Trajectories  
- `lr_trajectories.png` - Individual trajectories (faint) with median overlay (bold)

### 3. Summary Tables
- `experiment_count_breakdown.csv` - Detailed experiment count and distribution analysis
- `performance_summary.csv` - Overall performance metrics
- `factor_effect_table.csv` - GreedyLR factor effect analysis
- `parameter_combinations.csv` - Detailed parameter combination statistics

## Experiment Configuration Summary

### Total Experiment Count Calculation
{self._get_experiment_calculation_summary()}

## Key Findings Summary

### GreedyLR Parameter Effects
{self._get_greedy_parameter_insights()}

### Cross-Scheduler Performance
{self._get_cross_scheduler_insights()}

## Usage for Paper
All figures are publication-ready (300 DPI) and can be directly included in your paper's discussion section.
"""
        
        with open(self.output_dir / 'analysis_report.md', 'w') as f:
            f.write(report_content)
        
        print("✅ Markdown report generated")
    
    def _get_greedy_parameter_insights(self):
        """Generate insights about GreedyLR parameters"""
        greedy_df = self.df[self.df['scheduler_type'] == 'greedy']
        if len(greedy_df) == 0 or 'factor' not in greedy_df.columns:
            return "No GreedyLR parameter data available."
        
        # Best factor
        best_factor = greedy_df.groupby('factor')['final_loss'].mean().idxmin()
        
        insights = f"""
- **Optimal Factor**: {best_factor} showed best average performance
- **Factor Range**: Tested factors from 0.5 to 0.9 as requested
- **Total Parameter Combinations**: {len(greedy_df)} experiments across all parameter combinations
"""
        return insights
    
    def _get_cross_scheduler_insights(self):
        """Generate cross-scheduler insights"""
        best_scheduler = self.df.groupby('scheduler_type')['final_loss'].mean().idxmin()
        
        insights = f"""
- **Best Overall Scheduler**: {best_scheduler}
- **Noise Robustness**: Analysis shows performance across {len(self.df['noise_type'].unique())} noise conditions
- **Model Generalization**: Tested across {len(self.df['model_type'].unique())} different model architectures
"""
        return insights

def main():
    """Main function to run comprehensive analysis"""
    analyzer = ComprehensiveAnalyzer('robust_results.json')
    analyzer.generate_comprehensive_report()

if __name__ == "__main__":
    main()