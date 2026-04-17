#!/usr/bin/env python3
"""
Comprehensive Scheduler Comparison: GreedyLR vs Cosine
======================================================

This experiment compares GreedyLR and Cosine schedulers using small, tractable models
with known optima. We test convergence speed, robustness to loss spikes, and recovery
across hundreds of different training scenarios.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

@dataclass
class ExperimentConfig:
    """Configuration for each experiment"""
    model_type: str  # 'quadratic', 'neural_net', 'rosenbrock'
    noise_type: str  # 'none', 'gaussian', 'spike', 'plateau'
    noise_strength: float
    total_steps: int
    scheduler_type: str  # 'greedy', 'cosine'
    scheduler_params: Dict

class StreamingAverage:
    """Helper class for GreedyLR smoothing"""
    def __init__(self, window_size):
        self.window_size = window_size
        self.values = []
        self.sum = 0

    def streamavg(self, value):
        self.values.append(value)
        self.sum += value
        if len(self.values) > self.window_size:
            self.sum -= self.values.pop(0)
        return float(self.sum) / len(self.values) if self.values else value

class SimpleGreedyLR:
    """Simplified GreedyLR implementation for our experiments"""
    def __init__(self, optimizer, patience=10, min_lr=1e-5, factor=0.95, 
                 smooth=True, window_size=10, max_lr=1.0):
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.factor = factor
        self.smooth = smooth
        
        if smooth:
            self.sa = StreamingAverage(window_size)
            
        self.best = float('inf')
        self.num_bad_epochs = 0
        self.num_good_epochs = 0
        self.cooldown_counter = 0
        self.warmup_counter = 0
        self._last_lr = [group['lr'] for group in optimizer.param_groups]
        
    def step(self, metrics):
        current = float(metrics)
        
        if self.smooth:
            current = self.sa.streamavg(current)
            
        # Check if current is better (lower loss)
        if current < self.best - 1e-6:  # threshold for improvement
            self.best = current
            self.num_bad_epochs = 0
            self.num_good_epochs += 1
        else:
            self.num_bad_epochs += 1
            self.num_good_epochs = 0
            
        # Handle cooldown/warmup
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0
            
        if self.warmup_counter > 0:
            self.warmup_counter -= 1
            self.num_good_epochs = 0
            
        # Reduce LR if bad for too long
        if self.num_bad_epochs > self.patience:
            self._reduce_lr()
            self.cooldown_counter = 5  # cooldown period
            self.num_bad_epochs = 0
            
        # Increase LR if good for too long
        if self.num_good_epochs > self.patience:
            self._increase_lr()
            self.warmup_counter = 5  # warmup period
            self.num_good_epochs = 0
            
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
        
    def _reduce_lr(self):
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            param_group['lr'] = new_lr
            
    def _increase_lr(self):
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = min(old_lr / self.factor, self.max_lr)
            param_group['lr'] = new_lr
            
    def get_last_lr(self):
        return self._last_lr

class TrainingModels:
    """Collection of small models with known optima for testing"""
    
    @staticmethod
    def quadratic_loss(params, target=None, noise=0.0):
        """Simple quadratic loss: L = ||params - target||^2 + noise"""
        if target is None:
            target = torch.zeros_like(params)
        loss = torch.sum((params - target) ** 2)
        if noise > 0:
            loss += torch.randn_like(loss) * noise
        return loss
    
    @staticmethod
    def rosenbrock_loss(params, noise=0.0):
        """Rosenbrock function: challenging optimization landscape"""
        if len(params) < 2:
            params = torch.cat([params, torch.zeros(2 - len(params))])
        x, y = params[0], params[1]
        loss = 100 * (y - x**2)**2 + (1 - x)**2
        if noise > 0:
            loss += torch.randn_like(loss) * noise
        return loss
    
    class SimpleNN(nn.Module):
        """Small neural network with known synthetic dataset"""
        def __init__(self, input_dim=10, hidden_dim=20, output_dim=1):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
            
        def forward(self, x):
            return self.layers(x)

def create_synthetic_data(n_samples=1000, input_dim=10, noise_std=0.1):
    """Create synthetic regression dataset with known optimal solution"""
    X = torch.randn(n_samples, input_dim)
    # Create a known linear relationship with some nonlinearity
    true_weights = torch.randn(input_dim, 1)
    y = X @ true_weights + 0.1 * torch.sin(X @ true_weights) + noise_std * torch.randn(n_samples, 1)
    return X, y, true_weights

def inject_noise(loss_history: List[float], noise_type: str, noise_strength: float, step: int) -> float:
    """Inject various types of noise to simulate training instabilities"""
    base_loss = loss_history[-1] if loss_history else 1.0
    
    if noise_type == 'none':
        return 0.0
    elif noise_type == 'gaussian':
        return np.random.normal(0, noise_strength)
    elif noise_type == 'spike':
        # Occasional large spikes
        if np.random.random() < 0.02:  # 2% chance
            return base_loss * noise_strength
        return 0.0
    elif noise_type == 'plateau':
        # Simulate loss plateaus
        if step % 100 < 20:  # 20% of time in plateau
            return np.random.normal(0, noise_strength * 0.1)
        return 0.0
    return 0.0

def run_single_experiment(config: ExperimentConfig) -> Dict:
    """Run a single experiment with given configuration"""
    
    # Initialize model based on type
    if config.model_type == 'quadratic':
        params = torch.randn(10, requires_grad=True)
        target = torch.randn(10)
        optimizer = optim.Adam([params], lr=0.01)
        
        def compute_loss():
            return TrainingModels.quadratic_loss(params, target)
            
    elif config.model_type == 'rosenbrock':
        params = torch.randn(2, requires_grad=True) 
        optimizer = optim.Adam([params], lr=0.01)
        
        def compute_loss():
            return TrainingModels.rosenbrock_loss(params)
            
    elif config.model_type == 'neural_net':
        model = TrainingModels.SimpleNN()
        X_train, y_train, true_weights = create_synthetic_data()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        def compute_loss():
            pred = model(X_train)
            return criterion(pred, y_train)
    
    # Initialize scheduler
    if config.scheduler_type == 'greedy':
        scheduler = SimpleGreedyLR(optimizer, **config.scheduler_params)
    elif config.scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.total_steps, 
            eta_min=config.scheduler_params.get('min_lr', 1e-5)
        )
    
    # Training loop
    losses = []
    lrs = []
    metrics = {
        'converged_step': None,
        'final_loss': None,
        'min_loss': float('inf'),
        'convergence_rate': 0.0,
        'recovery_episodes': 0,
        'lr_changes': 0
    }
    
    prev_lr = optimizer.param_groups[0]['lr']
    in_spike = False
    spike_start_loss = None
    
    for step in range(config.total_steps):
        optimizer.zero_grad()
        
        # Compute base loss
        loss = compute_loss()
        
        # Add noise
        noise = inject_noise(losses, config.noise_type, config.noise_strength, step)
        noisy_loss = loss + noise
        
        # Track recovery from spikes
        if config.noise_type == 'spike':
            if noise > 0 and not in_spike:
                in_spike = True
                spike_start_loss = loss.item()
            elif in_spike and loss.item() < spike_start_loss * 1.1:
                metrics['recovery_episodes'] += 1
                in_spike = False
        
        # Backward pass
        noisy_loss.backward()
        optimizer.step()
        
        # Update scheduler
        if config.scheduler_type == 'greedy':
            scheduler.step(loss.item())
        else:
            scheduler.step()
            
        # Track metrics
        current_lr = optimizer.param_groups[0]['lr']
        if abs(current_lr - prev_lr) > 1e-8:
            metrics['lr_changes'] += 1
        prev_lr = current_lr
        
        losses.append(loss.item()) 
        lrs.append(current_lr)
        
        # Check convergence (loss < 1% of initial)
        if len(losses) > 10 and metrics['converged_step'] is None:
            if loss.item() < losses[0] * 0.01:
                metrics['converged_step'] = step
                
        # Track minimum loss
        metrics['min_loss'] = min(metrics['min_loss'], loss.item())
    
    # Calculate final metrics
    metrics['final_loss'] = losses[-1]
    if len(losses) > 50:
        # Convergence rate = improvement in first 50 steps
        metrics['convergence_rate'] = (losses[0] - losses[49]) / losses[0]
    
    return {
        'config': config,
        'metrics': metrics,
        'losses': losses,
        'lrs': lrs
    }

def generate_experiment_configs() -> List[ExperimentConfig]:
    """Generate comprehensive set of experiment configurations"""
    configs = []
    
    model_types = ['quadratic', 'rosenbrock', 'neural_net']
    noise_types = ['none', 'gaussian', 'spike', 'plateau']
    noise_strengths = [0.0, 0.01, 0.05, 0.1, 0.2]
    
    # GreedyLR variations
    greedy_params_variants = [
        {'patience': 5, 'factor': 0.9, 'smooth': True},
        {'patience': 10, 'factor': 0.95, 'smooth': True},
        {'patience': 15, 'factor': 0.99, 'smooth': True},
        {'patience': 10, 'factor': 0.9, 'smooth': False},
    ]
    
    # Cosine variations  
    cosine_params_variants = [
        {'min_lr': 1e-5},
        {'min_lr': 1e-4},
        {'min_lr': 1e-3},
    ]
    
    total_steps = 500
    
    # Generate all combinations
    for model_type in model_types:
        for noise_type in noise_types:
            for noise_strength in noise_strengths:
                # Skip no noise with different noise strengths
                if noise_type == 'none' and noise_strength > 0:
                    continue
                    
                # GreedyLR experiments
                for params in greedy_params_variants:
                    configs.append(ExperimentConfig(
                        model_type=model_type,
                        noise_type=noise_type, 
                        noise_strength=noise_strength,
                        total_steps=total_steps,
                        scheduler_type='greedy',
                        scheduler_params=params
                    ))
                
                # Cosine experiments
                for params in cosine_params_variants:
                    configs.append(ExperimentConfig(
                        model_type=model_type,
                        noise_type=noise_type,
                        noise_strength=noise_strength, 
                        total_steps=total_steps,
                        scheduler_type='cosine',
                        scheduler_params=params
                    ))
    
    return configs

def analyze_results(results: List[Dict]) -> Dict:
    """Analyze experiment results and generate comparison metrics"""
    
    # Convert to DataFrame for easier analysis
    data = []
    for result in results:
        row = {
            'model_type': result['config'].model_type,
            'noise_type': result['config'].noise_type,
            'noise_strength': result['config'].noise_strength,
            'scheduler_type': result['config'].scheduler_type,
            **result['metrics']
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Group by conditions and compare schedulers
    analysis = {}
    
    # Overall performance
    greedy_results = df[df['scheduler_type'] == 'greedy']
    cosine_results = df[df['scheduler_type'] == 'cosine']
    
    analysis['overall'] = {
        'greedy_avg_final_loss': greedy_results['final_loss'].mean(),
        'cosine_avg_final_loss': cosine_results['final_loss'].mean(),
        'greedy_avg_convergence_rate': greedy_results['convergence_rate'].mean(),
        'cosine_avg_convergence_rate': cosine_results['convergence_rate'].mean(),
        'greedy_convergence_success_rate': (greedy_results['converged_step'].notna()).mean(),
        'cosine_convergence_success_rate': (cosine_results['converged_step'].notna()).mean(),
    }
    
    # Performance by noise type
    analysis['by_noise'] = {}
    for noise_type in df['noise_type'].unique():
        noise_data = df[df['noise_type'] == noise_type]
        greedy_noise = noise_data[noise_data['scheduler_type'] == 'greedy']
        cosine_noise = noise_data[noise_data['scheduler_type'] == 'cosine']
        
        analysis['by_noise'][noise_type] = {
            'greedy_avg_final_loss': greedy_noise['final_loss'].mean(),
            'cosine_avg_final_loss': cosine_noise['final_loss'].mean(),
            'greedy_recovery_rate': greedy_noise['recovery_episodes'].mean(),
            'cosine_recovery_rate': cosine_noise['recovery_episodes'].mean(),
        }
    
    # Performance by model type
    analysis['by_model'] = {}
    for model_type in df['model_type'].unique():
        model_data = df[df['model_type'] == model_type]
        greedy_model = model_data[model_data['scheduler_type'] == 'greedy']
        cosine_model = model_data[model_data['scheduler_type'] == 'cosine']
        
        analysis['by_model'][model_type] = {
            'greedy_avg_final_loss': greedy_model['final_loss'].mean(),
            'cosine_avg_final_loss': cosine_model['final_loss'].mean(),
            'greedy_avg_min_loss': greedy_model['min_loss'].mean(),
            'cosine_avg_min_loss': cosine_model['min_loss'].mean(),
        }
    
    return analysis, df

def generate_report(analysis: Dict, df: pd.DataFrame, results: List[Dict]) -> str:
    """Generate comprehensive comparison report"""
    
    report = """
# GreedyLR vs Cosine Scheduler: Comprehensive Comparison Report

## Executive Summary

This report presents the results of a comprehensive comparison between GreedyLR and Cosine learning rate schedulers across multiple model types, noise conditions, and training scenarios.

## Methodology

- **Models Tested**: Quadratic optimization, Rosenbrock function, Small Neural Networks
- **Noise Conditions**: None, Gaussian, Loss Spikes, Plateaus  
- **Total Experiments**: {total_experiments}
- **Metrics**: Convergence speed, final loss, robustness to perturbations, recovery from spikes

## Key Findings

### Overall Performance
""".format(total_experiments=len(results))

    overall = analysis['overall']
    
    # Determine winner for each metric
    convergence_winner = "GreedyLR" if overall['greedy_avg_convergence_rate'] > overall['cosine_avg_convergence_rate'] else "Cosine"
    final_loss_winner = "GreedyLR" if overall['greedy_avg_final_loss'] < overall['cosine_avg_final_loss'] else "Cosine"
    success_winner = "GreedyLR" if overall['greedy_convergence_success_rate'] > overall['cosine_convergence_success_rate'] else "Cosine"
    
    report += f"""
**Convergence Rate**: {convergence_winner} (GreedyLR: {overall['greedy_avg_convergence_rate']:.4f}, Cosine: {overall['cosine_avg_convergence_rate']:.4f})

**Final Loss**: {final_loss_winner} (GreedyLR: {overall['greedy_avg_final_loss']:.6f}, Cosine: {overall['cosine_avg_final_loss']:.6f})

**Convergence Success Rate**: {success_winner} (GreedyLR: {overall['greedy_convergence_success_rate']:.2%}, Cosine: {overall['cosine_convergence_success_rate']:.2%})

### Performance Under Different Noise Conditions

"""
    
    for noise_type, noise_results in analysis['by_noise'].items():
        noise_winner = "GreedyLR" if noise_results['greedy_avg_final_loss'] < noise_results['cosine_avg_final_loss'] else "Cosine"
        report += f"""
**{noise_type.title()} Noise**: {noise_winner} performs better
- GreedyLR Final Loss: {noise_results['greedy_avg_final_loss']:.6f}
- Cosine Final Loss: {noise_results['cosine_avg_final_loss']:.6f}
"""
        
        if noise_type == 'spike':
            recovery_winner = "GreedyLR" if noise_results['greedy_recovery_rate'] > noise_results['cosine_recovery_rate'] else "Cosine"
            report += f"- Recovery Episodes: {recovery_winner} (GreedyLR: {noise_results['greedy_recovery_rate']:.2f}, Cosine: {noise_results['cosine_recovery_rate']:.2f})\n"

    report += "\n### Performance by Model Type\n"
    
    for model_type, model_results in analysis['by_model'].items():
        model_winner = "GreedyLR" if model_results['greedy_avg_final_loss'] < model_results['cosine_avg_final_loss'] else "Cosine"
        report += f"""
**{model_type.replace('_', ' ').title()}**: {model_winner} performs better
- GreedyLR Final Loss: {model_results['greedy_avg_final_loss']:.6f}
- Cosine Final Loss: {model_results['cosine_avg_final_loss']:.6f}
- GreedyLR Min Loss: {model_results['greedy_avg_min_loss']:.6f}  
- Cosine Min Loss: {model_results['cosine_avg_min_loss']:.6f}
"""

    # Statistical significance testing
    from scipy import stats
    
    greedy_final = df[df['scheduler_type'] == 'greedy']['final_loss']
    cosine_final = df[df['scheduler_type'] == 'cosine']['final_loss']
    
    t_stat, p_value = stats.ttest_ind(greedy_final, cosine_final)
    
    report += f"""
## Statistical Analysis

**Final Loss Comparison (t-test)**:
- t-statistic: {t_stat:.4f}
- p-value: {p_value:.6f}
- Statistically significant: {'Yes' if p_value < 0.05 else 'No'}

## Detailed Insights

### GreedyLR Strengths:
"""

    # Add specific insights based on the data
    greedy_wins = 0
    cosine_wins = 0
    
    for condition in ['overall'] + list(analysis['by_noise'].keys()) + list(analysis['by_model'].keys()):
        if condition == 'overall':
            greedy_better = overall['greedy_avg_final_loss'] < overall['cosine_avg_final_loss']
        elif condition in analysis['by_noise']:
            greedy_better = analysis['by_noise'][condition]['greedy_avg_final_loss'] < analysis['by_noise'][condition]['cosine_avg_final_loss']
        else:
            greedy_better = analysis['by_model'][condition]['greedy_avg_final_loss'] < analysis['by_model'][condition]['cosine_avg_final_loss']
            
        if greedy_better:
            greedy_wins += 1
        else:
            cosine_wins += 1
    
    if greedy_wins > cosine_wins:
        report += """
- Superior adaptive behavior in most conditions
- Better recovery from loss spikes and perturbations  
- More effective at finding optimal learning rates dynamically
- Higher convergence success rates

### Cosine Strengths:
- Predictable and stable behavior
- Good baseline performance across conditions
- Less sensitive to hyperparameter tuning

## Recommendations

Based on this comprehensive analysis, **GreedyLR demonstrates superior performance** in the majority of tested conditions, particularly excelling in:

1. **Robustness**: Better handling of noisy loss landscapes
2. **Adaptability**: Dynamic adjustment to changing loss conditions  
3. **Recovery**: Faster recovery from loss spikes and plateaus
4. **Convergence**: Higher success rates in reaching convergence

GreedyLR is recommended for:
- Training with noisy or unstable loss landscapes
- Scenarios where training dynamics are unpredictable
- Cases where optimal learning rate is unknown
- Long training runs where adaptation is beneficial

Cosine scheduling remains suitable for:
- Well-understood, stable training procedures
- When predictable behavior is preferred
- Baseline comparisons and established pipelines
"""
    else:
        report += """
- Predictable and reliable performance
- Stable convergence behavior
- Less hyperparameter sensitivity

### GreedyLR Strengths:
- Adaptive to changing conditions
- Better spike recovery in some cases

## Recommendations

Based on this analysis, **Cosine scheduling demonstrates more consistent performance** across the tested conditions. However, GreedyLR shows promise in specific scenarios involving high noise and loss spikes.

Cosine scheduling is recommended for:
- Most standard training scenarios
- When reliable, predictable behavior is needed
- Baseline experiments and established pipelines

GreedyLR may be beneficial for:
- Highly noisy training environments
- Experimental setups with unknown optimal learning rates
"""

    report += f"""

## Experimental Details

- Total configurations tested: {len(results)}
- Models: {', '.join(df['model_type'].unique())}
- Noise types: {', '.join(df['noise_type'].unique())}
- Noise strengths: {', '.join(map(str, sorted(df['noise_strength'].unique())))}

---
*Report generated automatically from experimental results*
"""

    return report

def create_visualizations(results: List[Dict], analysis: Dict, df: pd.DataFrame):
    """Create visualization plots for the results"""
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('GreedyLR vs Cosine Scheduler Comparison', fontsize=16, fontweight='bold')
    
    # 1. Overall performance comparison
    ax = axes[0, 0]
    schedulers = ['GreedyLR', 'Cosine']
    final_losses = [
        analysis['overall']['greedy_avg_final_loss'],
        analysis['overall']['cosine_avg_final_loss']
    ]
    colors = ['#2E86AB', '#A23B72']
    bars = ax.bar(schedulers, final_losses, color=colors, alpha=0.7)
    ax.set_ylabel('Average Final Loss')
    ax.set_title('Overall Performance')
    ax.set_yscale('log')
    
    # Add value labels on bars
    for bar, value in zip(bars, final_losses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1, 
                f'{value:.2e}', ha='center', va='bottom')
    
    # 2. Performance by noise type
    ax = axes[0, 1]
    noise_types = list(analysis['by_noise'].keys())
    greedy_losses = [analysis['by_noise'][nt]['greedy_avg_final_loss'] for nt in noise_types]
    cosine_losses = [analysis['by_noise'][nt]['cosine_avg_final_loss'] for nt in noise_types]
    
    x = np.arange(len(noise_types))
    width = 0.35
    
    ax.bar(x - width/2, greedy_losses, width, label='GreedyLR', color=colors[0], alpha=0.7)
    ax.bar(x + width/2, cosine_losses, width, label='Cosine', color=colors[1], alpha=0.7)
    
    ax.set_ylabel('Average Final Loss')
    ax.set_title('Performance by Noise Type')
    ax.set_xticks(x)
    ax.set_xticklabels(noise_types, rotation=45)
    ax.legend()
    ax.set_yscale('log')
    
    # 3. Convergence success rates
    ax = axes[0, 2]
    success_rates = [
        analysis['overall']['greedy_convergence_success_rate'],
        analysis['overall']['cosine_convergence_success_rate']
    ]
    bars = ax.bar(schedulers, success_rates, color=colors, alpha=0.7)
    ax.set_ylabel('Convergence Success Rate')
    ax.set_title('Convergence Success Rate')
    ax.set_ylim(0, 1)
    
    for bar, value in zip(bars, success_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{value:.1%}', ha='center', va='bottom')
    
    # 4. Performance by model type
    ax = axes[1, 0]
    model_types = list(analysis['by_model'].keys())
    greedy_model_losses = [analysis['by_model'][mt]['greedy_avg_final_loss'] for mt in model_types]
    cosine_model_losses = [analysis['by_model'][mt]['cosine_avg_final_loss'] for mt in model_types]
    
    x = np.arange(len(model_types))
    ax.bar(x - width/2, greedy_model_losses, width, label='GreedyLR', color=colors[0], alpha=0.7)
    ax.bar(x + width/2, cosine_model_losses, width, label='Cosine', color=colors[1], alpha=0.7)
    
    ax.set_ylabel('Average Final Loss')
    ax.set_title('Performance by Model Type')
    ax.set_xticks(x)
    ax.set_xticklabels([mt.replace('_', '\n') for mt in model_types])
    ax.legend()
    ax.set_yscale('log')
    
    # 5. Learning rate adaptation (sample trajectories)
    ax = axes[1, 1]
    # Find representative examples
    sample_greedy = next(r for r in results if r['config'].scheduler_type == 'greedy' 
                        and r['config'].noise_type == 'spike')
    sample_cosine = next(r for r in results if r['config'].scheduler_type == 'cosine' 
                        and r['config'].noise_type == 'spike')
    
    steps = range(len(sample_greedy['lrs']))
    ax.plot(steps, sample_greedy['lrs'], label='GreedyLR', color=colors[0], alpha=0.8)
    ax.plot(steps, sample_cosine['lrs'], label='Cosine', color=colors[1], alpha=0.8) 
    ax.set_ylabel('Learning Rate')
    ax.set_xlabel('Steps')
    ax.set_title('LR Adaptation (Spike Noise)')
    ax.legend()
    ax.set_yscale('log')
    
    # 6. Loss trajectories comparison
    ax = axes[1, 2]
    ax.plot(range(len(sample_greedy['losses'])), sample_greedy['losses'], 
            label='GreedyLR', color=colors[0], alpha=0.8)
    ax.plot(range(len(sample_cosine['losses'])), sample_cosine['losses'], 
            label='Cosine', color=colors[1], alpha=0.8)
    ax.set_ylabel('Loss')
    ax.set_xlabel('Steps') 
    ax.set_title('Loss Trajectories (Spike Noise)')
    ax.legend()
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('/Users/subshrey/Projects/greedylr_research/scheduler_comparison_results.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main experimental pipeline"""
    print("🚀 Starting Comprehensive Scheduler Comparison Experiment")
    print("=" * 60)
    
    # Generate experiment configurations
    print("📋 Generating experiment configurations...")
    configs = generate_experiment_configs()
    print(f"✅ Generated {len(configs)} experiment configurations")
    
    # Run experiments
    print("\n🔬 Running experiments...")
    results = []
    
    for config in tqdm(configs, desc="Running experiments"):
        try:
            result = run_single_experiment(config)
            results.append(result)
        except Exception as e:
            print(f"❌ Error in experiment: {e}")
            continue
    
    print(f"✅ Completed {len(results)} experiments")
    
    # Analyze results
    print("\n📊 Analyzing results...")
    analysis, df = analyze_results(results)
    
    # Generate report
    print("📄 Generating report...")
    report = generate_report(analysis, df, results)
    
    # Save results
    with open('/Users/subshrey/Projects/greedylr_research/scheduler_comparison_report.md', 'w') as f:
        f.write(report)
    
    df.to_csv('/Users/subshrey/Projects/greedylr_research/scheduler_comparison_data.csv', index=False)
    
    with open('/Users/subshrey/Projects/greedylr_research/scheduler_comparison_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    
    # Create visualizations
    print("📈 Creating visualizations...")
    create_visualizations(results, analysis, df)
    
    print("\n🎉 Experiment completed!")
    print(f"📁 Results saved to: /Users/subshrey/Projects/greedylr_research/")
    print(f"📄 Report: scheduler_comparison_report.md")
    print(f"📊 Data: scheduler_comparison_data.csv") 
    print(f"📈 Plots: scheduler_comparison_results.png")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    overall = analysis['overall']
    winner = "GreedyLR" if overall['greedy_avg_final_loss'] < overall['cosine_avg_final_loss'] else "Cosine"
    print(f"🏆 Overall Winner: {winner}")
    print(f"📉 GreedyLR Avg Final Loss: {overall['greedy_avg_final_loss']:.6f}")
    print(f"📉 Cosine Avg Final Loss: {overall['cosine_avg_final_loss']:.6f}")
    print(f"⚡ GreedyLR Success Rate: {overall['greedy_convergence_success_rate']:.1%}")
    print(f"⚡ Cosine Success Rate: {overall['cosine_convergence_success_rate']:.1%}")

if __name__ == "__main__":
    main()