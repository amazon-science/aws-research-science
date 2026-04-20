# GreedyLR

Adaptive learning rate scheduler that greedily selects the best learning rate at each step based on loss trajectory. Evaluated across 8,100 experiments showing consistent improvement over cosine, cosine-with-restarts, and exponential decay schedulers.

## Status

This repository is the **starting point for the next set of GreedyLR experiments**, building on the results below. Immediate next steps include integrating GreedyLR into GRPO fine-tuning (see [grpo/](grpo/)) and extending pre-training evaluations to larger models.

## Key Results

| Domain | GreedyLR Advantage | Significance |
|---|---|---|
| Neural Networks | 2× better median performance | p < 0.01 |
| Analytical Functions | 2.6× better median performance | p < 0.001 |
| Noisy Conditions | Exceptional robustness across all noise types | p < 0.001 |

See [docs/FINAL_COMPREHENSIVE_README.md](docs/FINAL_COMPREHENSIVE_README.md) for the full write-up.

---

## Repository Structure

```
GreedyLR/
├── experiments/    # experiment runners and launchers
├── analysis/       # result analysis and statistics
├── plotting/       # figure generation scripts
├── figures/        # all output figures (pub/, final/, research/, clear/)
├── docs/           # reports and documentation
├── training/       # LLM pre-training and fine-tuning code
├── grpo/           # GRPO fine-tuning experiments
└── archive/        # legacy fix scripts
```

---

## File Index

### experiments/
| File | Description |
|---|---|
| [comprehensive_scheduler_experiment.py](experiments/comprehensive_scheduler_experiment.py) | Main runner: 8,100 runs across 12 architectures × 9 noise types |
| [research_paper_scheduler_experiment.py](experiments/research_paper_scheduler_experiment.py) | Publication-quality experiment with full statistics |
| [scheduler_comparison_experiment.py](experiments/scheduler_comparison_experiment.py) | Simplified head-to-head scheduler comparison |
| [robust_comprehensive_experiment.py](experiments/robust_comprehensive_experiment.py) | Robustness-focused suite |
| [focused_recovery_experiment.py](experiments/focused_recovery_experiment.py) | LR recovery behavior analysis |
| [balanced_100k_experiment.py](experiments/balanced_100k_experiment.py) | Large-scale balanced experiment |
| [optimized_experiment.py](experiments/optimized_experiment.py) | GPU-optimized runner |
| [start_experiment.sh](experiments/start_experiment.sh) | Shell launcher |
| [monitor_progress.py](experiments/monitor_progress.py) | Live progress monitor |
| [test_gpu_speedup.py](experiments/test_gpu_speedup.py) | GPU vs CPU benchmarks |

### analysis/
| File | Description |
|---|---|
| [analyze_results.py](analysis/analyze_results.py) | General results analysis |
| [comprehensive_research_analysis.py](analysis/comprehensive_research_analysis.py) | Full statistical analysis pipeline |
| [architecture_specific_analysis.py](analysis/architecture_specific_analysis.py) | Per-architecture breakdown |
| [better_stability_analysis.py](analysis/better_stability_analysis.py) | Convergence stability analysis |
| [convergence_failure_analysis.py](analysis/convergence_failure_analysis.py) | Failure mode investigation |
| [recover_results.py](analysis/recover_results.py) | Result recovery from partial runs |
| [architecture_specific_analysis.json](analysis/architecture_specific_analysis.json) | Per-architecture results data |
| [scheduler_comparison_analysis.json](analysis/scheduler_comparison_analysis.json) | Scheduler comparison results data |

### plotting/
| File | Description |
|---|---|
| [publication_ready_plots.py](plotting/publication_ready_plots.py) | Final publication figures |
| [journal_quality_plots.py](plotting/journal_quality_plots.py) | Journal-format plots |
| [comprehensive_analysis_plots.py](plotting/comprehensive_analysis_plots.py) | Full analysis visualization |
| [advanced_analysis_plots.py](plotting/advanced_analysis_plots.py) | Advanced statistical plots |
| [create_final_plots.py](plotting/create_final_plots.py) | Final figure pipeline |

### figures/
| Directory | Description |
|---|---|
| [figures/pub/](figures/pub/) | Publication-ready figures (figures 1–3) |
| [figures/final/](figures/final/) | Complete set of final experiment plots |
| [figures/research/](figures/research/) | Research-phase plots and heatmaps |
| [figures/clear/](figures/clear/) | Clean versions of key figures |

### docs/
| File | Description |
|---|---|
| [docs/FINAL_COMPREHENSIVE_README.md](docs/FINAL_COMPREHENSIVE_README.md) | Full experimental analysis and results |
| [docs/COMPREHENSIVE_RESULTS.md](docs/COMPREHENSIVE_RESULTS.md) | Summary tables of all 8,100 outcomes |
| [docs/RESULTS.md](docs/RESULTS.md) | High-level results overview |
| [docs/EXPERIMENT_DOCUMENTATION.md](docs/EXPERIMENT_DOCUMENTATION.md) | Experimental design and methodology |
| [docs/scheduler_comparison_report.md](docs/scheduler_comparison_report.md) | Head-to-head scheduler comparison |

### training/
| File | Description |
|---|---|
| [training/fine-tune/run_llm_finetune.py](training/fine-tune/run_llm_finetune.py) | LoRA fine-tuning with GreedyLR, cosine, constant schedulers |
| [training/fine-tune/fine-tune-llm-doe.ipynb](training/fine-tune/fine-tune-llm-doe.ipynb) | Fine-tuning notebook (Falcon-7B, xP3mt dataset) |
| [training/pre-train-llama3.2-1b.py](training/pre-train-llama3.2-1b.py) | LLaMA 3.2-1B pre-training with GreedyLR |
| [training/pre-train-gpt2.py](training/pre-train-gpt2.py) | GPT-2 pre-training script |
| [training/compare_schedulers.py](training/compare_schedulers.py) | Direct scheduler comparison utility |
| [training/monitor_training.py](training/monitor_training.py) | Live training monitor |
| [training/README.md](training/README.md) | Training setup and usage guide |

### grpo/
| File | Description |
|---|---|
| [grpo/grpo_training_v1.ipynb](grpo/grpo_training_v1.ipynb) | GRPO fine-tuning on Llama-3.2-1B with TRL |
| [grpo/grpo_training_v2_unsloth.ipynb](grpo/grpo_training_v2_unsloth.ipynb) | Unsloth-optimized GRPO training (2× faster) |
| [grpo/grpo_sample_data.jsonl](grpo/grpo_sample_data.jsonl) | 32-sample GRPO dataset (math CoT format) |

#### From Loss Minimization to Reward Maximization

The core shift in the GRPO experiments is replacing the standard supervised cross-entropy loss with a reward-maximizing GRPO objective.

**Before — standard SFT (cross-entropy loss):**

In the pre-training and fine-tuning scripts (`training/`), the model minimizes token-level cross-entropy against ground-truth completions:

```python
# Standard causal LM training — loss computed internally by HuggingFace Trainer
trainer = Trainer(
    model=model,
    args=training_args,          # lr_scheduler_type="greedy", etc.
    train_dataset=train_dataset,
    ...
)
# Loss = -sum( log P(token_t | token_1..t-1) )  for each ground-truth token
```

**How the metric reaches GreedyLR — the single entry point:**

`GreedyLR.step(metric)` (`GreedyLR.py:149`) is the only place any signal enters the scheduler. It takes a scalar, compares it against the running best, and adjusts the LR:

```python
# transformers/src/transformers/GreedyLR.py:149-186
def step(self, metrics, epoch=None):
    current = float(metrics)          # whatever scalar is passed in — loss or reward
    if self.smooth:
        current = self.sa.streamavg(current)   # optional streaming average

    if self.is_better(current, self.best):     # direction depends on mode='min' or 'max'
        self.best = current
        self.num_good_epochs += 1
    else:
        self.num_bad_epochs += 1

    if self.num_bad_epochs > self.patience:
        self._reduce_lr(epoch)        # lr = lr * factor (e.g. 0.95)
    if self.num_good_epochs > self.patience:
        self._increase_lr(epoch)      # lr = lr / factor
```

**In SFT**, the HuggingFace Trainer calls this at `trainer.py:2622` after each gradient step, passing the cross-entropy training loss:

```python
# transformers/src/transformers/trainer.py:2564, 2622
tr_loss_step = self.training_step(model, inputs, ...)   # cross-entropy loss + backward pass
...
self.lr_scheduler.step(tr_loss_step)   # loss → GreedyLR, mode='min'
# loss goes down → num_good_epochs++ → LR increases
# loss stalls   → num_bad_epochs++  → LR decreases
```

**In GRPO**, the cross-entropy loss still exists — it is used internally by the GRPO policy gradient update to compute gradients. What changes is what GreedyLR watches. Instead of watching the loss, GreedyLR now receives the mean batch reward, and its mode is flipped to `'max'`:

```python
# How GreedyLR is wired into GRPO (mode='max', reward replaces loss as the metric)
scheduler = GreedyLR(optimizer, mode='max', ...)  # 'max' because higher reward = better

# After each batch, compute mean reward from reward_func outputs and step the scheduler
mean_reward = torch.tensor(reward_func(prompts, completions)).mean()
scheduler.step(mean_reward)   # reward goes up → num_good_epochs++ → LR increases
                               # reward stalls  → num_bad_epochs++  → LR decreases
```

The GRPO policy gradient loss (which drives the optimizer) is unchanged — GreedyLR just uses reward instead of loss as its signal for deciding whether to raise or lower the LR.

**After — GRPO (reward maximization):**

GRPO replaces the loss signal entirely with a reward function. The model generates multiple candidate completions per prompt, scores them, and updates policy weights to increase the probability of higher-reward outputs. There is no ground-truth token sequence to predict — the model is optimized against scalar rewards.

The reward in v1 combined format correctness with semantic similarity via embeddings:

```python
# grpo_training_v1.ipynb
def format_reward_func(completions):
    """1.0 if output matches 'answer: <...>', else 0.0"""
    pattern = r'^answer: <.*>$'
    matches = [re.match(pattern, content, re.DOTALL) for content in completions]
    return [1.0 if match else 0.0 for match in matches]

def answer_similarity(prompts, completions):
    """Cosine similarity between completion and ground-truth embeddings (b1ade-embed model)"""
    completion_embeddings = emodel.encode(processed_completions)
    ground_truth_embeddings = emodel.encode(valid_ground_truth)
    return emodel.similarity(completion_embeddings, ground_truth_embeddings).diagonal()

def reward_func(prompts, completions):
    # Final reward = semantic similarity × format gate (0 or 1)
    return answer_similarity(prompts, completions) * torch.Tensor(format_reward_func(completions))
```

In v2, the embedding similarity was replaced with ROUGE-L (lighter, no GPU embedding model needed), and a length reward was added to encourage fuller responses:

```python
# grpo_training_v2_unsloth.ipynb
def correctness_reward_func(prompts, completions, answer, **kwargs):
    """ROUGE-L F1 score between extracted completion and ground-truth answer"""
    rouge = Rouge()
    extracted = [extract_content(r) for r in responses]
    return [s["rouge-l"]["f"] for s in rouge.get_scores(extracted, answer)]

def len_reward_func(prompts, completions):
    """Normalized length reward — encourages longer (more complete) responses"""
    return [len(c) / 1024. for c in completions]

def reward_func(prompts, completions):
    # Final reward = ROUGE-L similarity × format gate × length
    return answer_similarity(prompts, completions) * format_reward * len_reward
```

The `GRPOTrainer` from TRL handles the policy gradient update — it passes `reward_funcs` in place of a loss function:

```python
trainer = GRPOTrainer(
    model="w601sxs/b1ade-1b-bf16",
    reward_funcs=reward_func,   # replaces loss_fn entirely
    args=GRPOConfig(
        learning_rate=1e-5,
        use_vllm=True,          # vLLM for fast multi-sample generation
        ...
    ),
    train_dataset=updated_dataset,
    peft_config=LoraConfig(task_type="CAUSAL_LM", r=4),
)
```

**Summary of changes across versions:**

| | SFT (training/) | GRPO v1 | GRPO v2 |
|---|---|---|---|
| **Objective** | Minimize cross-entropy | Maximize reward | Maximize reward |
| **Supervision signal** | Ground-truth tokens | Scalar reward per completion | Scalar reward per completion |
| **Similarity metric** | — | Embedding cosine sim (b1ade-embed) | ROUGE-L F1 |
| **Format enforcement** | — | Regex gate (× reward) | Regex gate (× reward) |
| **Length incentive** | — | None | `len(completion) / 1024` |
| **LR scheduler** | GreedyLR / cosine | Cosine (default) | Cosine (default) |
| **Model efficiency** | Full precision | LoRA r=4 | LoRA r=8 + Unsloth 4-bit |

> **Next step:** implement the GRPO + GreedyLR integration described above in `grpo_training_v2_unsloth.ipynb` and measure whether adaptive LR produces the same gains seen in supervised settings.
