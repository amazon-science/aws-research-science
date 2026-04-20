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

See [FINAL_COMPREHENSIVE_README.md](FINAL_COMPREHENSIVE_README.md) for the full write-up.

---

## Repository Index

### Documentation
| File | Description |
|---|---|
| [FINAL_COMPREHENSIVE_README.md](FINAL_COMPREHENSIVE_README.md) | Full experimental analysis and results |
| [COMPREHENSIVE_RESULTS.md](COMPREHENSIVE_RESULTS.md) | Summary tables of all 8,100 experiment outcomes |
| [RESULTS.md](RESULTS.md) | High-level results overview |
| [EXPERIMENT_DOCUMENTATION.md](EXPERIMENT_DOCUMENTATION.md) | Experimental design, configs, and methodology |
| [scheduler_comparison_report.md](scheduler_comparison_report.md) | Head-to-head scheduler comparison report |

### Core Experiment Scripts
| File | Description |
|---|---|
| [comprehensive_scheduler_experiment.py](comprehensive_scheduler_experiment.py) | Main experiment runner (8,100 runs, 12 architectures × 9 noise types) |
| [research_paper_scheduler_experiment.py](research_paper_scheduler_experiment.py) | Publication-quality experiment with full stats |
| [scheduler_comparison_experiment.py](scheduler_comparison_experiment.py) | Simplified scheduler comparison |
| [robust_comprehensive_experiment.py](robust_comprehensive_experiment.py) | Robustness-focused experiment suite |
| [focused_recovery_experiment.py](focused_recovery_experiment.py) | LR recovery behavior analysis |
| [balanced_100k_experiment.py](balanced_100k_experiment.py) | Large-scale balanced experiment |
| [optimized_experiment.py](optimized_experiment.py) | GPU-optimized experiment runner |

### Analysis Scripts
| File | Description |
|---|---|
| [analyze_results.py](analyze_results.py) | General results analysis |
| [comprehensive_research_analysis.py](comprehensive_research_analysis.py) | Full statistical analysis pipeline |
| [architecture_specific_analysis.py](architecture_specific_analysis.py) | Per-architecture breakdown |
| [better_stability_analysis.py](better_stability_analysis.py) | Convergence stability analysis |
| [convergence_failure_analysis.py](convergence_failure_analysis.py) | Failure mode investigation |
| [investigate_data.py](investigate_data.py) | Raw data inspection utilities |
| [recover_results.py](recover_results.py) | Result recovery from partial runs |

### Visualization Scripts
| File | Description |
|---|---|
| [publication_ready_plots.py](publication_ready_plots.py) | Final publication figures |
| [journal_quality_plots.py](journal_quality_plots.py) | Journal-format plot generation |
| [clean_professional_plots.py](clean_professional_plots.py) | Clean plot variants |
| [comprehensive_analysis_plots.py](comprehensive_analysis_plots.py) | Full analysis visualization |
| [advanced_analysis_plots.py](advanced_analysis_plots.py) | Advanced statistical plots |
| [create_final_plots.py](create_final_plots.py) | Final figure pipeline |

### Plot Outputs
| Directory | Description |
|---|---|
| [pub_plots/](pub_plots/) | Publication-ready figures (figures 1–3) |
| [final_plots/](final_plots/) | Full set of final experiment plots |
| [clear_plots/](clear_plots/) | Clean versions of key figures |
| [research_plots/](research_plots/) | Research-phase plots and heatmaps |

### Training Code
| File | Description |
|---|---|
| [training/fine-tune/run_llm_finetune.py](training/fine-tune/run_llm_finetune.py) | LoRA fine-tuning script with GreedyLR, cosine, and constant schedulers |
| [training/fine-tune/fine-tune-llm-doe.ipynb](training/fine-tune/fine-tune-llm-doe.ipynb) | Fine-tuning notebook (Falcon-7B, xP3mt dataset) |
| [training/pre-train-llama3.2-1b.py](training/pre-train-llama3.2-1b.py) | LLaMA 3.2-1B pre-training with GreedyLR |
| [training/pre-train-gpt2.py](training/pre-train-gpt2.py) | GPT-2 pre-training script |
| [training/pre-train-qwen2.5-0.5b-cosine.py](training/pre-train-qwen2.5-0.5b-cosine.py) | Qwen2.5-0.5B baseline (cosine) |
| [training/compare_schedulers.py](training/compare_schedulers.py) | Direct scheduler comparison utility |
| [training/monitor_training.py](training/monitor_training.py) | Live training monitor |
| [training/run_sequential_training.sh](training/run_sequential_training.sh) | Sequential training launcher |
| [training/README.md](training/README.md) | Training setup and usage guide |

### GRPO Fine-Tuning Experiments
| File | Description |
|---|---|
| [grpo/grpo_training_v1.ipynb](grpo/grpo_training_v1.ipynb) | GRPO fine-tuning on Llama-3.2-1B with TRL |
| [grpo/grpo_training_v2_unsloth.ipynb](grpo/grpo_training_v2_unsloth.ipynb) | Unsloth-optimized GRPO training (2× faster) |
| [grpo/grpo_sample_data.jsonl](grpo/grpo_sample_data.jsonl) | 32-sample GRPO dataset (math CoT format) |

### Utilities
| File | Description |
|---|---|
| [start_experiment.sh](start_experiment.sh) | Shell script to launch experiment runs |
| [monitor_progress.py](monitor_progress.py) | Live experiment progress monitor |
| [test_gpu_speedup.py](test_gpu_speedup.py) | GPU vs CPU speedup benchmarks |
| [markdown_to_pdf.py](markdown_to_pdf.py) | Convert markdown reports to PDF |
