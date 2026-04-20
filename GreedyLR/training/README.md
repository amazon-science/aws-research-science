# LLaMA 3.2 1B Training Tools

This directory contains scripts and utilities for training LLaMA 3.2 1B models with different learning rate schedulers, specifically comparing Greedy LR and Cosine LR approaches.

## Overview

These tools allow you to:
1. Train LLaMA 3.2 1B with Greedy LR or Cosine LR schedulers
2. Run sequential training phases with robust GPU memory management
3. Compare training results between different schedulers

## Setup Instructions for SageMaker Studio

### 1. Instance Requirements

- **Instance Type**: g5.12xlarge or larger
- **Storage**: At least 100GB to accommodate model checkpoints and datasets

### 2. Environment Setup

```bash
# Clone the repository
git clone https://github.com/balak4/transformers.git
cd transformers

# Create and activate conda environment
cd examples/greedy-lr
conda env create -f conda/pytorch_p310_greedy_v2.yml
conda activate pytorch_p310_greedy_v2

# Install modified transformers fork in local env
python3 -m pip install -e /home/sagemaker-user/transformers/
```

## Training Data Preparation

### Option 1: Use pre-prepared datasets (Recommended)

Pre-tokenized datasets are available in the following S3 bucket:
```bash
aws s3 ls s3://greedylr-research-artifacts/datasets/
```

Download your dataset of choice:
```bash
aws s3 cp s3://greedylr-research-artifacts/datasets/redpajama/redpajama_50K_seed_42/ ./datasets/redpajama/redpajama_50K_seed_42/ --recursive
```

### Option 2: Prepare your own dataset

If your dataset is not available in the S3 bucket, you can prepare it using the data preparation script:

```bash
# Example for RedPajama dataset
python llama3.2-1b_pretraining_data_prep.py \
  --dataset_id "togethercomputer/RedPajama-Data-1T" \
  --tokenizer_id "meta-llama/Llama-3.2-1B" \
  --train_samples 50000 \
  --text_column "text" \
  --output_name "redpajama_50K_seed_42"
```

## Scripts

### Training Scripts

- **`pre-train-llama3.2-1b.py`**: Main training script for LLaMA 3.2 1B
- **`llama3.2-1b_pretraining_data_prep.py`**: Data preparation script for pre-training
- **`run_sequential_training.sh`**: Run sequential training with Greedy LR followed by Cosine LR
- **`test_training_stability.py`**: Progressive stability testing framework

### Comparison Tools

- **`compare_schedulers.py`**: Generate comparison plots and reports between schedulers
- **`monitor_training.py`**: Real-time monitoring of training progress

## Quick Start

### Running Greedy LR → Cosine LR Sequential Training

```bash
# Ensure conda environment is activated
conda activate pytorch_p310_greedy_v2

# Run sequential training
# If using huggingface datasets, login to huggingface hub
huggingface-cli login
./run_sequential_training.sh
```

### Running Individual Training (Cosine or Greedy)

You can also run individual training with either scheduler using the main training script:

```bash
# For Greedy LR (multi-GPU)
torchrun --nproc_per_node=4 pre-train-llama3.2-1b.py --mode multi --lr_scheduler greedy

# For Cosine LR (multi-GPU)
torchrun --nproc_per_node=4 pre-train-llama3.2-1b.py --mode multi --lr_scheduler cosine
```

### Comparing Scheduler Results

After running both schedulers, you can compare their performance:

```bash
# Run with default paths
python compare_schedulers.py

# Or specify custom paths
python compare_schedulers.py \
  --greedy-dir ./logs/redpajama/meta-llama/Llama-3.2-1B/run_1/greedy/YYYY-MM-DD/tensorboard \
  --cosine-dir ./logs/redpajama/meta-llama/Llama-3.2-1B/run_1/cosine/YYYY-MM-DD/tensorboard \
  --output-dir ./my_comparison_results
```

## Training Configuration

The current configuration is optimized for training on 4 GPUs with DeepSpeed ZeRO-3:

- **Model**: LLaMA 3.2 1B
- **Batch Size**: 2 per device (effective batch size: 8)
- **Gradient Accumulation**: 64 steps
- **Training Steps**: 500 (default for each phase)
- **BF16 Precision**: Enabled
- **Gradient Checkpointing**: Enabled

## Understanding the Results

See `SEQUENTIAL_TRAINING_GUIDE.md` for details on the sequential training process and expected outcomes.

The comparison script will generate a directory containing:
- `comparison_report.md`: Detailed analysis of the two schedulers
- Various plots comparing metrics like loss and learning rate curves

## Monitoring Training Progress

For real-time monitoring of training, use the included monitoring script:

```bash
# In a separate terminal (while training is running)
conda activate pytorch_p310_greedy_v2
python monitor_training.py
```

See `MONITOR_USAGE.md` for more information on monitoring options.

## Troubleshooting

If you encounter issues, refer to `DEBUGGING_GUIDE.md` for:
- Common error solutions
- Memory optimization techniques
- NCCL timeout prevention strategies

## Training Stability Testing

Before running long training jobs, it's recommended to validate training stability using the included testing framework:

```bash
# Run stability tests for both schedulers
python test_training_stability.py

# Test only specific scheduler
python test_training_stability.py --schedulers greedy

# Run a specific test configuration
python test_training_stability.py --test basic
```

### Progressive Test Configurations

The framework runs a series of 5 tests with increasing complexity:

| Test | Steps | Eval | Save | Purpose |
|------|-------|------|------|---------|
| **basic** | 10 | ❌ | ❌ | Validate core training loop |
| **with_eval** | 50 | ✅ | ❌ | Test evaluation stability |
| **with_checkpoints** | 10 | ✅ | ✅ | Test checkpointing |
| **full_features** | 100 | ✅ | ✅ | Test complete functionality |
| **extended** | 500 | ✅ | ✅ | Extended stability test |

### Features

- **System Resource Checking**: Validates CPU, memory, and GPU resources
- **Process Cleanup**: Ensures clean testing environment for each test
- **Memory Tracking**: Monitors GPU memory usage during critical operations
- **Detailed Reporting**: Generates comprehensive reports with test results

### Interpreting Results

After tests complete, a detailed stability report is generated:
- **Location**: `./test_runs/stability_test_report.json`
- **Success Criteria**: All tests should pass before running full training
- **Failure Analysis**: If tests fail, check memory usage, NCCL communication, and dataset performance

## Memory Optimization

See `MEMORY_OPTIMIZATION_SUMMARY.md` for detailed memory usage information and optimization techniques used in this implementation.
