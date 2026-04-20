# Sequential Training Guide: Greedy LR → Cosine LR

This guide covers running sequential training phases with automatic GPU memory management between runs. This approach is optimized for SageMaker Studio environments on g5.12xlarge instances with 4 GPUs.

## Quick Start

```bash
# Navigate to the directory
cd examples/greedy-lr

# Activate the conda environment
conda activate pytorch_p310_greedy_v2

# Run sequential training
./run_sequential_training.sh
```

## What the Script Does

### Training Phases
1. **Phase 1**: Greedy LR training (500 steps)
2. **Comprehensive cleanup** with GPU memory verification
3. **Phase 2**: Cosine LR training (500 steps) 
4. **Final cleanup**

### Robust Memory Management
- **Process Termination**: Kills any lingering torchrun/Python processes
- **GPU Memory Clearing**: Uses CUDA cache clearing and garbage collection
- **Memory Verification**: Ensures GPU memory below 2GB threshold before proceeding
- **System Stabilization**: 30-second wait between phases for full cleanup

### Error Handling
- **Automatic cleanup** on script interruption (Ctrl+C)
- **Phase failure handling**: Stops execution if any phase fails
- **Memory threshold checking**: Warns if GPU memory can't be cleared sufficiently

## Configuration Options

Edit `run_sequential_training.sh` to customize:

```bash
# Number of GPUs to use
NUM_GPUS=4

# Wait time between phases (seconds)
CLEANUP_WAIT=30

# GPU memory threshold in GB (must be below this to proceed)
MEMORY_THRESHOLD_GB=2.0
```

## Output Structure

Results are saved to separate directories:
```
logs/codeparrot-ds/meta-llama/Llama-3.2-1B/run_test1/
├── greedy/2025-XX-XX/     # Greedy LR results
│   ├── tensorboard/       # TensorBoard logs
│   ├── output/           # Training checkpoints
│   ├── model/            # Final model
│   └── configs/          # Configuration files
└── cosine/2025-XX-XX/     # Cosine LR results
    ├── tensorboard/       # TensorBoard logs
    ├── output/           # Training checkpoints
    ├── model/            # Final model
    └── configs/          # Configuration files
```

## Monitoring

The script provides colored, timestamped logging:
- 🔵 **[INFO]**: General information and progress
- 🟢 **[SUCCESS]**: Successful operations
- 🟡 **[WARNING]**: Non-critical issues
- 🔴 **[ERROR]**: Critical failures

### GPU Status Monitoring
Real-time GPU memory and utilization display:
```
Current GPU Status:
  GPU 0 (NVIDIA A10G): 1024MB/23028MB (0% util)
  GPU 1 (NVIDIA A10G): 1024MB/23028MB (0% util)
  GPU 2 (NVIDIA A10G): 1024MB/23028MB (0% util)
  GPU 3 (NVIDIA A10G): 1024MB/23028MB (0% util)
  Total GPU Memory Used: 4.0GB
```

## Advanced Usage

### Single GPU Mode
To run with single GPU (if needed):
```bash
# Edit the script first
sed -i 's/NUM_GPUS=4/NUM_GPUS=1/' run_sequential_training.sh
./run_sequential_training.sh
```

### Custom Training Steps
To modify training duration, edit `pre-train-llama3.2-1b.py`:
```python
"max_steps": 1000,  # Change from 500 to 1000
```

### Manual Execution
If you prefer manual control:
```bash
# Phase 1: Greedy LR
torchrun --nproc_per_node=4 pre-train-llama3.2-1b.py --mode multi --lr_scheduler greedy

# Manual cleanup (optional)
python3 -c "import torch; [torch.cuda.empty_cache() for _ in range(torch.cuda.device_count())]; import gc; gc.collect()"

# Wait for stabilization
sleep 30

# Phase 2: Cosine LR  
torchrun --nproc_per_node=4 pre-train-llama3.2-1b.py --mode multi --lr_scheduler cosine
```

## Troubleshooting

### Common Issues

1. **"conda environment not activated"**
   ```bash
   conda activate pytorch_p310_greedy_v2
   ```

2. **"Training script not found"**
   ```bash
   cd examples/greedy-lr  # Ensure you're in the right directory
   ```

3. **"GPU memory not cleared"**
   - The script will continue with a warning
   - Consider increasing `MEMORY_THRESHOLD_GB` if persistent

4. **"bc command not found"**
   - The script will attempt to install it automatically
   - Manual install: `conda install -y bc`

### Memory Issues
If you encounter OOM errors:
1. **Reduce batch size** in `pre-train-llama3.2-1b.py`:
   ```python
   "per_device_train_batch_size": 1,  # Already optimized
   ```

2. **Use fewer GPUs**:
   ```bash
   sed -i 's/NUM_GPUS=4/NUM_GPUS=2/' run_sequential_training.sh
   ```

3. **Monitor with nvidia-smi**:
   ```bash
   watch -n 1 nvidia-smi
   ```

## Performance Expectations

### Resource Usage
- **GPU Memory**: ~14-16GB per GPU during training
- **CPU Usage**: Moderate (mainly data loading)
- **Disk I/O**: Model checkpoints and logs (~10-20GB total)

## Validation

After completion, validate results:

```bash
# Check log directories exist
ls -la logs/codeparrot-ds/meta-llama/Llama-3.2-1B/run_test1/*/

# View TensorBoard logs (optional)
tensorboard --logdir logs/codeparrot-ds/meta-llama/Llama-3.2-1B/run_test1/

# Check final models
ls -la logs/codeparrot-ds/meta-llama/Llama-3.2-1B/run_test1/*/model/
```

## Support

If you encounter issues:
1. Check the detailed logs in the terminal output
2. Verify GPU memory status with `nvidia-smi`
3. Ensure all dependencies are installed (`ninja`, `bc`, etc.)
4. Check that the conda environment is properly activated
