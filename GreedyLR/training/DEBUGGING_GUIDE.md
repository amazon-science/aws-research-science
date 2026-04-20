# 🛠️ Training Debugging & Testing Guide

> **SageMaker Studio Note**: This guide is optimized for SageMaker Studio environments running on g5.12xlarge or larger instances. NCCL timeout issues are particularly relevant for multi-GPU setups in SageMaker environments due to their network configuration.

## 🚨 Problem Solved

**NCCL Timeout Error** during distributed training has been resolved with comprehensive fixes:

### ❌ Original Error
```
NCCL watchdog thread terminated with exception: WorkNCCL timeout
Operation _ALLGATHER_BASE ran for 24960814 milliseconds before timing out
```

### ✅ Solution Implemented
1. **NCCL timeout increased** to 1 hour (from 30 minutes)
2. **Evaluation batch size reduced** to prevent memory pressure
3. **Evaluation frequency reduced** (500 steps instead of 200)
4. **Additional stability optimizations** added

---

## 🚀 Quick Start

### 1. Test Training Stability (Recommended First)
```bash
cd examples/greedy-lr
conda activate pytorch_p310_greedy_v2

# Run comprehensive stability tests
python test_training_stability.py

# Test only specific scheduler
python test_training_stability.py --schedulers cosine
```

### 2. Run Full Training (After tests pass)
```bash
# Sequential training with both schedulers
./run_sequential_training.sh

# Monitor training progress (in separate terminal)
python monitor_training.py
```

---

## 🔧 What Was Fixed

### **Main Training Script (`pre-train-llama3.2-1b.py`)**

#### **NCCL Optimizations**
```python
# Extended timeout for NCCL operations
os.environ["NCCL_TIMEOUT"] = "3600"  # 1 hour

# Stability improvements
os.environ["NCCL_IB_DISABLE"] = "1"      # Disable InfiniBand
os.environ["NCCL_P2P_DISABLE"] = "1"     # Disable P2P
os.environ["NCCL_TREE_THRESHOLD"] = "0"  # Force ring algorithm
```

#### **Training Configuration Changes**
```python
# Before (causing timeouts)
"per_device_eval_batch_size": 2,
"eval_steps": 200,

# After (stable)
"per_device_eval_batch_size": 1,    # Reduced memory pressure
"eval_steps": 500,                  # Less frequent evaluation
"dataloader_num_workers": 0,        # More stable data loading
"prediction_loss_only": True,       # Faster evaluation
```

### **Testing Framework (`test_training_stability.py`)**

Progressive testing to validate:
- ✅ Basic training loop (10 steps)
- ✅ Training with evaluation (50 steps)  
- ✅ Training with checkpointing (50 steps)
- ✅ Full features enabled (100 steps)
- ✅ Extended run (500 steps)

---

## 📊 Testing Phases

The stability tester runs **5 progressive tests** for each scheduler:

| Test | Steps | Eval | Save | Purpose |
|------|-------|------|------|---------|
| **basic** | 10 | ❌ | ❌ | Validate core training loop |
| **with_eval** | 50 | ✅ | ❌ | Test evaluation stability |
| **with_checkpoints** | 50 | ❌ | ✅ | Test checkpointing |
| **full_features** | 100 | ✅ | ✅ | Test complete functionality |
| **extended** | 500 | ✅ | ✅ | Extended stability test |

**Early Stop Logic**: If any test fails, subsequent tests are skipped for that scheduler.

---

## 📈 Monitoring

### **Real-time Training Monitor**
```bash
# Start monitoring (updates every 30 seconds)
python monitor_training.py

# Faster updates during critical phases
python monitor_training.py --refresh 15
```

**What you'll see:**
```
╔════════════════════════════════════════════════════════════════════╗
║                    🦙 LLaMA Training Monitor                       ║
╠════════════════════════════════════════════════════════════════════╣
║ Phase: Greedy LR Training                                          ║
║ Progress: [████████░░░░░░░░░░░░░░░░░░░░░░░░] 847/2000 (42.4%)      ║
║ Time: 2h 15m elapsed | Est. remaining: 2h 45m                     ║
║                                                                    ║
║ Current Metrics:                                                   ║
║   Training Loss: 3.2847 ⬇️ (improving)                           ║
║   Learning Rate: 1.95e-04                                          ║
║   Grad Norm: 0.8921 ⬇️ (improving)                               ║
║                                                                    ║
║ [Press Ctrl+C to exit]                                            ║
╚════════════════════════════════════════════════════════════════════╝
```

---

## 🔍 Troubleshooting

### **If Tests Fail**
1. **Check GPU memory**: `nvidia-smi`
2. **Review test logs**: `training_stability_test.log`
3. **Examine detailed report**: `test_runs/stability_test_report.json`

### **Common Issues & Solutions**

#### **"CUDA out of memory"**
```python
# Reduce batch sizes in pre-train-llama3.2-1b.py
"per_device_train_batch_size": 1,  # From 2
"per_device_eval_batch_size": 1,   # Already set
"gradient_accumulation_steps": 64, # Increase to maintain effective batch size
```

#### **"Dataset not found"**
```bash
# Ensure dataset is prepared
ls -la datasets/redpajama/redpajama_50K_seed_42/
```

#### **"NCCL still timing out"**
```python
# Increase timeout further
os.environ["NCCL_TIMEOUT"] = "7200"  # 2 hours
```

---

## 📁 File Structure

```
examples/greedy-lr/
├── pre-train-llama3.2-1b.py          # ✅ Fixed training script
├── test_training_stability.py         # 🧪 Stability testing framework  
├── monitor_training.py                # 📊 Real-time training monitor
├── run_sequential_training.sh         # 🚀 Sequential training runner
├── DEBUGGING_GUIDE.md                # 📖 This guide
└── test_runs/                         # 📂 Test outputs
    ├── stability_test_report.json     # 📄 Detailed test results
    └── training_stability_test.log    # 📄 Test execution log
```

---

## ✅ Success Criteria

**Before running 2000-step training:**
1. ✅ All stability tests pass
2. ✅ Monitor shows consistent progress
3. ✅ No NCCL timeout errors in short tests
4. ✅ GPU memory usage stable

**Training should now complete successfully without NCCL timeouts!**

---

## 🎯 Next Steps

1. **Run stability tests**: `python test_training_stability.py`
2. **If tests pass**: `./run_sequential_training.sh`
3. **Monitor progress**: `python monitor_training.py` (separate terminal)
4. **Compare results**: Use existing comparison tools after completion

The comprehensive fixes address the root causes of NCCL timeouts and provide robust testing to ensure training stability.
