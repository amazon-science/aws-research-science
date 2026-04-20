# Monitor Training Tool

The Monitor Training tool provides real-time visibility into your LLaMA training process by reading TensorBoard logs directly and displaying key metrics in a user-friendly dashboard. This is especially useful in SageMaker Studio environments where you want to monitor training without launching a separate TensorBoard server.

## Quick Start

```bash
# Activate the required conda environment
conda activate pytorch_p310_greedy_v2

# Start monitoring with default settings
python monitor_training.py

# Monitor a specific training run
python monitor_training.py --log-dir ./logs/redpajama/meta-llama/Llama-3.2-1B/run_3

# Refresh more frequently (every 15 seconds)
python monitor_training.py --refresh 15
```

## Features

- **Real-time metrics**: Training loss, evaluation loss, learning rate, gradient norm
- **Progress visualization**: Progress bar showing completion percentage
- **Time estimation**: Elapsed time and estimated remaining time
- **Training stage detection**: Automatically detects whether Greedy LR or Cosine LR is active
- **Trend indicators**: Shows whether metrics are improving, degrading, or stable

## Command-Line Options

```
Usage: monitor_training.py [OPTIONS]

Options:
  --log-dir TEXT    Base directory containing training logs
                   (default: ./logs/redpajama/meta-llama/Llama-3.2-1B/run_2)
  --refresh INTEGER Refresh interval in seconds (default: 30)
```

## Dashboard Interface

The monitor displays a dashboard like this:

```
╔════════════════════════════════════════════════════════════════════╗
║                    🦙 LLaMA Training Monitor                       ║
╠════════════════════════════════════════════════════════════════════╣
║ Last Updated: 15:30:45                                             ║
║ Phase: Greedy LR Training                                          ║
║ Progress: [████████░░░░░░░░░░░░░░░░░░░░░░░░] 847/2000 (42.4%)      ║
║ Time: 2h 15m elapsed | Est. remaining: 2h 45m                     ║
║                                                                    ║
║ Current Metrics:                                                   ║
║   Training Loss: 3.2847 ⬇️ (improving)                           ║
║   Learning Rate: 1.95e-04                                          ║
║   Grad Norm: 0.8921 ⬇️ (improving)                               ║
║                                                                    ║
║ Recent Loss Trend (last 5 steps):                                  ║
║   800: 3.3124 → 810: 3.3056 → 820: 3.2987 → 830: 3.2924 → 840: 3.2847 ║
║                                                                    ║
║ [Press Ctrl+C to exit]                                            ║
╚════════════════════════════════════════════════════════════════════╝
```

## Usage Tips

1. **Launch in a separate terminal** alongside your training process
2. **Leave running** for continuous monitoring throughout training
3. **Use faster refresh rates** (10-15 seconds) during critical phases
4. **Adjust terminal size** for optimal dashboard display (at least 80x24)

## Interpreting Trend Indicators

- **⬇️ (improving)**: Metric is consistently decreasing (good for loss)
- **⬆️ (degrading)**: Metric is consistently increasing (bad for loss)
- **➡️ (stable)**: Metric is neither significantly improving nor degrading

## Supported Metrics

The monitor automatically tracks and displays available metrics, typically including:

- **train/loss**: Loss on training data
- **eval/loss**: Loss on validation data (only appears during evaluation steps)
- **train/learning_rate**: Current learning rate
- **train/grad_norm**: Gradient norm during training
