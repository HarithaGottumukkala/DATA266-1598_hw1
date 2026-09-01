# DATA 266 - Homework 1 Metrics

## Neural Network Results

HP_ID = 2

Baseline configuration: hidden layers [64, 32], learning rate 0.001, 30 epochs.

Modified configuration: hidden layers [64, 32], learning rate 0.003, 30 epochs.

Training seeds used: 1598, 1599, 1600.

| Framework | Model | Learning Rate | Mean Test Accuracy (%) | Std. Dev. (percentage points) |
|---|---|---:|---:|---:|
| PyTorch | Baseline | 0.001 | 65.50 | 0.41 |
| PyTorch | Modified | 0.003 | 73.68 | 1.43 |
| TensorFlow | Baseline | 0.001 | 71.93 | 1.24 |
| TensorFlow | Modified | 0.003 | 73.10 | 0.83 |

## CUDA Matrix Multiplication Results

Profiler used: nvprof

Each matrix size was measured three times. The values below are the averages.

| Matrix Size | CPU (ms) | GPU Kernel (ms) | H2D + D2H (ms) | End-to-End Speedup |
|---:|---:|---:|---:|---:|
| 256 | 2.829 | 0.113 | 0.397 | 5.546x |
| 1024 | 214.517 | 4.794 | 4.907 | 22.113x |
| 4096 | 20472.542 | 317.206 | 77.207 | 51.906x |
