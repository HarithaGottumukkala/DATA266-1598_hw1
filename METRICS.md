# DATA 266 - Homework 1 Metrics

## Personal Parameters

- SID4 = 1598
- SEED = 1598
- SLICE = 598
- HP_ID = 2
- CLS_A = 8
- CLS_B = 5

## Neural Network Results

HP_ID = 2

Modified configuration:
- Hidden layers: [64, 32]
- Learning rate: 0.003
- Epochs: 30

| Framework | Model | Mean Test Accuracy | Standard Deviation |
|---|---|---:|---:|
| PyTorch | Baseline | TBD | TBD |
| PyTorch | HP_ID 2 Modified | TBD | TBD |
| TensorFlow | Baseline | TBD | TBD |
| TensorFlow | HP_ID 2 Modified | TBD | TBD |

## CUDA Matrix Multiplication Results

Profiler used: nvprof

| Matrix Size | CPU (ms) | GPU Kernel (ms) | H2D+D2H (ms) | End-to-End Speedup |
|---:|---:|---:|---:|---:|
| 256 | 2.829 | 0.113 | 0.397 | 5.546x |
| 1024 | 214.517 | 4.794 | 4.907 | 22.113x |
| 4096 | 20472.542 | 317.206 | 77.207 | 51.906x |
