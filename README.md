# DATA 266 - Homework 1

## Student Information

- Name: Haritha Gottumukkala
- SID4: 1598
- SEED: 1598
- SLICE: 598
- HP_ID: 2
- CLS_A: 8
- CLS_B: 5

## Homework Overview

This repository contains my submission for DATA 266 Homework 1.

The homework covers three main parts:

1. Autoregressive models and real-world examples.
2. Feed-forward neural network experiments using PyTorch and TensorFlow.
3. CUDA matrix multiplication with CPU/GPU timing and profiling.

## Neural Network Experiment

The diabetes dataset was split into:

- 70% training
- 15% validation
- 15% testing

The same fixed dataset split was used for both PyTorch and TensorFlow.

### Baseline Configuration

- Hidden layers: [64, 32]
- Learning rate: 0.001
- Epochs: 30

### Modified Configuration

My assigned HP_ID is 2.

- Hidden layers: [64, 32]
- Learning rate: 0.003
- Epochs: 30

The models were trained using seeds 1598, 1599, and 1600.

### Final Neural Network Results

| Framework | Model | Mean Test Accuracy (%) | Std. Dev. |
|---|---|---:|---:|
| PyTorch | Baseline | 65.50 | 0.41 |
| PyTorch | Modified | 73.68 | 1.43 |
| TensorFlow | Baseline | 71.93 | 1.24 |
| TensorFlow | Modified | 73.10 | 0.83 |

## CUDA Experiment

CUDA matrix multiplication was tested using:

- GPU: NVIDIA Tesla T4
- CUDA architecture: sm_75
- Block size: 16 x 16 threads
- Matrix sizes: 256, 1024, 4096
- Profiler: nvprof
- Timing repetitions: 3 per matrix size

### Average CUDA Results

| Matrix Size | CPU (ms) | GPU Kernel (ms) | H2D + D2H (ms) | End-to-End Speedup |
|---:|---:|---:|---:|---:|
| 256 | 2.829 | 0.113 | 0.397 | 5.546x |
| 1024 | 214.517 | 4.794 | 4.907 | 22.113x |
| 4096 | 20472.542 | 317.206 | 77.207 | 51.906x |

## Repository Files

- `neural_networks.ipynb` - Autoregressive model discussion and neural network experiments
- `cuda.ipynb` - CUDA implementation, timing experiments, and profiling
- `matrix_multiplication.cu` - CUDA matrix multiplication source code
- `METRICS.md` - Summary of final experiment results
- `RUN_LOG.txt` - Recorded experiment outputs
- `AI_USE.md` - AI-use appendix
- `reportdata266_1598.pdf` - Final homework report
- `README.md` - Repository overview

## Reproducibility

The notebooks were designed to run from top to bottom using the required fixed parameters and training seeds. Final reported results are preserved in the executed notebooks and summarized in `METRICS.md` and `RUN_LOG.txt`.
