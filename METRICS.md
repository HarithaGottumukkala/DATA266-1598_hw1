# DATA 266 - Homework 1 Metrics

## Personal Parameters

- SID4 = 1598
- SEED = 1598
- SLICE = 598
- HP_ID = 2
- CLS_A = 8
- CLS_B = 5


## Neural Network Experiment

### Dataset Split

- Training: 70%
- Validation: 15%
- Testing: 15%
- Split random state: SEED = 1598

The same training, validation, and testing split was used for all PyTorch and TensorFlow model comparisons.


### Baseline Configuration

- Hidden layers: [64, 32]
- Learning rate: 0.001
- Epochs: 30
- Evaluation metric: Accuracy


### HP_ID Modified Configuration

HP_ID = 2

- Hidden layers: [64, 32]
- Learning rate: 0.003
- Epochs: 30

For HP_ID = 2, the architecture and number of epochs remain the same as the baseline. The learning rate is increased from 0.001 to 0.003.


### Neural Network Test Accuracy

The dataset split was kept fixed, and each model was trained using training seeds 1598, 1599, and 1600.

| Framework | Model | Learning Rate | Mean Test Accuracy (%) | Standard Deviation |
|---|---|---:|---:|---:|
| PyTorch | Baseline | 0.001 | 65.50 | 0.41 |
| PyTorch | Modified | 0.003 | 73.68 | 1.43 |
| TensorFlow | Baseline | 0.001 | 71.93 | 1.24 |
| TensorFlow | Modified | 0.003 | 73.10 | 0.83 |


### Effect of the HP_ID Modification

| Framework | Baseline Mean Accuracy (%) | Modified Mean Accuracy (%) | Change |
|---|---:|---:|---:|
| PyTorch | 65.50 | 73.68 | +8.19 percentage points |
| TensorFlow | 71.93 | 73.10 | +1.17 percentage points |


## CUDA Matrix Multiplication

### CUDA Configuration

- GPU: NVIDIA Tesla T4
- CUDA architecture: sm_75
- CUDA block size: 16 × 16 threads
- Profiler: nvprof
- Timing repetitions: 3 per matrix size


### CUDA Timing Results

The values below are the averages from three timing runs for each matrix size.

| Matrix Size | CPU (ms) | GPU Kernel (ms) | H2D + D2H (ms) | End-to-End Speedup |
|---:|---:|---:|---:|---:|
| 256 | 2.829 | 0.113 | 0.397 | 5.546x |
| 1024 | 214.517 | 4.794 | 4.907 | 22.113x |
| 4096 | 20472.542 | 317.206 | 77.207 | 51.906x |


### CUDA Crossover Result

Among the tested matrix sizes, 256 × 256 was the smallest size where GPU end-to-end execution was faster than the CPU implementation.
