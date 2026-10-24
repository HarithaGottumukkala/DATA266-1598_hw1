# Homework 8 - Metrics Report

## Experiment Configuration

| Setting | Value |
|---|---|
| Dataset | MNIST |
| Training images | 60000 |
| Test images | 10000 |
| Batch size | 128 |
| Diffusion steps | 100 |
| Beta start | 0.0001 |
| Beta end | 0.0200 |
| CNN convolution layers | 6 |
| Total parameters | 355,009 |
| Trainable parameters | 355,009 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss | Mean Squared Error |
| Training epochs | 50 |

## Training Metrics

| Metric | Value |
|---|---:|
| Initial MSE | 0.1775 |
| Final MSE | 0.0602 |
| Loss reduction | 66.07% |
| Lowest recorded MSE | 0.0601 |

## Sampling Metrics

All times are measured in seconds over five runs
after a warm-up. Results describe batches of 10 images.

| Method | Steps | Mean Time (s) | Std Dev (s) |
|---|---:|---:|---:|
| DDPM | 100 | 0.1836 | 0.0160 |
| DDIM | 50 | 0.1437 | 0.0639 |
| DDIM | 20 | 0.0413 | 0.0001 |

DDIM with 20 steps was 3.48 times
faster than DDIM with 50 steps in this benchmark.

## Generated Image Shapes

- DDPM: [10, 1, 28, 28]
- DDIM 50: [10, 1, 28, 28]
- DDIM 20: [10, 1, 28, 28]

## Evaluation Notes

Training MSE measures noise-prediction error.
It does not directly measure generated image quality.

No FID or generated-digit classification accuracy
was computed in this experiment.
