# DATA 266 - Homework 7 Metrics

## Experiment Configuration

| Parameter | Value |
|---|---|
| SID4 | 1598 |
| Random seed | 1598 |
| Dataset | FashionMNIST |
| Training images | 60000 |
| Test images | 10000 |
| Framework | PyTorch 2.11.0+cpu |
| Device | cpu |
| Batch size | 128 |
| Latent dimension | 32 |
| Epochs | 15 |
| Optimizer | Adam |
| Learning rate | 0.001 |

## Model Comparison

| Metric | FC-AE | VAE |
|---|---:|---:|
| Trainable parameters | 222384 | 224464 |
| Epochs | 15 | 15 |
| Training objective | MSE | BCE + KL |
| Final training loss | 0.013095 | 243.907375 |
| Final test loss (training objective) | 0.013123 | 245.4949 |
| Test reconstruction MSE (deterministic) | 0.013123 | 0.017353 |
| Training time (seconds) | 191.43 | 208.80 |

## VAE Loss Components

| Metric | Final Value |
|---|---:|
| Reconstruction BCE per image | 232.1943 |
| KL divergence per image | 11.7131 |
| Sampled test reconstruction MSE | 0.018722 |
| Deterministic test reconstruction MSE | 0.017353 |

## Interpretation

Both models were trained on the same FashionMNIST dataset with
the same batch size, latent dimension, learning rate, and epoch count.

FC-AE directly optimizes reconstruction MSE, while VAE optimizes
reconstruction BCE together with KL divergence.

For a consistent comparison, test reconstruction MSE was calculated
for both models. The VAE was evaluated using its latent mean to
avoid random sampling during final reconstruction evaluation.

The model with lower test MSE has better pixel-level reconstruction
accuracy under this metric. Training time measures the duration
of each model's training and per-epoch evaluation loop.

## Notes

- Test images were not used for gradient updates.
- Training objectives are different and should not be compared directly.
- Pixel values were scaled to [0, 1].
- No additional HP_ID experiment was required for Homework 7.
