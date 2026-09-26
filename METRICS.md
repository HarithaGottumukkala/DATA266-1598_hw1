# DATA 266 — LoRA Fine-Tuning Metrics

## Experiment Setup

- Model: google/flan-t5-small
- Dataset: DialogSum
- Seed: 1598
- Training examples: 1999
- Validation examples: 499
- Epochs: 2
- Batch size: 8
- Gradient accumulation steps: 2
- Learning rate: 2e-4
- Precision: FP32
- Target modules: q, v

## LoRA Rank Comparison

| Metric | r = 4 | r = 16 |
|---|---:|---:|
| Total Parameters | 77,133,184 | 77,649,280 |
| Trainable Parameters | 172,032 | 688,128 |
| Trainable Percentage | 0.2230% | 0.8862% |
| Pre-training Batch Loss | 2.2771 | 2.2771 |
| Average Training Loss | 3.4834 | 3.4987 |
| Validation Loss | 1.4146 | 1.4198 |
| Training Time (minutes) | 2.70 | 2.79 |

## Output Comparison

The same two DialogSum test examples were used for the baseline, `r = 4`, and `r = 16` models.

The LoRA fine-tuned models produced more task-specific summaries than the original baseline on the tested examples. Increasing the rank from 4 to 16 increased adapter capacity, but it did not produce a clear improvement in summary quality for these examples.
