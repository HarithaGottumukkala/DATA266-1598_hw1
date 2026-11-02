# HW4.a Metrics

## Personal Parameters

| Parameter | Value |
|---|---:|
| SID4 | 1598 |
| SEED | 1598 |
| SLICE | 598 |
| HP_ID | 2 |
| CLS_A | 8 |
| CLS_B | 5 |

## Hardware

- Platform: Google Colab Pro
- GPU: NVIDIA A100-SXM4-40GB
- Available GPU memory: 39.49 GB

## Dataset

- Dataset: `cardiffnlp/tweet_eval`
- Configuration: sentiment
- Source split: original training split
- Source rows: 598 through 5,597, inclusive
- Selected rows: 5,000
- Training split: 3,500
- Validation split: 750
- Test split: 750
- Split seed: 1598

## BERT Baseline Configuration

| Setting | Value |
|---|---|
| Model | `google-bert/bert-base-uncased` |
| Precision | FP32 |
| Maximum sequence length | 128 |
| Physical batch size | 16 |
| Epochs | 4 |
| Learning rate | 0.00002 |
| Weight decay | 0.01 |
| Warmup ratio | 0.10 |
| Optimizer | AdamW |

## Per-Epoch Training Results

| Epoch | Training loss | Validation loss | Validation accuracy | Training time (s) | Samples/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.9124 | 0.7528 | 65.87% | 19.72 | 177.50 |
| 2 | 0.6196 | 0.7518 | 69.07% | 18.27 | 191.60 |
| 3 | 0.3898 | 0.8437 | 68.00% | 18.24 | 191.87 |
| 4 | 0.2325 | 0.9424 | 67.47% | 18.30 | 191.26 |

- Best validation epoch: 2
- Extra epochs after best epoch: 2
- Extra wall-clock cost after best epoch: 39.00 seconds
- Total fine-tuning wall-clock time: 79.49 seconds
- Peak allocated GPU memory: 2.478 GB
- Peak reserved GPU memory: 2.863 GB

These memory and time values are the HW4 baseline for HW4.5 only.

## Final Four-Epoch Test Results

- Test loss: 0.9031
- Test accuracy: 69.47%
- Macro F1-score: 68.27%
- Weighted F1-score: 69.50%

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Negative | 62.30% | 64.96% | 63.60% | 117 |
| Neutral | 68.70% | 72.09% | 70.35% | 344 |
| Positive | 73.78% | 68.17% | 70.86% | 289 |

### Confusion Matrix

| Actual \ Predicted | Negative | Neutral | Positive |
|---|---:|---:|---:|
| Negative | 76 | 31 | 10 |
| Neutral | 36 | 248 | 60 |
| Positive | 10 | 82 | 197 |

Worst class by recall: negative.

## Serving Benchmark

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Precision: bfloat16
- GPU: NVIDIA A100-SXM4-40GB
- Maximum generated tokens: 64
- Measured requests per configuration: 160

| Configuration | Tokens/sec | p50 latency (ms) | p95 latency (ms) |
|---|---:|---:|---:|
| HF pipeline, batch 1 | 30.31 | 1330.43 | 1690.91 |
| vLLM, batch 1 | 227.67 | 162.47 | 231.36 |
| vLLM, batch 8 | 1353.25 | 205.82 | 280.97 |
| vLLM, batch 32 | 3886.88 | 310.88 | 313.20 |

No tested vLLM batch size had latency worse than the Hugging Face baseline. Batch size 32 had the highest vLLM throughput, while batch size 1 had the lowest vLLM latency.
