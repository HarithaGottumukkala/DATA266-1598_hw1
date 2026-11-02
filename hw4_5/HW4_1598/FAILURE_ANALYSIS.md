# HW4.a and HW4.5 — Failure and Error Analysis

## Introduction

While working on this assignment, I faced several problems related to dataset loading, GPU memory, model evaluation, and performance measurement.

These problems were useful because they showed me that deep learning experiments depend not only on the model but also on the data, available hardware, library versions, and benchmarking method.

This document explains the main problems, how I handled them, and what I learned.

## 1. TweetEval Dataset Loading Error

**What happened:**

The first attempt to load the TweetEval dataset failed.

**Error:**

`ValueError: BuilderConfig 'sentiment' not found. Available: ['default']`

**Why it happened:**

The code requested a dataset configuration that was not available in the pinned dataset revision.

**What I changed:**

I loaded the sentiment training Parquet file directly from the fixed Hugging Face revision.

**Result:**

The corrected method loaded 45,615 source rows. I then selected 5,000 consecutive tweets from rows 598 to 5,597 and created the required 70/15/15 data split.

**What I learned:**

Dataset loading code can break when the available configurations or library behavior change. I should always verify the loaded dataset, columns, row count, and selected sample range.

## 2. Out-of-Memory Errors During Batch-Size Scaling

**What happened:**

I tested increasing physical batch sizes on the NVIDIA A100 and NVIDIA L4.

Both GPUs successfully completed the tested FP32 batch size of 256. Batch size 512 caused out-of-memory errors.

**Why it happened:**

Larger batches require more GPU memory for input tensors, intermediate activations, gradients, and other training operations.

**What I changed:**

I recorded the successful batch sizes and treated the OOM results as the stopping point for the powers-of-two sweep.

**Result:**

The largest successfully tested FP32 batch size was 256 on both GPUs.

At batch size 256, the A100 achieved approximately 247.19 samples/second, while the L4 achieved approximately 124.91 samples/second.

**What I learned:**

More GPU memory does not automatically mean a larger successful batch size in every tested sweep. Batch capacity also depends on the model, training settings, memory allocation, and available memory at the time of the run.

## 3. GPU Memory Contamination During the 8-bit Optimizer Test

**What happened:**

The initial 8-bit optimizer experiment failed at physical batch size 256.

**Why it happened:**

The GPU still had approximately 8.642 GiB of memory allocated before the clean benchmark. Objects remaining from previous experiments affected the available memory.

**What I changed:**

I restarted or cleaned the runtime and repeated the 8-bit optimizer experiment under cleaner conditions.

**Result:**

The clean experiment successfully processed batch size 256 and encountered OOM at batch size 512.

The clean batch-256 result was approximately 126.96 samples/second with 20.03 GiB of reported peak memory.

**What I learned:**

A fair GPU benchmark should begin from a controlled memory state. Otherwise, memory left over from earlier experiments can make the next technique look worse than it really is.

The final comparison uses the clean 8-bit results, while the original attempt is retained as evidence of the problem.

## 4. Gradient Checkpointing Reduced Speed

**What happened:**

Gradient checkpointing reduced memory use but also reduced training throughput.

**Why it happened:**

Checkpointing saves memory by keeping fewer intermediate activations. During backpropagation, some of these activations must be calculated again.

**Measured result at physical batch size 256:**

| Metric | L4 FP32 baseline | Gradient checkpointing |
|---|---:|---:|
| Throughput | 124.91 samples/s | 90.10 samples/s |
| Peak memory | 20.65 GiB | 6.48 GiB |

**Result:**

Gradient checkpointing saved a large amount of memory but came with a performance cost. It also allowed a larger tested physical batch size of 1,024.

**What I learned:**

The configuration that uses the least memory is not necessarily the fastest. I need to decide whether memory capacity or training speed is more important for the task.

## 5. FP16 Evaluation Used After FP32 Checkpointing Training

**What happened:**

My gradient-checkpointing model was trained with FP32, but the original evaluation function used FP16 autocast.

**Why it mattered:**

This introduced a difference in evaluation precision when comparing the result against an FP32 baseline.

**What I changed:**

I loaded the saved best checkpoint and evaluated it again without autocast.

**Result:**

- Original FP16 evaluation accuracy: 68.53%
- Corrected FP32 evaluation accuracy: 68.53%
- Corrected FP32 test loss: 0.8811

**What I learned:**

Even when accuracy does not change, it is important to document the exact evaluation settings and keep comparisons consistent.

## 6. Training Beyond the Best Validation Epoch

**What happened:**

In the original BERT fine-tuning experiment, validation accuracy stopped improving after epoch 2, but training continued through epoch 4.

**Observed values:**

- Best validation accuracy: 69.07%, at epoch 2
- Epoch 3 validation accuracy: 68.00%
- Epoch 4 validation accuracy: 67.47%
- Additional wall-clock cost after the best epoch: 39.00 seconds

**Why it happened:**

The experiment was configured to train for four epochs.

**Result:**

Training loss continued decreasing, but validation accuracy did not improve after epoch 2. This is consistent with increasing overfitting risk.

**What I learned:**

A lower training loss does not always mean better performance on unseen data. Validation monitoring and checkpoint selection are important.

## 7. vLLM Environment Warnings

**What happened:**

The vLLM environment produced dependency-conflict warnings during installation.

The Hugging Face generation pipeline also warned that both `max_length` and `max_new_tokens` were provided.

**What I changed:**

I checked the environment and continued only after vLLM could import and perform inference. The generation pipeline explicitly used `max_new_tokens=64`.

**Result:**

The serving experiment completed and produced throughput and latency measurements.

**What I learned:**

Library compatibility and generation settings should be checked carefully before comparing model-serving performance.

## 8. Different Latency Measurement Methods

**What happened:**

The original HW4.a experiment reported per-request latency for serving, while the later L4 experiment reported offline batch-completion durations.

**Why it mattered:**

The two measurements describe different things and cannot be compared as if they were identical request-latency benchmarks.

**What I changed:**

I documented the measurement difference and kept the original HW4.a results as historical reference values.

**Result:**

The L4 batch-completion benchmark was retained, but its percentile values were not presented as directly equivalent to the HW4.a per-request percentiles.

**What I learned:**

Performance numbers are meaningful only when the timing method, batch size, hardware, and inference settings are clearly explained.

## 9. KV-Cache and Context-Length Testing Limitation

**What happened:**

The L4 context-length experiment successfully tested prompts up to 1,984 tokens with 16 generated tokens.

The vLLM engine had a configured maximum model length of 2,048 tokens.

**Why it mattered:**

Successfully reaching the configured context range does not prove that the hardware cannot support a longer context under different settings.

Also, estimated KV-cache demand is not the same as independently measured KV-cache allocation.

**What I changed:**

I reported the longest context actually tested and described the engine configuration and measurement limitations.

**What I learned:**

I should not call a tested software limit an absolute hardware limit unless the experiment directly proves it.

## 10. Incomplete Hardware Coverage for Part D

**What happened:**

The HW4.5 serving experiment was completed on the L4, but a fresh matching HW4.5 serving run on the A100 was not completed.

**Why it mattered:**

The original HW4.a A100 serving results used a different experiment configuration and measurement procedure.

**What I changed:**

I kept the original A100 serving results as a reference and disclosed the missing matching measurement.

**What I learned:**

A meaningful GPU comparison needs the same workload and measurement method on both devices.

## 11. Part C — Optional Multi-GPU Experiment

Part C skipped (optional).

The optional Distributed Data Parallel experiment was not performed. No two-GPU speedup or scaling-efficiency results are claimed.

## Final Lessons

The biggest lesson from this assignment is that GPU performance is not determined by one number.

Batch size, GPU memory, mixed precision, checkpointing, optimizer choice, training accuracy, and inference settings all affect the final result.

I also learned that failed experiments should be documented instead of hidden. An out-of-memory error, an incorrect dataset-loading method, or an unfair benchmark can help explain how to improve the experimental setup.

Most importantly, I learned to use real measurements, check my code carefully, and explain the limitations of any comparison.
