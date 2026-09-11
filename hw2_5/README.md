# HW2.5 — Precision, Bandwidth, and the Cost of Attention

This repository contains the code, raw measurements, figures, and analysis for DATA 266 Homework 2.5.

The experiments study GPU performance across numerical precision, memory bandwidth, attention memory growth, optimized attention, and sustained thermal behavior.

---

## GPU Used

- GPU: NVIDIA GeForce RTX 4090
- GPU UUID: `GPU-b60240a9-31a3-126f-751f-afc1dcc1d73f`
- Reported VRAM: 24564 MiB
- Power Limit: 450 W
- Architecture: Ada Lovelace
- PyTorch: 2.11.0+cu128
- PyTorch CUDA Runtime: 12.8
- BF16 Support: Yes

Detailed GPU information is available in:

`raw/gpu_info/`

---

## Part A — GPU Information

The GPU environment was verified using:

- `nvidia-smi`
- `nvidia-smi -q`
- PyTorch CUDA checks
- NVCC version checks

The full `nvidia-smi -q` output is stored in:

`raw/gpu_info/nvidia_smi_4090.txt`

---

## Part B — Matrix Multiplication

Matrix multiplication was benchmarked using:

- FP32
- TF32
- FP16
- BF16

Matrix sizes:

- 1024
- 4096
- 8192
- 16384

The highest measured BF16 throughput was:

**164.69 TFLOPS at N = 8192**

This corresponds to approximately:

**99.69% of the 165.2 TFLOPS theoretical reference used in the analysis**

An additional FP8 experiment was also performed.

The highest measured FP8 result was:

**335.89 TFLOPS at N = 16384**

Results are stored in:

`raw/matmul/`

Main figure:

`figures/matmul_throughput_4090.png`

---

## Part C — Bandwidth and Roofline

A large FP32 elementwise-add workload was used as the memory-bound benchmark.

Measured effective memory bandwidth:

**920.74 GB/s**

This is approximately:

**91.34% of the 1008 GB/s specified bandwidth reference**

Arithmetic intensity results:

- Elementwise addition: `0.08333 FLOPs/byte`
- Roofline ridge point: `81.94 FLOPs/byte`
- FP32 matrix multiplication: `1365.33 FLOPs/byte`

Therefore:

- Elementwise addition was memory-bound.
- Matrix multiplication was compute-bound.

Results are stored in:

`raw/bandwidth/`

---

## Part D — Attention

The attention experiments used:

- Batch size: 4
- Number of heads: 8
- Head dimension: 64
- Precision: BF16

### Naive Attention

The naive implementation explicitly materialized the full attention matrix.

At sequence length `L = 16384`:

- Peak allocated memory: approximately **32.26 GiB**
- Median latency: approximately **892.22 ms**

The measured memory growth followed a strong quadratic relationship with sequence length.

Largest safely tested naive sequence length:

**18,944**

A clean CUDA OOM was not observed because the Windows WDDM environment allowed memory oversubscription/paging.

---

### Memory-Efficient Attention

The initial forced Flash Attention backend was unavailable in the installed PyTorch environment.

Backend testing showed:

- FLASH_ATTENTION: Failed
- EFFICIENT_ATTENTION: Success
- CUDNN_ATTENTION: Success
- MATH: Success

The optimized benchmark therefore used:

**PyTorch EFFICIENT_ATTENTION**

At `L = 16384`:

- Peak allocated memory: **0.25 GiB**
- Median latency: approximately **17.26 ms**

This was approximately:

**51.68x faster than the naive implementation at L = 16384**

Largest safely tested memory-efficient sequence length:

**1,572,864**

Peak allocated memory at that size:

**24.0 GiB**

Results are stored in:

`raw/attention/`

Figures:

- `figures/naive_attention_memory_fit_4090.png`
- `figures/attention_speedup_4090.png`

---

## Part E — Sustained Compute and Thermal Behavior

A BF16 matrix multiplication workload was executed for approximately 20 minutes.

Configuration:

- Matrix size: 8192 x 8192
- Precision: BF16
- Duration: 1200 seconds
- Telemetry interval: approximately 5 seconds

Key results:

- First 30-second peak throughput: **168.96 TFLOPS**
- Final 5-minute average throughput: **160.11 TFLOPS**
- Steady-state / peak ratio: **94.76%**
- Maximum GPU temperature: **77 C**
- Final 5-minute average temperature: **75.97 C**
- Final 5-minute average power: **449.55 W**
- Power limit: **450 W**
- Final average GPU utilization: **99.98%**

No clear thermal-throttle onset was observed.

The small reduction in sustained clock and throughput was more consistent with power-limited GPU boost behavior than severe thermal throttling.

Results are stored in:

`raw/thermal/`

Figures:

- `figures/thermal_clock_4090.png`
- `figures/thermal_temperature_4090.png`

---

## Part F — Summary

The final summary table is available in:

`METRICS.md`

Detailed UUID-labelled experiment traceability is available in:

`RUN_LOG.txt`

GPU reservation and usage information is available in:

`GPU_HOURS.md`

AI usage disclosure is available in:

`AI_USE.md`

---

## Repository Structure

```text
hw2_5/
│
├── README.md
├── METRICS.md
├── RUN_LOG.txt
├── GPU_HOURS.md
├── AI_USE.md
│
├── src/
│   ├── benchmark_matmul.py
│   ├── benchmark_fp8.py
│   ├── benchmark_roofline.py
│   ├── benchmark_attention.py
│   ├── attention_oom_probe.py
│   ├── efficient_attention_oom_probe.py
│   ├── probe_sdpa_backends.py
│   ├── compare_attention.py
│   ├── fit_attention_memory.py
│   ├── sustained_compute.py
│   ├── analyze_thermal.py
│   └── plot_matmul.py
│
├── raw/
│   ├── gpu_info/
│   ├── matmul/
│   ├── bandwidth/
│   ├── attention/
│   ├── thermal/
│   └── screenshots/
│
└── figures/
