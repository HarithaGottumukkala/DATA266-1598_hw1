# HW2.5 Metrics — Precision, Bandwidth, and the Cost of Attention

## GPU Under Test

| Property | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 |
| GPU UUID | `GPU-b60240a9-31a3-126f-751f-afc1dcc1d73f` |
| Reported VRAM | 24564 MiB |
| Configured Power Limit | 450 W |
| PyTorch | 2.11.0+cu128 |
| PyTorch CUDA Runtime | 12.8 |
| BF16 Support | Yes |
| Experiment Date | 2026-09-10 |

All measurements reported in this file were produced on the GPU identified by the UUID above.

Detailed commands, evidence files, and run-to-result traceability are recorded in `RUN_LOG.txt`.

---

# Table HW2.5.1 — Summary

| Measurement | Your GPU — RTX 4090 | Notes / Traceability |
|---|---:|---|
| **Peak achieved TFLOPS (BF16)** | **164.69 TFLOPS** | Peak BF16 matrix multiplication throughput at `N=8192`. Run ID: `B-4090-MATMUL` |
| **% of theoretical peak (BF16)** | **99.69%** | Measured 164.6871 TFLOPS relative to the 165.2 TFLOPS theoretical reference used for this experiment. Run ID: `B-4090-MATMUL` |
| **Effective bandwidth (GB/s)** | **920.74 GB/s** | Equivalent to 91.34% of the 1008 GB/s specified bandwidth reference. Run ID: `C-4090-ROOFLINE` |
| **Naive attention OOM length** | **No clean OOM observed; largest safely tested success = 18,944** | At `L=16384`, `nvidia-smi` reached 23945 MiB of 24564 MiB. Windows WDDM allowed memory oversubscription/paging, so testing was stopped before deliberately forcing excessive paging. Run ID: `D-4090-NAIVE` |
| **Fused / memory-efficient attention OOM length** | **No clean OOM observed; largest safely tested success = 1,572,864** | PyTorch Efficient Attention reached 24.0 GiB peak allocated memory at the largest safe test. Run ID: `D-4090-EFFICIENT-OOM` |
| **Steady-state / peak throughput** | **160.11 / 168.96 TFLOPS = 94.76%** | Final 5-minute average compared with first-30-second peak during the 20-minute sustained BF16 workload. Run ID: `E-4090-SUSTAINED` |
| **Throttle onset (s, or none)** | **None observed** | GPU stayed close to the 450 W power limit. Maximum temperature was 77 C and steady-state temperature was approximately 75.97 C. Behavior was more consistent with power-limited boost than severe thermal throttling. Run ID: `E-4090-SUSTAINED` |

---

# Part B — Matrix Multiplication Results

The matrix multiplication benchmark tested four matrix sizes and four required numerical precisions.

- Matrix sizes: `1024`, `4096`, `8192`, `16384`
- Precisions: FP32, TF32, FP16, BF16
- Warm-up iterations: 5
- Timed repetitions: 20

## Measured Throughput

| Matrix Size | FP32 TFLOPS | TF32 TFLOPS | FP16 TFLOPS | BF16 TFLOPS |
|---:|---:|---:|---:|---:|
| 1024 | 37.45 | 41.12 | 70.27 | 80.66 |
| 4096 | 55.31 | 82.14 | 169.36 | 156.42 |
| 8192 | 54.99 | 88.00 | 162.22 | **164.69** |
| 16384 | 50.79 | 86.70 | 157.26 | 160.23 |

The highest BF16 throughput observed was:

**164.6871 TFLOPS at N = 8192**

Using the 165.2 TFLOPS theoretical BF16 reference used in this assignment analysis:

**BF16 efficiency = 99.69% of theoretical peak**

For small matrices such as `N=1024`, throughput was substantially lower because the workload was not large enough to fully utilize the GPU's compute resources.

As matrix size increased, the GPU reached a much clearer throughput plateau.

Evidence:

- `raw/matmul/matmul_4090.csv`
- `raw/matmul/matmul_4090_with_theoretical.csv`
- `raw/matmul/matmul_fullrun_4090.txt`
- `figures/matmul_throughput_4090.png`

Run ID:

`B-4090-MATMUL`

---

# Additional Lower-Precision Experiment — FP8

An additional FP8 experiment was performed using:

`torch.float8_e4m3fn`

with:

`torch._scaled_mm`

## FP8 Results

| Matrix Size | FP8 Throughput |
|---:|---:|
| 1024 | 48.86 TFLOPS |
| 4096 | 294.34 TFLOPS |
| 8192 | 319.19 TFLOPS |
| 16384 | **335.89 TFLOPS** |

The highest measured FP8 throughput was:

**335.8943 TFLOPS at N = 16384**

Quantization/conversion time was excluded from the timed matrix multiplication measurement.

This experiment was included as the lower-precision extension beyond the required FP32, TF32, FP16, and BF16 measurements.

Evidence:

- `raw/matmul/fp8_capability_4090.txt`
- `raw/matmul/fp8_probe_4090.txt`
- `raw/matmul/fp8_benchmark_4090.csv`
- `raw/matmul/fp8_benchmark_run_4090.txt`

Run ID:

`B-4090-FP8`

---

# Part C — Bandwidth and Roofline Analysis

Two workloads were used to distinguish memory-bound and compute-bound behavior.

## Memory-Bound Workload

A large FP32 elementwise-add operation was used as the memory-bandwidth workload.

Measured effective bandwidth:

**920.74 GB/s**

Specified RTX 4090 bandwidth reference:

**1008 GB/s**

Percentage of specified bandwidth:

**91.34%**

Arithmetic intensity:

**0.08333 FLOPs/byte**

This arithmetic intensity is far below the measured roofline ridge point, so the workload is classified as:

**Memory-bound**

---

## Compute-Bound Workload

An FP32 matrix multiplication with `N=8192` was used as the compute-intensive workload.

Arithmetic intensity:

**1365.33 FLOPs/byte**

Measured roofline ridge point:

**81.94 FLOPs/byte**

Since:

`1365.33 >> 81.94`

the matrix multiplication lies on the compute-bound side of the roofline model.

Evidence:

- `raw/bandwidth/roofline_4090.csv`
- `raw/bandwidth/roofline_run_4090.txt`

Run ID:

`C-4090-ROOFLINE`

---

# Part D — Naive Attention

The naive scaled dot-product attention implementation used:

- Batch size: 4
- Number of heads: 8
- Head dimension: 64
- Precision: BF16
- Dropout: 0
- Causal attention: False

The implementation explicitly materialized the full attention matrix with shape:

`[B, H, L, L]`

This causes memory consumption to increase approximately quadratically as sequence length increases.

## Naive Attention Measurements

| Sequence Length | Peak Allocated Memory | Median Latency | Status |
|---:|---:|---:|---|
| 512 | 0.047 GiB | 0.111 ms | SUCCESS |
| 1024 | 0.149 GiB | 0.438 ms | SUCCESS |
| 2048 | 0.539 GiB | 1.879 ms | SUCCESS |
| 4096 | 2.070 GiB | 7.356 ms | SUCCESS |
| 8192 | 8.133 GiB | 28.978 ms | SUCCESS |
| 16384 | 32.258 GiB | 892.224 ms | SUCCESS |

At `L=16384`, PyTorch reported approximately:

**32.258 GiB peak allocated memory**

This value is larger than the physical VRAM capacity reported by the GPU.

A simultaneous `nvidia-smi` memory monitor showed a maximum physical GPU-memory usage of:

**23945 MiB / 24564 MiB**

The difference indicates that the Windows WDDM environment was allowing memory oversubscription and/or paging.

Therefore, PyTorch allocation statistics above physical VRAM should not be interpreted as 32 GiB physically resident on the GPU.

---

# Naive Attention Memory Growth

The fitted memory model was:

`M(L) = aL^2 + bL + c`

Measured quadratic coefficient:

**a = 1.19209290e-07 GiB/token²**

The fit produced:

**R² = 1.0000**

This provides strong experimental evidence that the memory requirement of the explicit naive implementation grows quadratically with sequence length.

The quadratic behavior comes from explicitly materializing the `L x L` attention score and probability matrices.

Evidence:

- `raw/attention/naive_attention_memory_fit_4090.txt`
- `raw/attention/naive_attention_memory_fit_run_4090.txt`
- `figures/naive_attention_memory_fit_4090.png`

Run ID:

`D-4090-NAIVE-MEMORY-FIT`

---

# Naive Attention OOM Boundary

Additional larger sequence lengths were tested after the required sweep.

The largest safely tested successful sequence length was:

**L = 18,944**

A clean CUDA OOM was not observed.

This should not be interpreted as unlimited GPU memory.

The Windows WDDM environment allowed oversubscription/paging, and the required `L=16384` experiment had already pushed physical GPU usage to approximately:

**23945 MiB / 24564 MiB**

Testing was therefore stopped rather than deliberately generating excessive host-memory paging.

For Table HW2.5.1, the result is reported as:

**No clean OOM observed; largest safely tested success = 18,944**

Run ID:

`D-4090-NAIVE`

---

# Part D — Memory-Efficient Attention

## Flash Attention Backend Check

The first attempt explicitly requested:

`FLASH_ATTENTION`

That attempt failed with:

`RuntimeError: No available kernel. Aborting execution.`

A backend capability probe was then performed.

| Backend | Result |
|---|---|
| FLASH_ATTENTION | FAILED |
| EFFICIENT_ATTENTION | SUCCESS |
| CUDNN_ATTENTION | SUCCESS |
| MATH | SUCCESS |

Therefore, the optimized experiment used:

**PyTorch EFFICIENT_ATTENTION**

rather than silently falling back to the normal math backend.

Evidence:

- `raw/attention/sdpa_backend_probe_4090.txt`
- `raw/attention/attention_fused_coarse_4090.csv`
- `raw/attention/attention_fused_coarse_run_4090.txt`

Run ID:

`D-4090-FLASH-PROBE`

---

# Memory-Efficient Attention Results

| Sequence Length | Peak Allocated Memory | Median Latency | Status |
|---:|---:|---:|---|
| 512 | 0.0078 GiB | 0.127 ms | SUCCESS |
| 1024 | 0.0156 GiB | 0.139 ms | SUCCESS |
| 2048 | 0.0313 GiB | 0.352 ms | SUCCESS |
| 4096 | 0.0625 GiB | 1.181 ms | SUCCESS |
| 8192 | 0.1250 GiB | 4.741 ms | SUCCESS |
| 16384 | **0.2500 GiB** | **17.264 ms** | SUCCESS |

At `L=16384`:

- Naive peak allocated memory: approximately **32.258 GiB**
- Efficient peak allocated memory: **0.250 GiB**

This is a dramatic reduction in temporary memory use.

The efficient implementation avoids explicitly materializing the complete `L x L` attention matrix in global GPU memory.

Evidence:

- `raw/attention/attention_fused_efficient_coarse_4090.csv`
- `raw/attention/attention_fused_efficient_coarse_run_4090.txt`

Run ID:

`D-4090-EFFICIENT`

---

# Attention Speedup

The speedup was calculated as:

`Speedup = Naive median latency / Efficient median latency`

| Sequence Length | Approximate Speedup |
|---:|---:|
| 512 | 0.87x |
| 1024 | 3.15x |
| 2048 | 5.33x |
| 4096 | 6.23x |
| 8192 | 6.11x |
| 16384 | **51.68x** |

At the smallest sequence length, the optimized implementation was slightly slower because kernel/backend overhead was significant relative to the small amount of computation.

As sequence length increased, the benefit became much larger.

At `L=16384`, the efficient implementation was approximately:

**51.68x faster than the naive implementation**

for the measured median latency.

Evidence:

- `raw/attention/attention_comparison_4090.csv`
- `raw/attention/attention_comparison_run_4090.txt`
- `figures/attention_speedup_4090.png`

Run ID:

`D-4090-ATTENTION-COMPARISON`

---

# Memory-Efficient Attention OOM Boundary

The efficient implementation allowed dramatically larger sequence lengths.

| Sequence Length | Peak Allocated Memory | Status |
|---:|---:|---|
| 32768 | 0.5 GiB | SUCCESS |
| 65536 | 1.0 GiB | SUCCESS |
| 131072 | 2.0 GiB | SUCCESS |
| 262144 | 4.0 GiB | SUCCESS |
| 524288 | 8.0 GiB | SUCCESS |
| 1048576 | 16.0 GiB | SUCCESS |
| 1572864 | **24.0 GiB** | SUCCESS |

The largest safely tested sequence length was:

**L = 1,572,864**

with:

**24.0 GiB peak allocated memory**

No clean OOM occurred within the safely tested range.

Testing was stopped at this point because the reported allocation was already approximately equal to the physical VRAM capacity, and the Windows WDDM environment had previously demonstrated oversubscription/paging behavior.

For Table HW2.5.1, this result is therefore reported as:

**No clean OOM observed; largest safely tested success = 1,572,864**

Evidence:

- `raw/attention/efficient_oom_coarse_boundary_4090.csv`
- `raw/attention/efficient_oom_coarse_boundary_run_4090.txt`
- `raw/attention/efficient_oom_large_boundary_4090.csv`
- `raw/attention/efficient_oom_large_boundary_run_4090.txt`

Run ID:

`D-4090-EFFICIENT-OOM`

---

# Part E — 20-Minute Sustained Compute

A sustained BF16 matrix multiplication workload was executed for approximately 20 minutes.

Configuration:

- Matrix size: `8192 x 8192`
- Precision: BF16
- Requested duration: 1200 seconds
- Telemetry interval: 5 seconds
- Telemetry samples: 238

The experiment recorded:

- GPU SM clock
- GPU memory clock
- GPU temperature
- GPU power draw
- Power limit
- GPU utilization
- GPU memory usage
- Sustained computation throughput

---

## First 30 Seconds

| Metric | Value |
|---|---:|
| Peak throughput | **168.96 TFLOPS** |
| Average throughput | 162.76 TFLOPS |
| Average SM clock | 2537.50 MHz |
| Average temperature | 55.67 C |
| Average power | 377.17 W |

---

## Final Five Minutes

| Metric | Value |
|---|---:|
| Average throughput | **160.11 TFLOPS** |
| Peak throughput | 161.96 TFLOPS |
| Average SM clock | 2488.00 MHz |
| Average temperature | 75.97 C |
| Average power | 449.55 W |
| Average GPU utilization | 99.98% |

---

## Full-Run Thermal Summary

| Metric | Value |
|---|---:|
| Maximum temperature | **77.00 C** |
| Average power | 447.70 W |
| Maximum power | 451.02 W |
| Configured power limit | 450.00 W |
| Throughput change | **-5.24%** |
| SM clock change | **-1.95%** |

The final five-minute average throughput relative to the initial 30-second peak was:

`160.11 / 168.96 = 94.76%`

Therefore:

**Steady-state / peak throughput = 94.76%**

---

# Throttling Analysis

No clear thermal-throttle onset was observed.

The GPU reached a maximum temperature of only:

**77 C**

while the final five-minute average temperature was:

**75.97 C**

At the same time, average steady-state power was:

**449.55 W**

against a configured power limit of:

**450 W**

The average SM clock decreased from:

**2537.50 MHz**

during the initial period to:

**2488.00 MHz**

during the final five minutes.

This represents a relatively small:

**1.95% reduction in average SM clock**

while power remained almost exactly at the configured limit.

Therefore, the measured steady-state behavior is more consistent with:

**power-limited GPU boost behavior**

rather than severe thermal throttling.

For Table HW2.5.1:

**Throttle onset = none observed**

Evidence:

- `raw/thermal/sustained_compute_run_4090.txt`
- `raw/thermal/thermal_4090.csv`
- `raw/thermal/throughput_4090.csv`
- `raw/thermal/thermal_analysis_4090.txt`
- `raw/thermal/thermal_analysis_run_4090.txt`
- `figures/thermal_clock_4090.png`
- `figures/thermal_temperature_4090.png`

Run ID:

`E-4090-SUSTAINED`

---

# Final RTX 4090 Observations

The experiments show several clear behaviors.

First, large BF16 matrix multiplication was able to reach approximately **164.69 TFLOPS**, which corresponds to approximately **99.69%** of the theoretical reference used in this analysis.

Second, the memory-bandwidth benchmark reached approximately **920.74 GB/s**, or **91.34%** of the specified bandwidth reference.

Third, naive attention showed clear quadratic memory growth. At `L=16384`, PyTorch reported approximately **32.26 GiB** peak allocated memory, while physical GPU memory usage measured through `nvidia-smi` was already close to the RTX 4090's available VRAM.

The memory-efficient attention implementation changed this behavior substantially. At the same `L=16384`, peak allocated memory was only **0.25 GiB**, and the measured latency was approximately **51.68x faster** than the naive implementation.

Finally, during the 20-minute sustained workload, the RTX 4090 maintained approximately **94.76%** of its initial peak throughput. Temperature stabilized around 76 C while power remained close to the 450 W limit. No clear thermal-throttle onset was observed.

---

# Part F Traceability

Every value in Table HW2.5.1 is linked to a UUID-labelled run in `RUN_LOG.txt`.

| Table HW2.5.1 Measurement | RUN_LOG Run ID |
|---|---|
| Peak achieved TFLOPS (BF16) | `B-4090-MATMUL` |
| % of theoretical peak (BF16) | `B-4090-MATMUL` |
| Effective bandwidth | `C-4090-ROOFLINE` |
| Naive attention OOM length | `D-4090-NAIVE` |
| Fused / memory-efficient attention OOM length | `D-4090-EFFICIENT-OOM` |
| Steady-state / peak throughput | `E-4090-SUSTAINED` |
| Throttle onset | `E-4090-SUSTAINED` |

GPU UUID for the runs above:

`GPU-b60240a9-31a3-126f-751f-afc1dcc1d73f`

---

# Generated Figures

The following figures are included with the submission:

- `figures/matmul_throughput_4090.png`
- `figures/naive_attention_memory_fit_4090.png`
- `figures/attention_speedup_4090.png`
- `figures/thermal_clock_4090.png`
- `figures/thermal_temperature_4090.png`

---

# Evidence Files

The detailed raw measurements are stored under:

- `raw/gpu_info/`
- `raw/matmul/`
- `raw/bandwidth/`
- `raw/attention/`
- `raw/thermal/`

The complete experiment trace is documented in:

`RUN_LOG.txt`
