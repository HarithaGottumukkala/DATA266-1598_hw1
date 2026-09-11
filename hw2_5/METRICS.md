# HW2.5 Metrics — Precision, Bandwidth, and the Cost of Attention

## Personal Parameters

The standing DATA 266 parameters for this submission are:

| Parameter | Value |
|---|---:|
| SID4 | 1598 |
| SEED | 1598 |
| SLICE | 598 |
| HP_ID | 2 |
| CLS_A | 8 |
| CLS_B | 5 |

For HW2.5, the main experiments are hardware timing and memory measurements, so
`SLICE`, `HP_ID`, `CLS_A`, and `CLS_B` are not directly used by the benchmark
design.

The source files in `src/` now use the common `reproducibility.py` helper so
that reruns set Python, NumPy, and PyTorch random seeds to `1598`.

The reported measurements below are the original GPU measurements collected
during the lab session. The 20-minute sustained-compute run explicitly used
`torch.manual_seed(1598)` during the measured run. Hardware timing values are
not expected to reproduce exactly because clock, temperature, power, and
system conditions can vary between runs.

---

# GPU Under Test

| Property | Observed / Vendor Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 |
| GPU UUID | `GPU-b60240a9-31a3-126f-751f-afc1dcc1d73f` |
| Architecture | NVIDIA Ada Lovelace |
| Tensor Cores | 4th Generation |
| Reported VRAM | 24564 MiB |
| Vendor Frame Buffer | 24 GB GDDR6X |
| Memory Interface | 384-bit |
| Memory Data Rate | 21 Gbps |
| Specified Memory Bandwidth | 1008 GB/s |
| Configured Power Limit | 450 W |
| PyTorch | 2.11.0+cu128 |
| PyTorch CUDA Runtime | 12.8 |
| NVCC Toolkit | CUDA 12.8 |
| BF16 Available | Yes |

## Tensor Core Precision Support

The NVIDIA Ada architecture documentation describes Tensor Core support for
the following reduced-precision modes:

- TF32
- FP16
- BF16
- FP8
- INT8
- INT4

For the percentage-of-theoretical calculations in Part B, I used the dense
non-sparse theoretical references below.

| Precision | Theoretical Reference |
|---|---:|
| FP32 | 82.6 TFLOPS |
| TF32 Tensor | 82.6 TFLOPS |
| FP16 Tensor with FP32 accumulation | 165.2 TFLOPS |
| BF16 Tensor with FP32 accumulation | 165.2 TFLOPS |

NVIDIA also reports higher values when structured sparsity is used. I did not
use the sparse values because my matrix multiplication benchmark was dense.

### Vendor Sources

[1] NVIDIA Corporation, *NVIDIA Ada GPU Architecture*, Appendix A:
GeForce RTX 4090 Full Specifications.

[2] NVIDIA Corporation, *GeForce RTX 4090 Graphics Cards for Gaming*,
official product specifications.

The vendor information above is also recorded in:

`raw/gpu_info/vendor_specs_4090.txt`

---

# Part A — Onboarding and Provenance

The workstation was identified before running the performance experiments.

The complete `nvidia-smi -q` output is stored in:

`raw/gpu_info/nvidia_smi_4090.txt`

Additional environment evidence is stored in:

- `raw/gpu_info/nvidia_smi_summary_4090.txt`
- `raw/gpu_info/gpu_identity_4090.txt`
- `raw/gpu_info/pytorch_cuda_4090.txt`
- `raw/gpu_info/nvcc_4090.txt`
- `raw/gpu_info/preflight_4090.txt`
- `raw/gpu_info/vendor_specs_4090.txt`

GPU UUID used throughout Parts B–E:

`GPU-b60240a9-31a3-126f-751f-afc1dcc1d73f`

Reservation and GPU-hour information is recorded separately in:

`GPU_HOURS.md`

and:

`raw/reservations/reservation_4090.txt`

---

# Part B — Precision and Achieved Throughput

Dense square matrix multiplication was tested at:

- `N = 1024`
- `N = 4096`
- `N = 8192`
- `N = 16384`

Required precisions:

- FP32
- TF32
- FP16
- BF16

Each configuration used:

- 5 warm-up iterations
- 20 timed repetitions

The GPU was synchronized around the timed operations, and median timing was
used to reduce the effect of individual timing variations.

Run ID:

`B-4090-MATMUL`

GPU UUID:

`GPU-b60240a9-31a3-126f-751f-afc1dcc1d73f`

## Matrix Multiplication Measurements

| N | Precision | Achieved TFLOPS | Theoretical TFLOPS | % of Theoretical |
|---:|---|---:|---:|---:|
| 1024 | FP32 | 37.45 | 82.60 | 45.34% |
| 4096 | FP32 | 55.31 | 82.60 | 66.97% |
| 8192 | FP32 | 54.99 | 82.60 | 66.57% |
| 16384 | FP32 | 50.79 | 82.60 | 61.49% |
| 1024 | TF32 | 41.12 | 82.60 | 49.78% |
| 4096 | TF32 | 82.14 | 82.60 | 99.45% |
| 8192 | TF32 | **88.00** | 82.60 | **106.54%** |
| 16384 | TF32 | 86.70 | 82.60 | 104.96% |
| 1024 | FP16 | 70.27 | 165.20 | 42.54% |
| 4096 | FP16 | **169.36** | 165.20 | **102.52%** |
| 8192 | FP16 | 162.22 | 165.20 | 98.19% |
| 16384 | FP16 | 157.26 | 165.20 | 95.19% |
| 1024 | BF16 | 80.66 | 165.20 | 48.83% |
| 4096 | BF16 | 156.42 | 165.20 | 94.68% |
| 8192 | BF16 | **164.69** | 165.20 | **99.69%** |
| 16384 | BF16 | 160.23 | 165.20 | 96.99% |

The highest required BF16 measurement was:

**164.6871 TFLOPS at N = 8192**

This corresponds to:

**99.69% of the 165.2 TFLOPS dense BF16 theoretical reference**

---

## Throughput Plateau

The measured throughput did not keep increasing as matrix size increased.
Once the GPU had enough work to keep its execution resources busy, the
throughput entered a relatively stable high-performance region.

### FP32

FP32 reached its main plateau around `N=4096`.

The measured results were:

- `N=4096`: 55.31 TFLOPS
- `N=8192`: 54.99 TFLOPS

Increasing the matrix size further did not improve FP32 throughput.

### TF32

TF32 entered its high-throughput region at approximately `N=4096` and reached
its highest measured value at `N=8192`.

- `N=4096`: 82.14 TFLOPS
- `N=8192`: 88.00 TFLOPS
- `N=16384`: 86.70 TFLOPS

I therefore consider the TF32 plateau to begin around `N=4096`, with the best
measurement occurring at `N=8192`.

### FP16

FP16 reached its high-throughput region around `N=4096`.

- `N=4096`: 169.36 TFLOPS
- `N=8192`: 162.22 TFLOPS
- `N=16384`: 157.26 TFLOPS

Making the matrix larger did not produce higher throughput.

### BF16

BF16 approached its plateau at `N=4096` and reached its best measured result
at `N=8192`.

- `N=4096`: 156.42 TFLOPS
- `N=8192`: 164.69 TFLOPS
- `N=16384`: 160.23 TFLOPS

I therefore use approximately `N=8192` as the clearest BF16 plateau point.

---

## Why Small Matrices Do Not Reach Peak Throughput

The `N=1024` result was much lower for every precision.

A small matrix does not create enough parallel work to keep all GPU execution
resources busy for a long time. Fixed costs such as launching the kernel and
setting up the operation also take a larger fraction of the total runtime.

As matrix size increases, there is much more independent work available, so
the GPU can use more of its compute resources at the same time.

This is why the larger matrices came much closer to the theoretical
throughput of the RTX 4090.

---

## Why Some Measurements Are Slightly Above 100%

The TF32 measurements at `N=8192` and `N=16384`, and the FP16 measurement at
`N=4096`, were slightly above the nominal dense theoretical reference.

I did not clip or modify these measurements.

The theoretical numbers are fixed reference values, while the real GPU can
operate at different boost clocks depending on workload, power, temperature,
and the individual card. Timing variation also has an effect on short hardware
benchmarks.

For that reason, the measured values are reported exactly as observed even
when the calculated percentage is slightly above 100%.

Evidence:

- `raw/matmul/matmul_4090.csv`
- `raw/matmul/matmul_4090_with_theoretical.csv`
- `raw/matmul/matmul_fullrun_4090.txt`

Figure:

`figures/matmul_throughput_4090.png`

---

# Part B — Lower-Precision FP8 Experiment

The software stack exposed FP8 support, so I also tested a lower precision
instead of only documenting an unsupported attempt.

The experiment used:

`torch.float8_e4m3fn`

and:

`torch._scaled_mm`

## FP8 Measurements

| Matrix Size | Achieved FP8 TFLOPS |
|---:|---:|
| 1024 | 48.86 |
| 4096 | 294.34 |
| 8192 | 319.19 |
| 16384 | **335.89** |

The highest measured FP8 throughput was:

**335.8943 TFLOPS at N = 16384**

Quantization/conversion time was excluded from the timed matrix multiplication.
The output path used BF16 where required by the tested PyTorch operation.

Evidence:

- `raw/matmul/fp8_capability_4090.txt`
- `raw/matmul/fp8_probe_4090.txt`
- `raw/matmul/fp8_benchmark_4090.csv`
- `raw/matmul/fp8_benchmark_run_4090.txt`

Run ID:

`B-4090-FP8`

---

# Part C — Bandwidth-Bound vs. Compute-Bound

Two different workloads were used:

1. A large FP32 elementwise addition as the memory-bound operation.
2. A large FP32 matrix multiplication as the compute-bound operation.

Run ID:

`C-4090-ROOFLINE`

GPU UUID:

`GPU-b60240a9-31a3-126f-751f-afc1dcc1d73f`

## Effective Memory Bandwidth

Measured effective bandwidth:

**920.74 GB/s**

RTX 4090 specified bandwidth:

**1008 GB/s**

Percentage of specified bandwidth:

**91.34%**

---

## Arithmetic Intensity

For the elementwise-add workload:

**Arithmetic intensity = 0.08333 FLOPs/byte**

For the FP32 matrix multiplication:

**Arithmetic intensity = 1365.33 FLOPs/byte**

Measured roofline ridge point:

**81.94 FLOPs/byte**

The elementwise-add arithmetic intensity is far below the ridge point:

`0.08333 < 81.94`

so it is classified as:

**Memory-bound**

The matrix multiplication arithmetic intensity is far above the ridge point:

`1365.33 > 81.94`

so it is classified as:

**Compute-bound**

Evidence:

- `raw/bandwidth/roofline_4090.csv`
- `raw/bandwidth/roofline_run_4090.txt`

---

# Part D — Cost of Attention

The attention experiments used one fixed configuration:

| Parameter | Value |
|---|---:|
| Batch size | 4 |
| Number of heads | 8 |
| Head dimension | 64 |
| Precision | BF16 |
| Dropout | 0 |
| Causal | False |

---

# Naive Scaled Dot-Product Attention

The naive implementation explicitly computed and stored the complete
sequence-by-sequence attention matrix.

The main operations were:

`Q @ K^T`

followed by scaling, softmax, and:

`Attention Probabilities @ V`

The score and probability matrices contain `L x L` values for each batch and
head. This is why memory use grows rapidly when sequence length increases.

Run ID:

`D-4090-NAIVE`

GPU UUID:

`GPU-b60240a9-31a3-126f-751f-afc1dcc1d73f`

## Naive Attention Measurements

| Sequence Length | Peak Allocated Memory | Median Forward Latency | Status |
|---:|---:|---:|---|
| 512 | 0.047 GiB | 0.111 ms | SUCCESS |
| 1024 | 0.149 GiB | 0.438 ms | SUCCESS |
| 2048 | 0.539 GiB | 1.879 ms | SUCCESS |
| 4096 | 2.070 GiB | 7.356 ms | SUCCESS |
| 8192 | 8.133 GiB | 28.978 ms | SUCCESS |
| 16384 | **32.258 GiB** | **892.224 ms** | SUCCESS |

All six sequence lengths required by the assignment completed.

---

# Naive Attention Physical-Memory Verification

At `L=16384`, PyTorch reported:

**32.2579 GiB peak allocated memory**

This is greater than the physical VRAM capacity of the RTX 4090.

To understand this result, I ran a simultaneous `nvidia-smi` memory monitor.

Maximum physical GPU-memory usage observed:

**23945 MiB**

Reported total GPU memory:

**24564 MiB**

This showed that the physical card was already almost full.

The Windows workstation was using the WDDM driver model. In this environment,
memory can be oversubscribed and paged instead of immediately producing a
normal CUDA out-of-memory exception.

Therefore, the 32.258 GiB PyTorch allocation should not be interpreted as
32 GiB physically resident in the RTX 4090 VRAM.

Evidence:

`raw/attention/nvidia_memory_watch_16384_4090.txt`

---

# Naive Attention OOM Search

Additional sequence lengths were tested after the required coarse sweep.

Successful probes included:

- `L=17408`
- `L=17920`
- `L=18432`
- `L=18944`

Largest safely tested successful sequence length:

**18,944**

Smallest tested failing sequence length:

**Not observed within the safely tested range**

Because physical GPU memory was already nearly full and WDDM continued to
allow memory oversubscription/paging, I stopped increasing the sequence length
instead of deliberately creating excessive host-memory paging.

For this reason, I do not claim an exact naive-attention OOM boundary.

The result used in Part F is:

**No clean OOM observed; largest safely tested success = 18,944**

---

# Naive Attention Memory-Curve Fit

Peak memory was fitted as a function of sequence length using:

`M(L) = aL^2 + bL + c`

Measured quadratic coefficient:

**a = 1.19209290e-07 GiB/token²**

Measured fit quality:

**R² = 1.0000**

This result confirms from my own measurements that the explicit naive
attention implementation had a strong quadratic memory term.

The behavior comes from materializing the sequence-by-sequence attention
matrices whose size grows with `L²`.

Evidence:

- `raw/attention/naive_attention_memory_fit_4090.txt`
- `raw/attention/naive_attention_memory_fit_run_4090.txt`

Figure:

`figures/naive_attention_memory_fit_4090.png`

Run ID:

`D-4090-NAIVE-MEMORY-FIT`

---

# Initial Flash Attention Failure

My first optimized implementation explicitly requested PyTorch's:

`FLASH_ATTENTION`

All required sequence lengths failed with:

`RuntimeError: No available kernel. Aborting execution.`

I did not hide this failure or silently change the result.

A separate backend probe produced:

| Backend | Result |
|---|---|
| FLASH_ATTENTION | FAILED |
| EFFICIENT_ATTENTION | SUCCESS |
| CUDNN_ATTENTION | SUCCESS |
| MATH | SUCCESS |

The installed Windows/PyTorch environment therefore did not provide an
available Flash Attention kernel for this test configuration.

Because the assignment allows a fused **or memory-efficient** implementation,
I used PyTorch's explicitly selected:

`EFFICIENT_ATTENTION`

backend for the optimized comparison.

Evidence:

- `raw/attention/attention_fused_coarse_4090.csv`
- `raw/attention/attention_fused_coarse_run_4090.txt`
- `raw/attention/sdpa_backend_probe_4090.txt`

Run ID:

`D-4090-FLASH-PROBE`

---

# Memory-Efficient Attention Measurements

Run ID:

`D-4090-EFFICIENT`

GPU UUID:

`GPU-b60240a9-31a3-126f-751f-afc1dcc1d73f`

| Sequence Length | Peak Allocated Memory | Median Forward Latency | Status |
|---:|---:|---:|---|
| 512 | 0.0078 GiB | 0.127 ms | SUCCESS |
| 1024 | 0.0156 GiB | 0.139 ms | SUCCESS |
| 2048 | 0.0313 GiB | 0.352 ms | SUCCESS |
| 4096 | 0.0625 GiB | 1.181 ms | SUCCESS |
| 8192 | 0.1250 GiB | 4.741 ms | SUCCESS |
| 16384 | **0.2500 GiB** | **17.264 ms** | SUCCESS |

At `L=16384`:

Naive peak allocated memory:

**32.258 GiB**

Memory-efficient peak allocated memory:

**0.250 GiB**

The optimized implementation avoided explicitly storing the complete
`L x L` score/probability matrices in GPU memory. Instead, the attention
calculation was performed in a memory-efficient way using smaller intermediate
working data.

This greatly reduced the amount of temporary memory required.

Evidence:

- `raw/attention/attention_fused_efficient_coarse_4090.csv`
- `raw/attention/attention_fused_efficient_coarse_run_4090.txt`

---

# Attention Speedup

Speedup was calculated using:

`Speedup = Naive Median Latency / Efficient Median Latency`

| Sequence Length | Naive Latency | Efficient Latency | Speedup |
|---:|---:|---:|---:|
| 512 | 0.111 ms | 0.127 ms | 0.87x |
| 1024 | 0.438 ms | 0.139 ms | 3.15x |
| 2048 | 1.879 ms | 0.352 ms | 5.33x |
| 4096 | 7.356 ms | 1.181 ms | 6.23x |
| 8192 | 28.978 ms | 4.741 ms | 6.11x |
| 16384 | 892.224 ms | 17.264 ms | **51.68x** |

At `L=512`, the memory-efficient version was slightly slower. The sequence
was small enough that optimized-backend overhead was large compared with the
actual computation.

For larger sequence lengths, the benefit became much clearer.

At `L=16384`, memory-efficient attention was approximately:

**51.68x faster**

than the measured naive implementation.

Evidence:

- `raw/attention/attention_comparison_4090.csv`
- `raw/attention/attention_comparison_run_4090.txt`

Figure:

`figures/attention_speedup_4090.png`

Run ID:

`D-4090-ATTENTION-COMPARISON`

---

# Memory-Efficient Attention Boundary Search

The memory-efficient implementation allowed much larger sequence lengths.

| Sequence Length | Peak Allocated Memory | Forward Latency | Status |
|---:|---:|---:|---|
| 32768 | 0.5 GiB | 72.94 ms | SUCCESS |
| 65536 | 1.0 GiB | 277.77 ms | SUCCESS |
| 131072 | 2.0 GiB | 1105.55 ms | SUCCESS |
| 262144 | 4.0 GiB | 4467.27 ms | SUCCESS |
| 524288 | 8.0 GiB | 18555.90 ms | SUCCESS |
| 1048576 | 16.0 GiB | 78659.87 ms | SUCCESS |
| 1572864 | **24.0 GiB** | **182901.69 ms** | SUCCESS |

Largest safely tested successful sequence length:

**1,572,864**

Smallest tested failing sequence length:

**Not observed within the safely tested range**

At the largest test, PyTorch already reported approximately 24.0 GiB peak
allocated memory. Since this was approximately equal to the card's physical
VRAM and WDDM oversubscription had already been observed, I stopped the search
instead of intentionally forcing extreme paging.

I therefore do not claim an exact OOM boundary.

For Part F, the result is reported as:

**No clean OOM observed; largest safely tested success = 1,572,864**

Evidence:

- `raw/attention/efficient_oom_coarse_boundary_4090.csv`
- `raw/attention/efficient_oom_coarse_boundary_run_4090.txt`
- `raw/attention/efficient_oom_large_boundary_4090.csv`
- `raw/attention/efficient_oom_large_boundary_run_4090.txt`

Run ID:

`D-4090-EFFICIENT-OOM`

---

# Part E — Sustained Load and Thermal Behavior

A sustained BF16 matrix multiplication workload was run for approximately
20 minutes.

Configuration:

| Parameter | Value |
|---|---:|
| Matrix size | 8192 x 8192 |
| Precision | BF16 |
| Requested duration | 1200 seconds |
| Telemetry interval | 5 seconds |
| Telemetry samples | 238 |

The telemetry log recorded:

- SM clock
- memory clock
- GPU temperature
- power draw
- configured power limit
- GPU utilization
- GPU memory usage

Run ID:

`E-4090-SUSTAINED`

GPU UUID:

`GPU-b60240a9-31a3-126f-751f-afc1dcc1d73f`

---

## First 30 Seconds

| Metric | Result |
|---|---:|
| Peak throughput | **168.96 TFLOPS** |
| Average throughput | 162.76 TFLOPS |
| Average SM clock | 2537.50 MHz |
| Average temperature | 55.67 C |
| Average power | 377.17 W |

---

## Final Five Minutes

| Metric | Result |
|---|---:|
| Average throughput | **160.11 TFLOPS** |
| Peak throughput | 161.96 TFLOPS |
| Average SM clock | 2488.00 MHz |
| Average temperature | 75.97 C |
| Average power | 449.55 W |
| Average GPU utilization | 99.98% |

---

## Full-Run Summary

| Metric | Result |
|---|---:|
| Maximum temperature | 77.00 C |
| Average power | 447.70 W |
| Maximum power | 451.02 W |
| Configured power limit | 450.00 W |
| Throughput change | -5.24% |
| SM clock change | -1.95% |

Steady-state throughput relative to the initial peak:

`160.11 / 168.96 = 0.9476`

Therefore:

**Steady-state / peak throughput = 94.76%**

---

# Clock and Temperature Behavior

During the beginning of the run, the GPU was still warming up.

As the run continued, temperature increased and later stabilized at
approximately 76 C.

In the final five minutes:

- Average temperature was 75.97 C.
- Average power was 449.55 W.
- The configured power limit was 450 W.
- Average utilization was 99.98%.
- Average SM clock was 2488 MHz.

The GPU therefore spent most of the steady-state period very close to its
configured power ceiling.

The combined clock and temperature figure is:

`figures/thermal_clock_temperature_4090.png`

Additional individual figures are:

- `figures/thermal_clock_4090.png`
- `figures/thermal_temperature_4090.png`

---

# Throttling Analysis

**Throttle onset: None clearly observed**

The GPU did not show evidence of severe thermal throttling.

The maximum measured temperature was:

**77 C**

while the final five-minute average was:

**75.97 C**

At the same time, steady-state average power was:

**449.55 W**

against the configured:

**450 W power limit**

The average SM clock decreased by only:

**1.95%**

between the initial period and the final five minutes.

The measured behavior is therefore more consistent with the GPU reaching its
power ceiling and adjusting its normal boost clock than with severe
temperature-driven throttling.

Evidence:

- `raw/thermal/sustained_compute_run_4090.txt`
- `raw/thermal/thermal_4090.csv`
- `raw/thermal/throughput_4090.csv`
- `raw/thermal/thermal_analysis_4090.txt`
- `raw/thermal/thermal_analysis_run_4090.txt`

---

# Part F — Table HW2.5.1

## Final Summary Table

| Measurement | Your GPU — RTX 4090 | Notes |
|---|---:|---|
| **Peak achieved TFLOPS (BF16)** | **164.69 TFLOPS** | Highest required BF16 matmul measurement at `N=8192`. Run: `B-4090-MATMUL` |
| **% of theoretical peak (BF16)** | **99.69%** | 164.6871 TFLOPS relative to the 165.2 TFLOPS dense BF16 reference. Run: `B-4090-MATMUL` |
| **Effective bandwidth (GB/s)** | **920.74 GB/s** | 91.34% of the 1008 GB/s vendor bandwidth. Run: `C-4090-ROOFLINE` |
| **Naive attention OOM length** | **No clean OOM; largest safe success = 18,944** | WDDM oversubscription/paging prevented a clean CUDA OOM in the safely tested range. Run: `D-4090-NAIVE` |
| **Fused / memory-efficient attention OOM length** | **No clean OOM; largest safe success = 1,572,864** | Efficient Attention reached 24.0 GiB peak allocated memory at the largest safe test. Run: `D-4090-EFFICIENT-OOM` |
| **Steady-state / peak throughput** | **160.11 / 168.96 TFLOPS = 94.76%** | Final five-minute average compared with the first-30-second peak. Run: `E-4090-SUSTAINED` |
| **Throttle onset (s, or none)** | **None observed** | Maximum temperature 77 C; steady power approximately 449.55 W against a 450 W limit. Run: `E-4090-SUSTAINED` |

**Table HW2.5.1 — Summary of the RTX 4090 measurements. Every value is
traceable to a UUID-labelled run in `RUN_LOG.txt`.**

---

# Part F Traceability

| Table Measurement | RUN_LOG Run ID | Primary Evidence |
|---|---|---|
| Peak achieved TFLOPS (BF16) | `B-4090-MATMUL` | `raw/matmul/matmul_4090.csv` |
| % theoretical BF16 | `B-4090-MATMUL` | `raw/matmul/matmul_4090_with_theoretical.csv` |
| Effective bandwidth | `C-4090-ROOFLINE` | `raw/bandwidth/roofline_4090.csv` |
| Naive attention OOM result | `D-4090-NAIVE` | `raw/attention/naive_oom_step4_4090.csv` |
| Efficient attention OOM result | `D-4090-EFFICIENT-OOM` | `raw/attention/efficient_oom_large_boundary_4090.csv` |
| Steady-state / peak throughput | `E-4090-SUSTAINED` | `raw/thermal/thermal_analysis_4090.txt` |
| Throttle onset | `E-4090-SUSTAINED` | `raw/thermal/thermal_4090.csv` |

GPU UUID for all listed benchmark runs:

`GPU-b60240a9-31a3-126f-751f-afc1dcc1d73f`

---

# Key Findings

The experiments produced several clear results.

First, lower-precision matrix multiplication made much better use of the RTX
4090 than the small FP32 workload. The highest required BF16 measurement was
164.69 TFLOPS, which was approximately 99.69% of the dense theoretical
reference used in this analysis.

Second, the memory-bound workload achieved 920.74 GB/s, which was approximately
91.34% of the card's specified memory bandwidth. Its arithmetic intensity was
very low, so adding more compute capability would not remove its main
bottleneck.

Third, naive attention became expensive very quickly as sequence length grew.
The fitted memory curve showed a quadratic term with coefficient
`1.19209290e-07 GiB/token²` and an `R²` of 1.0.

At `L=16384`, the naive implementation reported approximately 32.26 GiB peak
allocated memory, compared with only 0.25 GiB for the memory-efficient
implementation.

Fourth, the memory-efficient implementation became much faster at large
sequence lengths. At `L=16384`, the measured speedup was approximately 51.68x.

Finally, the RTX 4090 sustained 94.76% of its initial peak throughput during the
last five minutes of the 20-minute compute test. Temperature remained around
76 C while power stayed almost exactly at the 450 W limit. I therefore did not
observe a clear thermal-throttle onset.

---

# Figures

The following figures are included with the assignment:

- `figures/matmul_throughput_4090.png`
- `figures/naive_attention_memory_fit_4090.png`
- `figures/attention_speedup_4090.png`
- `figures/thermal_clock_temperature_4090.png`
- `figures/thermal_clock_4090.png`
- `figures/thermal_temperature_4090.png`

---

# Supporting Files

Detailed experiment records:

`RUN_LOG.txt`

Reservation and GPU-hour record:

`GPU_HOURS.md`

AI-use disclosure:

`AI_USE.md`

Raw measurements:

- `raw/gpu_info/`
- `raw/matmul/`
- `raw/bandwidth/`
- `raw/attention/`
- `raw/thermal/`
- `raw/reservations/`

All Part B–E results reported in this file correspond to the RTX 4090 identified
by GPU UUID:

`GPU-b60240a9-31a3-126f-751f-afc1dcc1d73f`
