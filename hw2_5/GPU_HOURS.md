# HW2.5 GPU Reservation and GPU-Hour Record

## GPU Used

**GPU:** NVIDIA GeForce RTX 4090

**GPU UUID:**  
`GPU-b60240a9-31a3-126f-751f-afc1dcc1d73f`

**Experiment Date:**  
September 10, 2026

---

## GPU Identity Evidence

The assigned workstation was verified using `nvidia-smi`, `nvidia-smi -q`,
PyTorch CUDA checks, and the CUDA compiler version.

The following hardware/software information was captured directly from
the GPU workstation.

| Property | Observed Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 |
| GPU UUID | `GPU-b60240a9-31a3-126f-751f-afc1dcc1d73f` |
| NVIDIA/KMD Version | 610.60 |
| CUDA UMD Version | 13.3 |
| Reported GPU Memory | 24564 MiB |
| Configured Power Limit | 450 W |
| Driver Model | WDDM |
| Architecture | Ada Lovelace |
| PyTorch Version | 2.11.0+cu128 |
| PyTorch CUDA Runtime | 12.8 |
| CUDA Available in PyTorch | True |
| BF16 Supported | True |
| NVCC CUDA Toolkit | 12.8 |
| NVCC Release | V12.8.93 |

Evidence files:

- `raw/gpu_info/nvidia_smi_summary_4090.txt`
- `raw/gpu_info/nvidia_smi_4090.txt`
- `raw/gpu_info/gpu_identity_4090.txt`
- `raw/gpu_info/pytorch_cuda_4090.txt`
- `raw/gpu_info/nvcc_4090.txt`

---

## Reservation Information

The RTX 4090 workstation was reserved through the SJSU GPU/HPC lab
reservation system.

**Assigned Lab:** ISB 836

**Assigned Machine:** Machine 2

**Reservation Start:** September 10, 2026, 12:00 PM PDT

**Reservation End:** September 11, 2026, 11:59 AM PDT

**Reserved Window:** Approximately 24 hours

The reservation window represents the time during which the workstation
was allocated. It should not be interpreted as 24 hours of continuous GPU
computation.

No username, password, or workstation credential is included in this
repository.

---

## Actual GPU Usage Evidence

The earliest GPU activity directly visible in the retained experiment
evidence is the `nvidia-smi` capture at:

**September 10, 2026, 15:04:30 PDT**

A second detailed `nvidia-smi -q` capture was recorded at approximately:

**September 10, 2026, 15:05 PDT**

The later required sustained-compute experiment finished at:

**September 10, 2026, 17:33:03 PDT**

Using the earliest clearly evidenced GPU activity and the final recorded
GPU workload completion time:

`17:33:03 - 15:04:30 = 2 hours 28 minutes 33 seconds`

Therefore, the minimum directly evidenced GPU usage duration is:

**2.48 GPU-hours**

This value represents the minimum duration that can be directly supported
by the retained experiment logs and screenshots.

If additional GPU work occurred before 15:04:30, the true physical lab
usage duration would be greater. I therefore report 2.48 GPU-hours as an
evidence-based minimum rather than claiming the complete reservation
window as active GPU usage.

---

## GPU Work Performed During the Session

The RTX 4090 was used for the following HW2.5 experiments:

| Part | Work Performed |
|---|---|
| Part A | GPU identification and complete `nvidia-smi -q` capture |
| Part A | PyTorch/CUDA environment verification |
| Part A | CUDA compiler version verification |
| Part B | FP32 matrix multiplication benchmark |
| Part B | TF32 matrix multiplication benchmark |
| Part B | FP16 matrix multiplication benchmark |
| Part B | BF16 matrix multiplication benchmark |
| Part B Extension | FP8 capability and throughput experiment |
| Part C | Memory-bandwidth benchmark |
| Part C | Arithmetic-intensity and roofline analysis |
| Part D | Naive scaled dot-product attention |
| Part D | Attention memory-growth experiment |
| Part D | Naive large-sequence boundary testing |
| Part D | SDPA backend capability testing |
| Part D | Memory-efficient attention benchmark |
| Part D | Naive vs. efficient attention comparison |
| Part D | Efficient-attention large-sequence testing |
| Part E | 20-minute sustained BF16 compute workload |
| Part E | GPU clock, temperature, power, utilization, and memory logging |

---

## Required 20-Minute Sustained Compute

The required sustained-compute workload was executed using:

- Precision: BF16
- Matrix size: `8192 x 8192`
- Requested duration: 1200 seconds
- Telemetry interval: approximately 5 seconds

The run completed successfully.

The final recorded completion time was:

**2026-09-10 17:33:03**

Evidence:

- `raw/thermal/sustained_compute_run_4090.txt`
- `raw/thermal/thermal_4090.csv`
- `raw/thermal/throughput_4090.csv`
- `raw/thermal/thermal_analysis_4090.txt`

---

## Reserved Time vs. Actual GPU Time

| Measurement | Value |
|---|---:|
| Reserved workstation window | ~24 hours |
| Earliest directly evidenced GPU activity | 15:04:30 PDT |
| Final recorded workload completion | 17:33:03 PDT |
| Minimum evidenced GPU usage | **2 h 28 m 33 s** |
| Minimum evidenced GPU-hours | **2.48 GPU-hours** |

The reservation duration and actual measured GPU usage are intentionally
reported separately.

The full reservation window was not treated as active GPU computation.

---

## Traceability

GPU UUID used throughout the experiment:

`GPU-b60240a9-31a3-126f-751f-afc1dcc1d73f`

The UUID appears in the corresponding experiment records in:

`RUN_LOG.txt`

Summary metrics are reported in:

`METRICS.md`

Raw measurements are stored under:

- `raw/gpu_info/`
- `raw/matmul/`
- `raw/bandwidth/`
- `raw/attention/`
- `raw/thermal/`

---

## Final GPU-Hour Statement

For HW2.5, the RTX 4090 reservation window was approximately 24 hours.

Based on the retained timestamped GPU evidence, at least **2.48 GPU-hours**
of workstation/GPU usage are directly documented on September 10, 2026.

This value is reported as the evidence-supported minimum actual usage and
is kept separate from the much larger reservation window.
