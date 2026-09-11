import csv
import statistics
import subprocess
from datetime import datetime

import torch


GPU_LABEL = "4090"

# Official RTX 4090 reference values
SPECIFIED_BANDWIDTH_GB_S = 1008.0
THEORETICAL_FP32_TFLOPS = 82.6

# Memory-bound experiment
ELEMENT_COUNT = 200_000_000

# Compute-bound experiment
MATMUL_N = 8192

WARMUP = 10
REPETITIONS = 30


def get_gpu_uuid():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid",
            "--format=csv,noheader"
        ],
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout.strip()


def time_operation(operation, warmup, repetitions):
    for _ in range(warmup):
        operation()

    torch.cuda.synchronize()

    times_ms = []

    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        operation()
        end.record()

        torch.cuda.synchronize()

        times_ms.append(start.elapsed_time(end))

    return times_ms


if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is not available.")

gpu_name = torch.cuda.get_device_name(0)
gpu_uuid = get_gpu_uuid()

print("=" * 80)
print("HW2.5 PART C — BANDWIDTH-BOUND VS COMPUTE-BOUND")
print("=" * 80)

print("Timestamp:", datetime.now().isoformat())
print("GPU:", gpu_name)
print("GPU UUID:", gpu_uuid)
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print()

# ---------------------------------------------------------------------
# MEMORY-BOUND: FP32 ELEMENTWISE ADD
# ---------------------------------------------------------------------

print("-" * 80)
print("MEMORY-BOUND OPERATION: FP32 ELEMENTWISE ADD")
print("-" * 80)

a = torch.randn(
    ELEMENT_COUNT,
    device="cuda",
    dtype=torch.float32
)

b = torch.randn(
    ELEMENT_COUNT,
    device="cuda",
    dtype=torch.float32
)

c = torch.empty_like(a)

times = time_operation(
    lambda: torch.add(a, b, out=c),
    WARMUP,
    REPETITIONS
)

median_ms = statistics.median(times)
mean_ms = statistics.mean(times)
stdev_ms = statistics.stdev(times)

# Each element:
# read A = 4 bytes
# read B = 4 bytes
# write C = 4 bytes
#
# Total = 12 bytes per element
bytes_moved = ELEMENT_COUNT * 12

seconds = median_ms / 1000.0

effective_bandwidth_gb_s = (
    bytes_moved / seconds / 1e9
)

bandwidth_percent = (
    effective_bandwidth_gb_s
    / SPECIFIED_BANDWIDTH_GB_S
    * 100
)

# One floating-point addition per element
elementwise_flops = ELEMENT_COUNT

elementwise_ai = (
    elementwise_flops / bytes_moved
)

print(f"Elements                    : {ELEMENT_COUNT:,}")
print(f"Median latency              : {median_ms:.4f} ms")
print(f"Mean latency                : {mean_ms:.4f} ms")
print(f"Std deviation               : {stdev_ms:.4f} ms")
print(f"Effective bandwidth         : {effective_bandwidth_gb_s:.2f} GB/s")
print(f"Specified bandwidth         : {SPECIFIED_BANDWIDTH_GB_S:.2f} GB/s")
print(f"Percentage of specification : {bandwidth_percent:.2f}%")
print(f"Arithmetic intensity        : {elementwise_ai:.6f} FLOPs/byte")

del a, b, c
torch.cuda.empty_cache()

# ---------------------------------------------------------------------
# COMPUTE-BOUND: FP32 MATRIX MULTIPLICATION
# ---------------------------------------------------------------------

print()
print("-" * 80)
print("COMPUTE-BOUND OPERATION: FP32 MATRIX MULTIPLICATION")
print("-" * 80)

# Make sure TF32 is disabled so this is explicitly FP32.
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

A = torch.randn(
    (MATMUL_N, MATMUL_N),
    device="cuda",
    dtype=torch.float32
)

B = torch.randn(
    (MATMUL_N, MATMUL_N),
    device="cuda",
    dtype=torch.float32
)

C = torch.empty(
    (MATMUL_N, MATMUL_N),
    device="cuda",
    dtype=torch.float32
)

matmul_times = time_operation(
    lambda: torch.mm(A, B, out=C),
    WARMUP,
    REPETITIONS
)

matmul_median_ms = statistics.median(matmul_times)
matmul_seconds = matmul_median_ms / 1000.0

matmul_flops = 2 * (MATMUL_N ** 3)

achieved_tflops = (
    matmul_flops
    / matmul_seconds
    / 1e12
)

# Algorithmic minimum traffic:
# read A + read B + write C
matmul_bytes = (
    3
    * (MATMUL_N ** 2)
    * 4
)

matmul_ai = (
    matmul_flops
    / matmul_bytes
)

# Roofline ridge point:
# peak FLOPs / memory bandwidth
ridge_point = (
    THEORETICAL_FP32_TFLOPS * 1e12
    / (SPECIFIED_BANDWIDTH_GB_S * 1e9)
)

memory_side = (
    "MEMORY-BOUND"
    if elementwise_ai < ridge_point
    else "COMPUTE-BOUND"
)

compute_side = (
    "MEMORY-BOUND"
    if matmul_ai < ridge_point
    else "COMPUTE-BOUND"
)

print(f"Matrix size N               : {MATMUL_N}")
print(f"Median latency              : {matmul_median_ms:.4f} ms")
print(f"Achieved throughput         : {achieved_tflops:.2f} TFLOPS")
print(f"Arithmetic intensity        : {matmul_ai:.2f} FLOPs/byte")

print()
print("-" * 80)
print("ROOFLINE CLASSIFICATION")
print("-" * 80)

print(f"Roofline ridge point        : {ridge_point:.2f} FLOPs/byte")
print(
    f"Elementwise add             : "
    f"{elementwise_ai:.6f} FLOPs/byte -> {memory_side}"
)
print(
    f"Matrix multiplication       : "
    f"{matmul_ai:.2f} FLOPs/byte -> {compute_side}"
)

# ---------------------------------------------------------------------
# SAVE CSV
# ---------------------------------------------------------------------

output_file = "raw/bandwidth/roofline_4090.csv"

with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "timestamp",
        "gpu_label",
        "gpu_name",
        "gpu_uuid",
        "operation",
        "precision",
        "problem_size",
        "median_ms",
        "effective_bandwidth_gb_s",
        "bandwidth_percent",
        "achieved_tflops",
        "arithmetic_intensity_flops_per_byte",
        "ridge_point_flops_per_byte",
        "roofline_classification"
    ])

    writer.writerow([
        datetime.now().isoformat(),
        GPU_LABEL,
        gpu_name,
        gpu_uuid,
        "elementwise_add",
        "FP32",
        ELEMENT_COUNT,
        median_ms,
        effective_bandwidth_gb_s,
        bandwidth_percent,
        "",
        elementwise_ai,
        ridge_point,
        memory_side
    ])

    writer.writerow([
        datetime.now().isoformat(),
        GPU_LABEL,
        gpu_name,
        gpu_uuid,
        "square_matmul",
        "FP32",
        MATMUL_N,
        matmul_median_ms,
        "",
        "",
        achieved_tflops,
        matmul_ai,
        ridge_point,
        compute_side
    ])

print()
print("=" * 80)
print("PART C COMPLETE")
print("=" * 80)
print("Saved:", output_file)