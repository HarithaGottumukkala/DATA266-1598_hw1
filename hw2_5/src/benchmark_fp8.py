import csv
import statistics
import subprocess
from datetime import datetime

import torch


SIZES = [1024, 4096, 8192, 16384]
WARMUP = 5
REPETITIONS = 20
GPU_LABEL = "4090"


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


def scaled_mm(a, b, scale_a, scale_b):
    try:
        return torch._scaled_mm(
            a,
            b,
            scale_a=scale_a,
            scale_b=scale_b,
            out_dtype=torch.bfloat16
        )
    except TypeError:
        return torch._scaled_mm(
            a,
            b,
            scale_a=scale_a,
            scale_b=scale_b,
            output_dtype=torch.bfloat16
        )


def benchmark_size(n):
    torch.cuda.empty_cache()

    # Create higher-precision source tensors.
    a_hp = torch.randn(
        n, n,
        device="cuda",
        dtype=torch.bfloat16
    )

    b_hp = torch.randn(
        n, n,
        device="cuda",
        dtype=torch.bfloat16
    )

    fp8_dtype = torch.float8_e4m3fn
    fp8_max = torch.finfo(fp8_dtype).max

    # Per-tensor scaling used only during preparation.
    quant_a = fp8_max / a_hp.abs().max()
    quant_b = fp8_max / b_hp.abs().max()

    a_fp8 = (a_hp * quant_a).to(fp8_dtype)

    # Layout used successfully by our FP8 probe.
    b_fp8 = (
        (b_hp.T * quant_b)
        .to(fp8_dtype)
        .contiguous()
        .T
    )

    scale_a = quant_a.reciprocal().float()
    scale_b = quant_b.reciprocal().float()

    # Quantization is intentionally NOT included in GEMM timing.
    del a_hp, b_hp
    torch.cuda.empty_cache()

    # Warmup.
    for _ in range(WARMUP):
        output = scaled_mm(
            a_fp8,
            b_fp8,
            scale_a,
            scale_b
        )

    torch.cuda.synchronize()

    times_ms = []

    for _ in range(REPETITIONS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        output = scaled_mm(
            a_fp8,
            b_fp8,
            scale_a,
            scale_b
        )

        end.record()
        torch.cuda.synchronize()

        times_ms.append(start.elapsed_time(end))

    median_ms = statistics.median(times_ms)
    mean_ms = statistics.mean(times_ms)

    stdev_ms = (
        statistics.stdev(times_ms)
        if len(times_ms) > 1
        else 0.0
    )

    seconds = median_ms / 1000.0

    flops = 2 * (n ** 3)

    achieved_tflops = (
        flops
        / seconds
        / 1e12
    )

    del a_fp8, b_fp8, output
    torch.cuda.empty_cache()

    return median_ms, mean_ms, stdev_ms, achieved_tflops


if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is unavailable.")

if not hasattr(torch, "_scaled_mm"):
    raise RuntimeError(
        "torch._scaled_mm is not exposed by this PyTorch build."
    )

gpu_name = torch.cuda.get_device_name(0)
gpu_uuid = get_gpu_uuid()

print("=" * 78)
print("HW2.5 PART B4 — FP8 E4M3 THROUGHPUT BENCHMARK")
print("=" * 78)

print("Timestamp:", datetime.now().isoformat())
print("GPU:", gpu_name)
print("GPU UUID:", gpu_uuid)
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("Input precision: FP8 E4M3")
print("Output dtype: BF16")
print("Warmup iterations:", WARMUP)
print("Timed repetitions:", REPETITIONS)
print("Quantization included in timing: NO")
print()

rows = []

for n in SIZES:
    print(f"Running FP8 E4M3, N={n} ...", end=" ", flush=True)

    try:
        median_ms, mean_ms, stdev_ms, tflops = benchmark_size(n)

        print(
            f"{median_ms:.4f} ms | "
            f"{tflops:.3f} TFLOPS"
        )

        rows.append({
            "timestamp": datetime.now().isoformat(),
            "gpu_label": GPU_LABEL,
            "gpu_name": gpu_name,
            "gpu_uuid": gpu_uuid,
            "N": n,
            "input_precision": "FP8_E4M3",
            "output_dtype": "BF16",
            "warmup": WARMUP,
            "repetitions": REPETITIONS,
            "median_ms": median_ms,
            "mean_ms": mean_ms,
            "stdev_ms": stdev_ms,
            "achieved_tflops": tflops,
            "quantization_in_timing": "NO",
            "status": "SUCCESS",
            "error": ""
        })

    except Exception as e:
        print("FAILED")
        print(type(e).__name__, str(e))

        rows.append({
            "timestamp": datetime.now().isoformat(),
            "gpu_label": GPU_LABEL,
            "gpu_name": gpu_name,
            "gpu_uuid": gpu_uuid,
            "N": n,
            "input_precision": "FP8_E4M3",
            "output_dtype": "BF16",
            "warmup": WARMUP,
            "repetitions": REPETITIONS,
            "median_ms": "",
            "mean_ms": "",
            "stdev_ms": "",
            "achieved_tflops": "",
            "quantization_in_timing": "NO",
            "status": "FAILED",
            "error": f"{type(e).__name__}: {e}"
        })

output_file = "raw/matmul/fp8_benchmark_4090.csv"

with open(output_file, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=rows[0].keys()
    )

    writer.writeheader()
    writer.writerows(rows)

print()
print("=" * 78)
print("FP8 BENCHMARK COMPLETE")
print("=" * 78)
print("Saved:", output_file)