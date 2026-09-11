import argparse
import csv
import statistics
import subprocess
from datetime import datetime

import torch


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


def set_fp32_mode(mode):
    """
    FP32 = TF32 disabled.
    TF32 = TF32 enabled for float32 matrix multiplication.
    """
    if mode == "FP32":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

    elif mode == "TF32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def benchmark_matmul(n, precision, warmup, repetitions):
    if precision == "FP32":
        dtype = torch.float32
        set_fp32_mode("FP32")

    elif precision == "TF32":
        dtype = torch.float32
        set_fp32_mode("TF32")

    elif precision == "FP16":
        dtype = torch.float16

    elif precision == "BF16":
        dtype = torch.bfloat16

    else:
        raise ValueError(f"Unsupported precision: {precision}")

    torch.cuda.empty_cache()

    a = torch.randn((n, n), device="cuda", dtype=dtype)
    b = torch.randn((n, n), device="cuda", dtype=dtype)
    c = torch.empty((n, n), device="cuda", dtype=dtype)

    # Warm-up
    for _ in range(warmup):
        torch.mm(a, b, out=c)

    torch.cuda.synchronize()

    times_ms = []

    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        torch.mm(a, b, out=c)

        end.record()

        torch.cuda.synchronize()

        elapsed_ms = start.elapsed_time(end)
        times_ms.append(elapsed_ms)

    median_ms = statistics.median(times_ms)
    mean_ms = statistics.mean(times_ms)
    stdev_ms = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0

    seconds = median_ms / 1000.0

    # Dense square matrix multiplication:
    # approximately 2 * N^3 floating-point operations.
    flops = 2 * (n ** 3)

    achieved_tflops = flops / seconds / 1e12

    del a, b, c
    torch.cuda.empty_cache()

    return {
        "N": n,
        "precision": precision,
        "warmup": warmup,
        "repetitions": repetitions,
        "median_ms": median_ms,
        "mean_ms": mean_ms,
        "stdev_ms": stdev_ms,
        "achieved_tflops": achieved_tflops,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gpu-label",
        required=True,
        help="Example: 4090 or 5090"
    )

    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[1024, 4096, 8192, 16384]
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=5
    )

    parser.add_argument(
        "--reps",
        type=int,
        default=20
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available.")

    gpu_name = torch.cuda.get_device_name(0)
    gpu_uuid = get_gpu_uuid()

    print("=" * 80)
    print("HW2.5 PART B — MATRIX MULTIPLICATION BENCHMARK")
    print("=" * 80)
    print(f"Timestamp           : {datetime.now().isoformat()}")
    print(f"GPU                 : {gpu_name}")
    print(f"GPU UUID            : {gpu_uuid}")
    print(f"PyTorch             : {torch.__version__}")
    print(f"CUDA runtime        : {torch.version.cuda}")
    print(f"Warm-up iterations  : {args.warmup}")
    print(f"Timed repetitions   : {args.reps}")
    print(f"Matrix sizes        : {args.sizes}")
    print("=" * 80)

    output_file = f"raw/matmul/matmul_{args.gpu_label}.csv"

    fieldnames = [
        "timestamp",
        "gpu_label",
        "gpu_name",
        "gpu_uuid",
        "pytorch_version",
        "cuda_runtime",
        "N",
        "precision",
        "warmup",
        "repetitions",
        "median_ms",
        "mean_ms",
        "stdev_ms",
        "achieved_tflops",
        "status",
        "error"
    ]

    rows = []

    precisions = [
        "FP32",
        "TF32",
        "FP16",
        "BF16"
    ]

    for precision in precisions:
        print()
        print("-" * 80)
        print(f"PRECISION: {precision}")
        print("-" * 80)

        for n in args.sizes:
            print(f"Running {precision}, N={n} ...", end=" ", flush=True)

            try:
                result = benchmark_matmul(
                    n=n,
                    precision=precision,
                    warmup=args.warmup,
                    repetitions=args.reps
                )

                print(
                    f"{result['median_ms']:.3f} ms | "
                    f"{result['achieved_tflops']:.3f} TFLOPS"
                )

                rows.append({
                    "timestamp": datetime.now().isoformat(),
                    "gpu_label": args.gpu_label,
                    "gpu_name": gpu_name,
                    "gpu_uuid": gpu_uuid,
                    "pytorch_version": torch.__version__,
                    "cuda_runtime": torch.version.cuda,
                    "N": result["N"],
                    "precision": result["precision"],
                    "warmup": result["warmup"],
                    "repetitions": result["repetitions"],
                    "median_ms": result["median_ms"],
                    "mean_ms": result["mean_ms"],
                    "stdev_ms": result["stdev_ms"],
                    "achieved_tflops": result["achieved_tflops"],
                    "status": "SUCCESS",
                    "error": ""
                })

            except Exception as e:
                print("FAILED")
                print(f"  {type(e).__name__}: {e}")

                rows.append({
                    "timestamp": datetime.now().isoformat(),
                    "gpu_label": args.gpu_label,
                    "gpu_name": gpu_name,
                    "gpu_uuid": gpu_uuid,
                    "pytorch_version": torch.__version__,
                    "cuda_runtime": torch.version.cuda,
                    "N": n,
                    "precision": precision,
                    "warmup": args.warmup,
                    "repetitions": args.reps,
                    "median_ms": "",
                    "mean_ms": "",
                    "stdev_ms": "",
                    "achieved_tflops": "",
                    "status": "FAILED",
                    "error": f"{type(e).__name__}: {e}"
                })

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {output_file}")
    print(f"Total configurations: {len(rows)}")


if __name__ == "__main__":
    main()