import argparse
import csv
import gc
import math
import statistics
import subprocess
from datetime import datetime

import torch
from reproducibility import set_reproducibility

set_reproducibility()
import torch.nn.functional as F


# ============================================================
# Fixed experiment configuration
# ============================================================

BATCH_SIZE = 4
NUM_HEADS = 8
HEAD_DIM = 64
DTYPE = torch.bfloat16

WARMUP = 2
REPETITIONS = 5


# ============================================================
# GPU information
# ============================================================

def get_gpu_uuid():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    return result.stdout.strip()


# ============================================================
# GPU cleanup
# ============================================================

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

    try:
        torch.cuda.synchronize()
    except Exception:
        pass


# ============================================================
# Naive scaled dot-product attention
# ============================================================

def naive_attention(q, k, v):
    """
    Naive attention explicitly materializes the full
    [B, H, L, L] attention-score matrix.
    """

    # Q @ K^T
    scores = torch.matmul(
        q,
        k.transpose(-2, -1),
    )

    # Scaling
    scores = scores / math.sqrt(HEAD_DIM)

    # Softmax probabilities
    probabilities = torch.softmax(
        scores,
        dim=-1,
    )

    # Attention probabilities @ V
    output = torch.matmul(
        probabilities,
        v,
    )

    return output


# ============================================================
# Memory-efficient scaled dot-product attention
# ============================================================

def fused_attention(q, k, v):
    """
    Use PyTorch's Efficient Attention SDPA backend.

    The Flash Attention backend was tested separately on this
    workstation and failed because this PyTorch build did not
    provide a compatible Flash kernel.

    The Efficient Attention backend was tested successfully,
    so it is used here as the assignment's memory-efficient
    attention implementation.
    """

    from torch.nn.attention import (
        SDPBackend,
        sdpa_kernel,
    )

    # Explicitly force the Efficient Attention backend.
    # This prevents PyTorch from silently falling back
    # to the ordinary math implementation.
    with sdpa_kernel(
        SDPBackend.EFFICIENT_ATTENTION
    ):
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=False,
        )

    return output


# ============================================================
# Benchmark one sequence length
# ============================================================

def benchmark_sequence_length(
    sequence_length,
    mode,
):
    cleanup()

    q = None
    k = None
    v = None
    output = None

    try:

        # ----------------------------------------------------
        # Allocate Q, K and V
        # Shape: [B, H, L, D]
        # ----------------------------------------------------

        q = torch.randn(
            BATCH_SIZE,
            NUM_HEADS,
            sequence_length,
            HEAD_DIM,
            device="cuda",
            dtype=DTYPE,
        )

        k = torch.randn(
            BATCH_SIZE,
            NUM_HEADS,
            sequence_length,
            HEAD_DIM,
            device="cuda",
            dtype=DTYPE,
        )

        v = torch.randn(
            BATCH_SIZE,
            NUM_HEADS,
            sequence_length,
            HEAD_DIM,
            device="cuda",
            dtype=DTYPE,
        )

        torch.cuda.synchronize()

        # ----------------------------------------------------
        # Select implementation
        # ----------------------------------------------------

        if mode == "naive":

            operation = lambda: naive_attention(
                q,
                k,
                v,
            )

            backend = "explicit_naive"

        elif mode == "fused":

            operation = lambda: fused_attention(
                q,
                k,
                v,
            )

            backend = "efficient_attention"

        else:

            raise ValueError(
                f"Unknown mode: {mode}"
            )

        # ----------------------------------------------------
        # Warm-up
        # ----------------------------------------------------

        with torch.inference_mode():

            for _ in range(WARMUP):

                output = operation()

                torch.cuda.synchronize()

                del output
                output = None

        # ----------------------------------------------------
        # Reset peak-memory statistics
        #
        # Q, K and V remain allocated.
        # Therefore our peak includes the inputs plus
        # temporary forward-pass allocations.
        # ----------------------------------------------------

        torch.cuda.reset_peak_memory_stats()

        baseline_allocated = (
            torch.cuda.memory_allocated()
        )

        # ----------------------------------------------------
        # Timed repetitions
        # ----------------------------------------------------

        times_ms = []

        with torch.inference_mode():

            for _ in range(REPETITIONS):

                start = torch.cuda.Event(
                    enable_timing=True
                )

                end = torch.cuda.Event(
                    enable_timing=True
                )

                start.record()

                output = operation()

                end.record()

                torch.cuda.synchronize()

                elapsed_ms = (
                    start.elapsed_time(end)
                )

                times_ms.append(
                    elapsed_ms
                )

                del output
                output = None

        # ----------------------------------------------------
        # Memory measurements
        # ----------------------------------------------------

        peak_allocated = (
            torch.cuda.max_memory_allocated()
        )

        peak_reserved = (
            torch.cuda.max_memory_reserved()
        )

        # ----------------------------------------------------
        # Timing statistics
        # ----------------------------------------------------

        median_ms = statistics.median(
            times_ms
        )

        mean_ms = statistics.mean(
            times_ms
        )

        stdev_ms = (
            statistics.stdev(times_ms)
            if len(times_ms) > 1
            else 0.0
        )

        # ----------------------------------------------------
        # Successful result
        # ----------------------------------------------------

        return {

            "status":
                "SUCCESS",

            "backend":
                backend,

            "baseline_gib":
                baseline_allocated
                / (1024 ** 3),

            "peak_allocated_gib":
                peak_allocated
                / (1024 ** 3),

            "peak_reserved_gib":
                peak_reserved
                / (1024 ** 3),

            "median_latency_ms":
                median_ms,

            "mean_latency_ms":
                mean_ms,

            "stdev_latency_ms":
                stdev_ms,

            "error":
                "",
        }

    # --------------------------------------------------------
    # Explicit PyTorch CUDA OOM
    # --------------------------------------------------------

    except torch.OutOfMemoryError as e:

        return {

            "status":
                "OOM",

            "backend":
                (
                    "explicit_naive"
                    if mode == "naive"
                    else "efficient_attention"
                ),

            "baseline_gib":
                "",

            "peak_allocated_gib":
                "",

            "peak_reserved_gib":
                "",

            "median_latency_ms":
                "",

            "mean_latency_ms":
                "",

            "stdev_latency_ms":
                "",

            "error":
                f"{type(e).__name__}: {str(e)}",
        }

    # --------------------------------------------------------
    # Other failures
    # --------------------------------------------------------

    except Exception as e:

        message = str(e)

        if "out of memory" in message.lower():

            status = "OOM"

        else:

            status = "FAILED"

        return {

            "status":
                status,

            "backend":
                (
                    "explicit_naive"
                    if mode == "naive"
                    else "efficient_attention"
                ),

            "baseline_gib":
                "",

            "peak_allocated_gib":
                "",

            "peak_reserved_gib":
                "",

            "median_latency_ms":
                "",

            "mean_latency_ms":
                "",

            "stdev_latency_ms":
                "",

            "error":
                f"{type(e).__name__}: {message}",
        }

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    finally:

        if output is not None:
            del output

        if q is not None:
            del q

        if k is not None:
            del k

        if v is not None:
            del v

        cleanup()


# ============================================================
# Main program
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gpu-label",
        required=True,
        help="GPU label such as 4090 or 5090",
    )

    parser.add_argument(
        "--mode",
        choices=[
            "naive",
            "fused",
        ],
        required=True,
    )

    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--tag",
        default="coarse",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # CUDA check
    # --------------------------------------------------------

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA GPU is unavailable."
        )

    gpu_name = (
        torch.cuda.get_device_name(0)
    )

    gpu_uuid = (
        get_gpu_uuid()
    )

    props = (
        torch.cuda.get_device_properties(0)
    )

    total_vram_gib = (
        props.total_memory
        / (1024 ** 3)
    )

    # --------------------------------------------------------
    # Experiment header
    # --------------------------------------------------------

    print("=" * 80)

    print(
        "HW2.5 PART D - "
        "ATTENTION BENCHMARK"
    )

    print("=" * 80)

    print(
        "Timestamp          :",
        datetime.now().isoformat()
    )

    print(
        "GPU                :",
        gpu_name
    )

    print(
        "GPU UUID           :",
        gpu_uuid
    )

    print(
        "Visible VRAM       :",
        f"{total_vram_gib:.3f} GiB"
    )

    print(
        "Mode               :",
        args.mode
    )

    if args.mode == "naive":

        print(
            "Backend            :",
            "explicit_naive"
        )

    else:

        print(
            "Backend            :",
            "efficient_attention"
        )

    print(
        "Batch size         :",
        BATCH_SIZE
    )

    print(
        "Number of heads    :",
        NUM_HEADS
    )

    print(
        "Head dimension     :",
        HEAD_DIM
    )

    print(
        "Precision          :",
        "BF16"
    )

    print(
        "Warmup iterations  :",
        WARMUP
    )

    print(
        "Timed repetitions  :",
        REPETITIONS
    )

    print(
        "Sequence lengths   :",
        args.sizes
    )

    print("=" * 80)

    # --------------------------------------------------------
    # Output CSV filename
    # --------------------------------------------------------

    output_file = (
        f"raw/attention/"
        f"attention_{args.mode}_"
        f"{args.tag}_"
        f"{args.gpu_label}.csv"
    )

    rows = []

    # --------------------------------------------------------
    # Run each requested sequence length
    # --------------------------------------------------------

    for length in args.sizes:

        print()
        print("-" * 80)

        print(
            f"Running "
            f"{args.mode.upper()} "
            f"attention at "
            f"L={length} ..."
        )

        result = (
            benchmark_sequence_length(
                length,
                args.mode,
            )
        )

        # ----------------------------------------------------
        # Display result
        # ----------------------------------------------------

        if result["status"] == "SUCCESS":

            print(
                "Status              :",
                "SUCCESS"
            )

            print(
                "Backend             :",
                result["backend"]
            )

            print(
                "Baseline allocated  :",
                f"{result['baseline_gib']:.3f} GiB"
            )

            print(
                "Peak allocated      :",
                f"{result['peak_allocated_gib']:.3f} GiB"
            )

            print(
                "Peak reserved       :",
                f"{result['peak_reserved_gib']:.3f} GiB"
            )

            print(
                "Median latency      :",
                f"{result['median_latency_ms']:.3f} ms"
            )

            print(
                "Mean latency        :",
                f"{result['mean_latency_ms']:.3f} ms"
            )

            print(
                "Latency std dev     :",
                f"{result['stdev_latency_ms']:.3f} ms"
            )

        else:

            print(
                "Status              :",
                result["status"]
            )

            print(
                "Backend             :",
                result["backend"]
            )

            print(
                "Error               :",
                result["error"]
            )

        # ----------------------------------------------------
        # Save row
        # ----------------------------------------------------

        rows.append({

            "timestamp":
                datetime.now().isoformat(),

            "gpu_label":
                args.gpu_label,

            "gpu_name":
                gpu_name,

            "gpu_uuid":
                gpu_uuid,

            "visible_vram_gib":
                total_vram_gib,

            "mode":
                args.mode,

            "backend":
                result["backend"],

            "sequence_length":
                length,

            "batch_size":
                BATCH_SIZE,

            "num_heads":
                NUM_HEADS,

            "head_dim":
                HEAD_DIM,

            "dtype":
                "BF16",

            "warmup":
                WARMUP,

            "repetitions":
                REPETITIONS,

            "baseline_memory_gib":
                result["baseline_gib"],

            "peak_allocated_gib":
                result["peak_allocated_gib"],

            "peak_reserved_gib":
                result["peak_reserved_gib"],

            "median_latency_ms":
                result["median_latency_ms"],

            "mean_latency_ms":
                result["mean_latency_ms"],

            "stdev_latency_ms":
                result["stdev_latency_ms"],

            "status":
                result["status"],

            "error":
                result["error"],
        })

    # --------------------------------------------------------
    # Write CSV
    # --------------------------------------------------------

    with open(
        output_file,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print()
    print("=" * 80)

    print(
        "ATTENTION BENCHMARK COMPLETE"
    )

    print("=" * 80)

    print(
        "Saved:",
        output_file
    )


if __name__ == "__main__":
    main()
