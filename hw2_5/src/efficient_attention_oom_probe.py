import argparse
import csv
import gc
import subprocess
from datetime import datetime

import torch
from reproducibility import set_reproducibility

set_reproducibility()
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


BATCH_SIZE = 4
NUM_HEADS = 8
HEAD_DIM = 64
DTYPE = torch.bfloat16


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


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

    try:
        torch.cuda.synchronize()
    except Exception:
        pass


def efficient_attention(q, k, v):

    with sdpa_kernel(
        SDPBackend.EFFICIENT_ATTENTION
    ):
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=False
        )

    return output


def probe(length):

    cleanup()

    q = None
    k = None
    v = None
    output = None

    try:

        q = torch.randn(
            BATCH_SIZE,
            NUM_HEADS,
            length,
            HEAD_DIM,
            device="cuda",
            dtype=DTYPE
        )

        k = torch.randn_like(q)
        v = torch.randn_like(q)

        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()

        baseline = torch.cuda.memory_allocated()

        start = torch.cuda.Event(
            enable_timing=True
        )

        end = torch.cuda.Event(
            enable_timing=True
        )

        with torch.inference_mode():

            start.record()

            output = efficient_attention(
                q,
                k,
                v
            )

            end.record()

            torch.cuda.synchronize()

        latency_ms = start.elapsed_time(end)

        peak_allocated = (
            torch.cuda.max_memory_allocated()
        )

        peak_reserved = (
            torch.cuda.max_memory_reserved()
        )

        return {
            "status": "SUCCESS",
            "baseline_gib":
                baseline / (1024 ** 3),
            "peak_allocated_gib":
                peak_allocated / (1024 ** 3),
            "peak_reserved_gib":
                peak_reserved / (1024 ** 3),
            "latency_ms":
                latency_ms,
            "error": ""
        }

    except torch.OutOfMemoryError as e:

        return {
            "status": "OOM",
            "baseline_gib": "",
            "peak_allocated_gib": "",
            "peak_reserved_gib": "",
            "latency_ms": "",
            "error":
                f"{type(e).__name__}: {e}"
        }

    except Exception as e:

        message = str(e)

        if "out of memory" in message.lower():
            status = "OOM"
        else:
            status = "FAILED"

        return {
            "status": status,
            "baseline_gib": "",
            "peak_allocated_gib": "",
            "peak_reserved_gib": "",
            "latency_ms": "",
            "error":
                f"{type(e).__name__}: {message}"
        }

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


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gpu-label",
        required=True
    )

    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        required=True
    )

    parser.add_argument(
        "--tag",
        default="coarse"
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU unavailable."
        )

    gpu_name = torch.cuda.get_device_name(0)
    gpu_uuid = get_gpu_uuid()

    props = torch.cuda.get_device_properties(0)

    visible_vram_gib = (
        props.total_memory / (1024 ** 3)
    )

    print("=" * 78)
    print(
        "HW2.5 PART D - "
        "EFFICIENT ATTENTION OOM PROBE"
    )
    print("=" * 78)

    print("Timestamp:", datetime.now().isoformat())
    print("GPU:", gpu_name)
    print("GPU UUID:", gpu_uuid)

    print(
        "Visible VRAM:",
        f"{visible_vram_gib:.3f} GiB"
    )

    print("Backend: EFFICIENT_ATTENTION")
    print("Batch size:", BATCH_SIZE)
    print("Heads:", NUM_HEADS)
    print("Head dimension:", HEAD_DIM)
    print("Precision: BF16")
    print("Test lengths:", args.sizes)

    print("=" * 78)

    rows = []

    for length in args.sizes:

        print()
        print("-" * 78)
        print(f"Testing L={length} ...")

        result = probe(length)

        print(
            "Status:",
            result["status"]
        )

        if result["status"] == "SUCCESS":

            print(
                "Baseline allocated:",
                f"{result['baseline_gib']:.3f} GiB"
            )

            print(
                "Peak allocated:",
                f"{result['peak_allocated_gib']:.3f} GiB"
            )

            print(
                "Peak reserved:",
                f"{result['peak_reserved_gib']:.3f} GiB"
            )

            print(
                "Forward latency:",
                f"{result['latency_ms']:.3f} ms"
            )

        else:

            print(
                "Error:",
                result["error"]
            )

        rows.append({

            "timestamp":
                datetime.now().isoformat(),

            "gpu_label":
                args.gpu_label,

            "gpu_name":
                gpu_name,

            "gpu_uuid":
                gpu_uuid,

            "backend":
                "efficient_attention",

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

            "baseline_memory_gib":
                result["baseline_gib"],

            "peak_allocated_gib":
                result["peak_allocated_gib"],

            "peak_reserved_gib":
                result["peak_reserved_gib"],

            "latency_ms":
                result["latency_ms"],

            "status":
                result["status"],

            "error":
                result["error"]
        })

        # Stop automatically after the first failure/OOM.
        if result["status"] != "SUCCESS":
            print()
            print(
                "Stopping search after first "
                "unsuccessful sequence length."
            )
            break

    output_file = (
        f"raw/attention/"
        f"efficient_oom_{args.tag}_"
        f"{args.gpu_label}.csv"
    )

    with open(
        output_file,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 78)
    print("EFFICIENT ATTENTION OOM PROBE COMPLETE")
    print("=" * 78)

    print(
        "Saved:",
        output_file
    )


if __name__ == "__main__":
    main()
