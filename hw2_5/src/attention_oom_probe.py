import argparse
import csv
import gc
import math
import subprocess
from datetime import datetime

import torch
from reproducibility import set_reproducibility

set_reproducibility()
# ------------------------------------------------------------
# Fixed attention configuration
# ------------------------------------------------------------

BATCH_SIZE = 4
NUM_HEADS = 8
HEAD_DIM = 64
DTYPE = torch.bfloat16

# Limit this PyTorch process to the visible GPU-memory capacity.
MEMORY_FRACTION = 1.0


# ------------------------------------------------------------
# GPU information
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Cleanup helper
# ------------------------------------------------------------

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

    try:
        torch.cuda.synchronize()
    except Exception:
        pass


# ------------------------------------------------------------
# Naive scaled dot-product attention
# ------------------------------------------------------------

def naive_attention(q, k, v):

    # Q @ K^T
    #
    # Shape:
    # [B, H, L, D] @ [B, H, D, L]
    # ->
    # [B, H, L, L]
    #
    # This explicitly materializes the full L x L
    # attention-score matrix.
    scores = torch.matmul(
        q,
        k.transpose(-2, -1)
    )

    # Scale by sqrt(head dimension).
    scores = scores / math.sqrt(HEAD_DIM)

    # Convert scores to attention probabilities.
    probs = torch.softmax(
        scores,
        dim=-1
    )

    # Attention probabilities @ values.
    output = torch.matmul(
        probs,
        v
    )

    return output


# ------------------------------------------------------------
# Test one sequence length
# ------------------------------------------------------------

def probe(length):

    cleanup()

    q = None
    k = None
    v = None
    output = None

    try:

        # ----------------------------------------------------
        # Allocate Q, K and V
        # ----------------------------------------------------

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

        # Reset peak statistics immediately before
        # the attention forward pass.
        torch.cuda.reset_peak_memory_stats()

        baseline_allocated = (
            torch.cuda.memory_allocated()
        )

        # ----------------------------------------------------
        # One forward pass
        # ----------------------------------------------------

        with torch.inference_mode():

            output = naive_attention(
                q,
                k,
                v
            )

            torch.cuda.synchronize()

        # ----------------------------------------------------
        # Memory measurements
        # ----------------------------------------------------

        peak_allocated = (
            torch.cuda.max_memory_allocated()
        )

        peak_reserved = (
            torch.cuda.max_memory_reserved()
        )

        return {
            "status": "SUCCESS",

            "baseline_memory_gib":
                baseline_allocated / (1024 ** 3),

            "peak_allocated_gib":
                peak_allocated / (1024 ** 3),

            "peak_reserved_gib":
                peak_reserved / (1024 ** 3),

            "error": ""
        }

    except torch.OutOfMemoryError as e:

        return {
            "status": "OOM",
            "baseline_memory_gib": "",
            "peak_allocated_gib": "",
            "peak_reserved_gib": "",
            "error":
                f"{type(e).__name__}: {str(e)}"
        }

    except Exception as e:

        message = str(e)

        if "out of memory" in message.lower():
            status = "OOM"
        else:
            status = "FAILED"

        return {
            "status": status,
            "baseline_memory_gib": "",
            "peak_allocated_gib": "",
            "peak_reserved_gib": "",
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


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

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
        default="boundary"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # CUDA check
    # --------------------------------------------------------

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU unavailable."
        )

    device = 0

    gpu_name = torch.cuda.get_device_name(device)

    gpu_uuid = get_gpu_uuid()

    props = torch.cuda.get_device_properties(device)

    total_vram_bytes = props.total_memory

    total_vram_gib = (
        total_vram_bytes / (1024 ** 3)
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Cap this process to the GPU's visible memory capacity.
    #
    # This helps prevent the OOM-boundary experiment from
    # continuing far beyond the visible GPU-memory capacity
    # through Windows/WDDM oversubscription behavior.
    # --------------------------------------------------------

    torch.cuda.set_per_process_memory_fraction(
        MEMORY_FRACTION,
        device=device
    )

    allocator_limit_gib = (
        total_vram_gib * MEMORY_FRACTION
    )

    # --------------------------------------------------------
    # Experiment information
    # --------------------------------------------------------

    print("=" * 78)
    print(
        "HW2.5 PART D — "
        "NAIVE ATTENTION OOM BOUNDARY PROBE"
    )
    print("=" * 78)

    print("Timestamp:", datetime.now().isoformat())
    print("GPU:", gpu_name)
    print("GPU UUID:", gpu_uuid)

    print(
        "Visible VRAM:",
        f"{total_vram_gib:.3f} GiB"
    )

    print(
        "PyTorch memory fraction:",
        MEMORY_FRACTION
    )

    print(
        "Allocator limit:",
        f"{allocator_limit_gib:.3f} GiB"
    )

    print("Batch size:", BATCH_SIZE)
    print("Heads:", NUM_HEADS)
    print("Head dimension:", HEAD_DIM)
    print("Precision: BF16")
    print("Test lengths:", args.sizes)

    print("=" * 78)

    rows = []

    # --------------------------------------------------------
    # Run requested sequence lengths
    # --------------------------------------------------------

    for length in args.sizes:

        print()
        print("-" * 78)

        print(
            f"Testing L={length} ..."
        )

        result = probe(length)

        print(
            "Status:",
            result["status"]
        )

        if result["status"] == "SUCCESS":

            print(
                "Baseline allocated:",
                f"{result['baseline_memory_gib']:.3f} GiB"
            )

            print(
                "Peak allocated:",
                f"{result['peak_allocated_gib']:.3f} GiB"
            )

            print(
                "Peak reserved:",
                f"{result['peak_reserved_gib']:.3f} GiB"
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

            "visible_vram_gib":
                total_vram_gib,

            "memory_fraction":
                MEMORY_FRACTION,

            "allocator_limit_gib":
                allocator_limit_gib,

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
                result["baseline_memory_gib"],

            "peak_allocated_gib":
                result["peak_allocated_gib"],

            "peak_reserved_gib":
                result["peak_reserved_gib"],

            "status":
                result["status"],

            "error":
                result["error"]
        })

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    output_file = (
        f"raw/attention/"
        f"naive_oom_{args.tag}_{args.gpu_label}.csv"
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
    print("OOM PROBE COMPLETE")
    print("=" * 78)

    print(
        "Saved:",
        output_file
    )


if __name__ == "__main__":
    main()
