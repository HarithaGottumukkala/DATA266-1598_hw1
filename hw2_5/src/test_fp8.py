import subprocess
import torch


def gpu_uuid():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid",
            "--format=csv,noheader"
        ],
        capture_output=True,
        text=True
    )
    return result.stdout.strip()


print("=" * 75)
print("HW2.5 PART B4 — LOWER PRECISION / FP8 PROBE")
print("=" * 75)

print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    print("RESULT: FAILED — CUDA GPU unavailable")
    raise SystemExit

device = torch.cuda.current_device()
props = torch.cuda.get_device_properties(device)

print("GPU:", torch.cuda.get_device_name(device))
print("GPU UUID:", gpu_uuid())
print("Compute capability:", f"{props.major}.{props.minor}")

print()
print("FP8 API availability:")
print(
    "torch.float8_e4m3fn:",
    hasattr(torch, "float8_e4m3fn")
)
print(
    "torch.float8_e5m2:",
    hasattr(torch, "float8_e5m2")
)
print(
    "torch._scaled_mm:",
    hasattr(torch, "_scaled_mm")
)
print(
    "torch.nn.functional.scaled_mm:",
    hasattr(torch.nn.functional, "scaled_mm")
)

print()
print("Attempting FP8 E4M3 matrix multiplication...")

try:
    N = 1024

    # Start with BF16 tensors.
    a_hp = torch.randn(
        N, N,
        device="cuda",
        dtype=torch.bfloat16
    )

    b_hp = torch.randn(
        N, N,
        device="cuda",
        dtype=torch.bfloat16
    )

    fp8_dtype = torch.float8_e4m3fn
    fp8_max = torch.finfo(fp8_dtype).max

    # Per-tensor quantization scales.
    quant_a = fp8_max / a_hp.abs().max()
    quant_b = fp8_max / b_hp.abs().max()

    a_fp8 = (a_hp * quant_a).to(fp8_dtype)

    # Store B so that the matrix supplied to scaled_mm has
    # a layout compatible with the FP8 GEMM path.
    b_fp8 = (
        (b_hp.T * quant_b)
        .to(fp8_dtype)
        .contiguous()
        .T
    )

    # Dequantization scales.
    scale_a = quant_a.reciprocal().float()
    scale_b = quant_b.reciprocal().float()

    torch.cuda.synchronize()

    if hasattr(torch, "_scaled_mm"):
        try:
            output = torch._scaled_mm(
                a_fp8,
                b_fp8,
                scale_a=scale_a,
                scale_b=scale_b,
                out_dtype=torch.bfloat16
            )

            torch.cuda.synchronize()

            print("API used: torch._scaled_mm")
            print("Output shape:", tuple(output.shape))
            print("Output dtype:", output.dtype)
            print("FP8 E4M3 GEMM: SUCCESS")

        except TypeError:
            # Some PyTorch builds use output_dtype instead.
            output = torch._scaled_mm(
                a_fp8,
                b_fp8,
                scale_a=scale_a,
                scale_b=scale_b,
                output_dtype=torch.bfloat16
            )

            torch.cuda.synchronize()

            print("API used: torch._scaled_mm")
            print("Output shape:", tuple(output.shape))
            print("Output dtype:", output.dtype)
            print("FP8 E4M3 GEMM: SUCCESS")

    else:
        print("FP8 E4M3 GEMM: NOT TESTED")
        print("Reason: torch._scaled_mm is not exposed by this build.")

except Exception as e:
    print()
    print("FP8 E4M3 GEMM: FAILED")
    print("Exception type:", type(e).__name__)
    print("Exception message:", str(e))

finally:
    torch.cuda.empty_cache()

print()
print("=" * 75)
print("FP8 PROBE COMPLETE")
print("=" * 75)