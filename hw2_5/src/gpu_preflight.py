import sys
import platform

try:
    import torch
except ImportError:
    print("=" * 70)
    print("HW2.5 GPU PRE-FLIGHT CHECK")
    print("=" * 70)
    print("PyTorch is not installed in this Python environment.")
    print("Do not install anything yet on the GPU workstation.")
    print("First inspect the workstation environment.")
    sys.exit(1)


# ------------------------------------------------------------
# HW2.5 reproducibility settings
# ------------------------------------------------------------

from reproducibility import (
    SID4,
    SEED,
    SLICE,
    HP_ID,
    CLS_A,
    CLS_B,
    set_reproducibility,
)

set_reproducibility()


def divider():
    print("=" * 70)


# ------------------------------------------------------------
# Assignment parameters
# ------------------------------------------------------------

divider()
print("HW2.5 GPU PRE-FLIGHT CHECK")
divider()

print("PERSONAL PARAMETERS")
print("-" * 70)

print(f"SID4                 : {SID4}")
print(f"SEED                 : {SEED}")
print(f"SLICE                : {SLICE}")
print(f"HP_ID                : {HP_ID}")
print(f"CLS_A                : {CLS_A}")
print(f"CLS_B                : {CLS_B}")

print()
print(
    "SEED is applied to Python, NumPy, and PyTorch "
    "through reproducibility.py."
)

# ------------------------------------------------------------
# Software environment
# ------------------------------------------------------------

divider()
print("SOFTWARE ENVIRONMENT")
divider()

print(
    f"Python version       : "
    f"{sys.version.split()[0]}"
)

print(
    f"Operating system     : "
    f"{platform.platform()}"
)

print(
    f"PyTorch version      : "
    f"{torch.__version__}"
)

print(
    f"PyTorch CUDA runtime : "
    f"{torch.version.cuda}"
)

print(
    f"CUDA available       : "
    f"{torch.cuda.is_available()}"
)


# ------------------------------------------------------------
# Stop safely if CUDA is unavailable
# ------------------------------------------------------------

if not torch.cuda.is_available():

    print()
    print(
        "No CUDA-capable NVIDIA GPU was detected."
    )

    print(
        "This is expected when the script is run "
        "on a non-NVIDIA preparation machine."
    )

    print(
        "Run this script again on the RTX 4090 "
        "or RTX 5090 workstation."
    )

    sys.exit(0)


# ------------------------------------------------------------
# CUDA device information
# ------------------------------------------------------------

divider()
print("CUDA DEVICE INFORMATION")
divider()

device_index = torch.cuda.current_device()

device_name = torch.cuda.get_device_name(
    device_index
)

props = torch.cuda.get_device_properties(
    device_index
)

print(
    f"CUDA device index    : "
    f"{device_index}"
)

print(
    f"GPU name             : "
    f"{device_name}"
)

print(
    f"Compute capability   : "
    f"{props.major}.{props.minor}"
)

print(
    f"Total VRAM           : "
    f"{props.total_memory / (1024 ** 3):.2f} GiB"
)


# ------------------------------------------------------------
# BF16 support
# ------------------------------------------------------------

try:

    print(
        f"BF16 supported       : "
        f"{torch.cuda.is_bf16_supported()}"
    )

except Exception as e:

    print(
        f"BF16 support check   : "
        f"unavailable ({e})"
    )


# ------------------------------------------------------------
# Float32 / TF32 configuration
# ------------------------------------------------------------

divider()
print("FLOAT32 / TF32 CONFIGURATION")
divider()

try:

    print(
        "Float32 matmul precision:",
        torch.get_float32_matmul_precision(),
    )

except Exception as e:

    print(
        "Could not read float32 matmul precision:",
        e,
    )


try:

    print(
        "TF32 matmul allowed:",
        torch.backends.cuda.matmul.allow_tf32,
    )

except Exception as e:

    print(
        "Could not inspect TF32 setting:",
        e,
    )


# ------------------------------------------------------------
# FP32 matrix multiplication test
# ------------------------------------------------------------

divider()
print("FP32 MATRIX MULTIPLICATION TEST")
divider()

try:

    a = torch.randn(
        (1024, 1024),
        device="cuda",
        dtype=torch.float32,
    )

    b = torch.randn(
        (1024, 1024),
        device="cuda",
        dtype=torch.float32,
    )

    c = a @ b

    torch.cuda.synchronize()

    print(
        "FP32 matrix multiplication: SUCCESS"
    )

    del a, b, c

    torch.cuda.empty_cache()

except Exception as e:

    print(
        "FP32 matrix multiplication: FAILED"
    )

    print(
        f"{type(e).__name__}: {e}"
    )


# ------------------------------------------------------------
# FP16 matrix multiplication test
# ------------------------------------------------------------

divider()
print("FP16 MATRIX MULTIPLICATION TEST")
divider()

try:

    a = torch.randn(
        (1024, 1024),
        device="cuda",
        dtype=torch.float16,
    )

    b = torch.randn(
        (1024, 1024),
        device="cuda",
        dtype=torch.float16,
    )

    c = a @ b

    torch.cuda.synchronize()

    print(
        "FP16 matrix multiplication: SUCCESS"
    )

    del a, b, c

    torch.cuda.empty_cache()

except Exception as e:

    print(
        "FP16 matrix multiplication: FAILED"
    )

    print(
        f"{type(e).__name__}: {e}"
    )


# ------------------------------------------------------------
# BF16 matrix multiplication test
# ------------------------------------------------------------

divider()
print("BF16 MATRIX MULTIPLICATION TEST")
divider()

try:

    a = torch.randn(
        (1024, 1024),
        device="cuda",
        dtype=torch.bfloat16,
    )

    b = torch.randn(
        (1024, 1024),
        device="cuda",
        dtype=torch.bfloat16,
    )

    c = a @ b

    torch.cuda.synchronize()

    print(
        "BF16 matrix multiplication: SUCCESS"
    )

    del a, b, c

    torch.cuda.empty_cache()

except Exception as e:

    print(
        "BF16 matrix multiplication: FAILED"
    )

    print(
        f"{type(e).__name__}: {e}"
    )


# ------------------------------------------------------------
# GPU memory status
# ------------------------------------------------------------

divider()
print("GPU MEMORY STATUS")
divider()

print(
    f"Allocated memory     : "
    f"{torch.cuda.memory_allocated() / (1024 ** 2):.2f} MiB"
)

print(
    f"Reserved memory      : "
    f"{torch.cuda.memory_reserved() / (1024 ** 2):.2f} MiB"
)


# ------------------------------------------------------------
# Final status
# ------------------------------------------------------------

divider()
print("PRE-FLIGHT CHECK COMPLETE")
divider()

print(
    f"Reproducibility seed : {SEED}"
)

print(
    f"GPU tested           : {device_name}"
)

print(
    "FP32, FP16, and BF16 test operations completed."
)

divider()