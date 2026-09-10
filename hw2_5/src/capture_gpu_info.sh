#!/bin/bash

# HW2.5 — GPU Environment and Provenance Capture
# Run this script from inside the hw2_5 directory on the GPU workstation.

echo "======================================================================"
echo "HW2.5 — GPU INFORMATION CAPTURE"
echo "======================================================================"
echo

# Make sure nvidia-smi exists before continuing.
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi was not found."
    echo "This script must be run on the RTX 4090 or RTX 5090 lab workstation."
    exit 1
fi

mkdir -p raw/gpu_info

echo "Detected NVIDIA GPU:"
nvidia-smi --query-gpu=name,uuid --format=csv,noheader
echo

read -p "Enter GPU label (4090 or 5090): " GPU_LABEL

if [[ "$GPU_LABEL" != "4090" && "$GPU_LABEL" != "5090" ]]; then
    echo "ERROR: Please enter exactly 4090 or 5090."
    exit 1
fi

NVIDIA_Q_FILE="raw/gpu_info/nvidia_smi_${GPU_LABEL}.txt"
NVIDIA_SUMMARY_FILE="raw/gpu_info/nvidia_smi_summary_${GPU_LABEL}.txt"
ENV_FILE="raw/gpu_info/environment_${GPU_LABEL}.txt"

echo
echo "Capturing complete nvidia-smi -q output..."

nvidia-smi -q > "$NVIDIA_Q_FILE"

echo "Capturing GPU summary..."

{
    echo "HW2.5 GPU SUMMARY"
    echo "Captured: $(date)"
    echo
    echo "GPU LABEL:"
    echo "$GPU_LABEL"
    echo
    echo "GPU NAME AND UUID:"
    nvidia-smi --query-gpu=name,uuid --format=csv,noheader
    echo
    echo "DRIVER VERSION:"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader
    echo
    echo "VRAM:"
    nvidia-smi --query-gpu=memory.total --format=csv,noheader
    echo
    echo "POWER LIMIT:"
    nvidia-smi --query-gpu=power.limit --format=csv,noheader
    echo
    echo "FULL NVIDIA-SMI HEADER:"
    nvidia-smi
} > "$NVIDIA_SUMMARY_FILE"

echo "Capturing software environment..."

{
    echo "======================================================================"
    echo "HW2.5 SOFTWARE ENVIRONMENT"
    echo "======================================================================"
    echo

    echo "Timestamp:"
    date

    echo
    echo "Hostname:"
    hostname

    echo
    echo "Operating System:"
    uname -a

    echo
    echo "GPU:"
    nvidia-smi --query-gpu=name,uuid --format=csv,noheader

    echo
    echo "NVIDIA Driver:"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader

    echo
    echo "NVIDIA-SMI:"
    nvidia-smi

    echo
    echo "NVCC:"
    if command -v nvcc >/dev/null 2>&1; then
        nvcc --version
    else
        echo "nvcc not found"
    fi

    echo
    echo "Python:"
    python3 --version 2>&1

    echo
    echo "PyTorch / CUDA:"
    python3 - <<'PY'
try:
    import torch

    print("PyTorch version:", torch.__version__)
    print("PyTorch CUDA runtime:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)

        print("GPU name:", torch.cuda.get_device_name(device))
        print("Compute capability:", f"{props.major}.{props.minor}")
        print("Total VRAM (GiB):", round(props.total_memory / (1024**3), 2))
        print("BF16 supported:", torch.cuda.is_bf16_supported())

except ImportError:
    print("PyTorch is not installed in this Python environment.")
except Exception as e:
    print("PyTorch inspection failed:", type(e).__name__, str(e))
PY

} > "$ENV_FILE"

echo
echo "======================================================================"
echo "CAPTURE COMPLETE"
echo "======================================================================"

echo
echo "Created:"
echo "  $NVIDIA_Q_FILE"
echo "  $NVIDIA_SUMMARY_FILE"
echo "  $ENV_FILE"

echo
echo "IMPORTANT:"
echo "The nvidia_smi_${GPU_LABEL}.txt file contains the complete"
echo "nvidia-smi -q output required for Part A."
echo
