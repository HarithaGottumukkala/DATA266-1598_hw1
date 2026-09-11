import subprocess
import torch
from reproducibility import set_reproducibility

set_reproducibility()
import torch.nn.functional as F

from torch.nn.attention import SDPBackend, sdpa_kernel


BATCH_SIZE = 4
NUM_HEADS = 8
SEQUENCE_LENGTH = 512
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


print("=" * 78)
print("HW2.5 PART D — SDPA BACKEND PROBE")
print("=" * 78)

print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("GPU UUID:", get_gpu_uuid())

print()
print("Test configuration:")
print("Batch size:", BATCH_SIZE)
print("Heads:", NUM_HEADS)
print("Sequence length:", SEQUENCE_LENGTH)
print("Head dimension:", HEAD_DIM)
print("Precision: BF16")

print("=" * 78)

q = torch.randn(
    BATCH_SIZE,
    NUM_HEADS,
    SEQUENCE_LENGTH,
    HEAD_DIM,
    device="cuda",
    dtype=DTYPE
)

k = torch.randn_like(q)
v = torch.randn_like(q)


backends = [
    ("FLASH_ATTENTION", SDPBackend.FLASH_ATTENTION),
    ("EFFICIENT_ATTENTION", SDPBackend.EFFICIENT_ATTENTION),
    ("CUDNN_ATTENTION", SDPBackend.CUDNN_ATTENTION),
    ("MATH", SDPBackend.MATH),
]


for name, backend in backends:

    print()
    print("-" * 78)
    print("Testing:", name)

    try:

        torch.cuda.synchronize()

        with torch.inference_mode():

            with sdpa_kernel(backend):

                output = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p=0.0,
                    is_causal=False
                )

        torch.cuda.synchronize()

        print("Status: SUCCESS")
        print("Output shape:", tuple(output.shape))
        print("Output dtype:", output.dtype)

        del output

    except Exception as e:

        print("Status: FAILED")
        print("Exception:", type(e).__name__)
        print("Message:", str(e))


del q, k, v

torch.cuda.empty_cache()

print()
print("=" * 78)
print("BACKEND PROBE COMPLETE")
print("=" * 78)
