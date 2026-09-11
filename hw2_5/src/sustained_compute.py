import argparse
import csv
import subprocess
import threading
import time
from datetime import datetime

import torch


MATRIX_SIZE = 8192
DTYPE = torch.bfloat16

TELEMETRY_INTERVAL_SECONDS = 5
DEFAULT_DURATION_SECONDS = 1200


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


def query_telemetry():

    query = (
        "timestamp,"
        "uuid,"
        "clocks.sm,"
        "clocks.mem,"
        "temperature.gpu,"
        "power.draw,"
        "power.limit,"
        "utilization.gpu,"
        "memory.used,"
        "memory.total"
    )

    result = subprocess.run(
        [
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    values = [
        value.strip()
        for value in result.stdout.strip().split(",")
    ]

    return {
        "nvidia_timestamp": values[0],
        "gpu_uuid": values[1],
        "sm_clock_mhz": values[2],
        "memory_clock_mhz": values[3],
        "temperature_c": values[4],
        "power_draw_w": values[5],
        "power_limit_w": values[6],
        "gpu_utilization_percent": values[7],
        "memory_used_mib": values[8],
        "memory_total_mib": values[9],
    }


def telemetry_worker(
    output_file,
    stop_event,
    experiment_start,
):

    fieldnames = [
        "elapsed_seconds",
        "nvidia_timestamp",
        "gpu_uuid",
        "sm_clock_mhz",
        "memory_clock_mhz",
        "temperature_c",
        "power_draw_w",
        "power_limit_w",
        "gpu_utilization_percent",
        "memory_used_mib",
        "memory_total_mib",
    ]

    with open(
        output_file,
        "w",
        newline="",
        buffering=1,
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        while not stop_event.is_set():

            try:

                sample = query_telemetry()

                sample["elapsed_seconds"] = (
                    time.monotonic()
                    - experiment_start
                )

                writer.writerow(sample)
                f.flush()

                print(
                    "[Telemetry] "
                    f"t={sample['elapsed_seconds']:.1f}s | "
                    f"clock={sample['sm_clock_mhz']} MHz | "
                    f"temp={sample['temperature_c']} C | "
                    f"power={sample['power_draw_w']} W | "
                    f"util={sample['gpu_utilization_percent']}%"
                )

            except Exception as e:

                print(
                    "[Telemetry warning]",
                    type(e).__name__,
                    str(e),
                )

            stop_event.wait(
                TELEMETRY_INTERVAL_SECONDS
            )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gpu-label",
        required=True,
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=DEFAULT_DURATION_SECONDS,
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA GPU is unavailable."
        )

    gpu_name = torch.cuda.get_device_name(0)
    gpu_uuid = get_gpu_uuid()

    telemetry_file = (
        f"raw/thermal/"
        f"thermal_{args.gpu_label}.csv"
    )

    throughput_file = (
        f"raw/thermal/"
        f"throughput_{args.gpu_label}.csv"
    )

    print("=" * 80)
    print("HW2.5 PART E - 20 MINUTE SUSTAINED COMPUTE TEST")
    print("=" * 80)

    print(
        "Start time          :",
        datetime.now().isoformat()
    )

    print(
        "GPU                 :",
        gpu_name
    )

    print(
        "GPU UUID            :",
        gpu_uuid
    )

    print(
        "Matrix size         :",
        MATRIX_SIZE
    )

    print(
        "Precision           :",
        "BF16"
    )

    print(
        "Duration            :",
        args.duration,
        "seconds"
    )

    print(
        "Telemetry interval  :",
        TELEMETRY_INTERVAL_SECONDS,
        "seconds"
    )

    print("=" * 80)

    torch.manual_seed(1598)

    a = torch.randn(
        MATRIX_SIZE,
        MATRIX_SIZE,
        device="cuda",
        dtype=DTYPE,
    )

    b = torch.randn(
        MATRIX_SIZE,
        MATRIX_SIZE,
        device="cuda",
        dtype=DTYPE,
    )

    # Initial warm-up
    print()
    print("Warming up GPU...")

    with torch.inference_mode():

        for _ in range(10):
            c = torch.matmul(a, b)

    torch.cuda.synchronize()

    del c

    print("Warm-up complete.")
    print()

    stop_event = threading.Event()

    experiment_start = time.monotonic()

    telemetry_thread = threading.Thread(
        target=telemetry_worker,
        args=(
            telemetry_file,
            stop_event,
            experiment_start,
        ),
        daemon=True,
    )

    telemetry_thread.start()

    throughput_fields = [
        "elapsed_seconds",
        "window_seconds",
        "matmuls",
        "tflops",
    ]

    total_matmuls = 0

    try:

        with open(
            throughput_file,
            "w",
            newline="",
            buffering=1,
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=throughput_fields,
            )

            writer.writeheader()

            print(
                "Starting sustained BF16 "
                "matrix multiplication..."
            )

            print()

            while (
                time.monotonic()
                - experiment_start
                < args.duration
            ):

                window_start = time.monotonic()

                matmuls = 0

                # Perform a block of matrix multiplications.
                with torch.inference_mode():

                    for _ in range(20):

                        c = torch.matmul(
                            a,
                            b,
                        )

                        matmuls += 1

                torch.cuda.synchronize()

                window_end = time.monotonic()

                window_seconds = (
                    window_end
                    - window_start
                )

                elapsed_seconds = (
                    window_end
                    - experiment_start
                )

                total_matmuls += matmuls

                flops_per_matmul = (
                    2
                    * MATRIX_SIZE
                    * MATRIX_SIZE
                    * MATRIX_SIZE
                )

                total_flops = (
                    flops_per_matmul
                    * matmuls
                )

                tflops = (
                    total_flops
                    / window_seconds
                    / 1e12
                )

                writer.writerow({
                    "elapsed_seconds":
                        elapsed_seconds,

                    "window_seconds":
                        window_seconds,

                    "matmuls":
                        matmuls,

                    "tflops":
                        tflops,
                })

                f.flush()

                if (
                    int(elapsed_seconds) % 30
                    < 2
                ):

                    print(
                        f"[Compute] "
                        f"elapsed={elapsed_seconds:.1f}s | "
                        f"throughput={tflops:.2f} TFLOPS"
                    )

                del c

    finally:

        stop_event.set()

        telemetry_thread.join(
            timeout=10
        )

        torch.cuda.synchronize()

        del a
        del b

        torch.cuda.empty_cache()

    actual_duration = (
        time.monotonic()
        - experiment_start
    )

    print()
    print("=" * 80)
    print("SUSTAINED COMPUTE TEST COMPLETE")
    print("=" * 80)

    print(
        "Actual duration     :",
        f"{actual_duration:.2f} seconds"
    )

    print(
        "Total matmuls       :",
        total_matmuls
    )

    print(
        "Telemetry saved     :",
        telemetry_file
    )

    print(
        "Throughput saved    :",
        throughput_file
    )

    print(
        "End time            :",
        datetime.now().isoformat()
    )


if __name__ == "__main__":
    main()