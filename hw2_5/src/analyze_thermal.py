import argparse
import csv
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def to_float(row, key):
    return float(row[key])


def mean(values):
    return statistics.mean(values) if values else float("nan")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gpu-label",
        required=True
    )

    args = parser.parse_args()

    thermal_file = Path(
        f"raw/thermal/thermal_{args.gpu_label}.csv"
    )

    throughput_file = Path(
        f"raw/thermal/throughput_{args.gpu_label}.csv"
    )

    output_file = Path(
        f"raw/thermal/thermal_analysis_{args.gpu_label}.txt"
    )

    clock_plot = Path(
        f"figures/thermal_clock_{args.gpu_label}.png"
    )

    temperature_plot = Path(
        f"figures/thermal_temperature_{args.gpu_label}.png"
    )

    # ========================================================
    # Load telemetry
    # ========================================================

    thermal_rows = read_csv(
        thermal_file
    )

    throughput_rows = read_csv(
        throughput_file
    )

    thermal_times = [
        to_float(row, "elapsed_seconds")
        for row in thermal_rows
    ]

    sm_clocks = [
        to_float(row, "sm_clock_mhz")
        for row in thermal_rows
    ]

    memory_clocks = [
        to_float(row, "memory_clock_mhz")
        for row in thermal_rows
    ]

    temperatures = [
        to_float(row, "temperature_c")
        for row in thermal_rows
    ]

    powers = [
        to_float(row, "power_draw_w")
        for row in thermal_rows
    ]

    power_limits = [
        to_float(row, "power_limit_w")
        for row in thermal_rows
    ]

    utilizations = [
        to_float(row, "gpu_utilization_percent")
        for row in thermal_rows
    ]

    # ========================================================
    # Throughput data
    # ========================================================

    throughput_times = [
        to_float(row, "elapsed_seconds")
        for row in throughput_rows
    ]

    throughput_values = [
        to_float(row, "tflops")
        for row in throughput_rows
    ]

    experiment_end = max(
        thermal_times
    )

    # ========================================================
    # First 30 seconds
    # ========================================================

    first_30_throughput = [
        tflops
        for elapsed, tflops
        in zip(
            throughput_times,
            throughput_values
        )
        if elapsed <= 30.0
    ]

    first_30_clocks = [
        clock
        for elapsed, clock
        in zip(
            thermal_times,
            sm_clocks
        )
        if elapsed <= 30.0
    ]

    first_30_temps = [
        temp
        for elapsed, temp
        in zip(
            thermal_times,
            temperatures
        )
        if elapsed <= 30.0
    ]

    first_30_powers = [
        power
        for elapsed, power
        in zip(
            thermal_times,
            powers
        )
        if elapsed <= 30.0
    ]

    # ========================================================
    # Last five minutes
    # ========================================================

    final_5min_start = (
        experiment_end - 300.0
    )

    final_5min_throughput = [
        tflops
        for elapsed, tflops
        in zip(
            throughput_times,
            throughput_values
        )
        if elapsed >= final_5min_start
    ]

    final_5min_clocks = [
        clock
        for elapsed, clock
        in zip(
            thermal_times,
            sm_clocks
        )
        if elapsed >= final_5min_start
    ]

    final_5min_temps = [
        temp
        for elapsed, temp
        in zip(
            thermal_times,
            temperatures
        )
        if elapsed >= final_5min_start
    ]

    final_5min_powers = [
        power
        for elapsed, power
        in zip(
            thermal_times,
            powers
        )
        if elapsed >= final_5min_start
    ]

    final_5min_utils = [
        util
        for elapsed, util
        in zip(
            thermal_times,
            utilizations
        )
        if elapsed >= final_5min_start
    ]

    # ========================================================
    # Required performance comparison
    # ========================================================

    first_30_peak_tflops = max(
        first_30_throughput
    )

    first_30_avg_tflops = mean(
        first_30_throughput
    )

    steady_avg_tflops = mean(
        final_5min_throughput
    )

    steady_peak_tflops = max(
        final_5min_throughput
    )

    throughput_change_percent = (
        (
            steady_avg_tflops
            - first_30_peak_tflops
        )
        / first_30_peak_tflops
        * 100.0
    )

    # ========================================================
    # Thermal / clock statistics
    # ========================================================

    initial_avg_clock = mean(
        first_30_clocks
    )

    steady_avg_clock = mean(
        final_5min_clocks
    )

    clock_change_percent = (
        (
            steady_avg_clock
            - initial_avg_clock
        )
        / initial_avg_clock
        * 100.0
    )

    max_temperature = max(
        temperatures
    )

    steady_avg_temperature = mean(
        final_5min_temps
    )

    avg_power = mean(
        powers
    )

    max_power = max(
        powers
    )

    steady_avg_power = mean(
        final_5min_powers
    )

    steady_avg_util = mean(
        final_5min_utils
    )

    nominal_power_limit = mean(
        power_limits
    )

    # ========================================================
    # Interpretation
    # ========================================================

    power_limit_ratio = (
        steady_avg_power
        / nominal_power_limit
    )

    if (
        power_limit_ratio >= 0.95
        and steady_avg_temperature < 85
    ):
        interpretation = (
            "The GPU operated very close to its configured "
            "power limit during steady state. The clock reduction "
            "is therefore more consistent with power-limited "
            "boost behavior than severe thermal throttling."
        )

    elif steady_avg_temperature >= 85:
        interpretation = (
            "The GPU reached a relatively high steady-state "
            "temperature, so thermal effects may have contributed "
            "to reduced boost clocks."
        )

    else:
        interpretation = (
            "No strong evidence of severe thermal throttling "
            "was observed from the recorded telemetry."
        )

    # ========================================================
    # Build text report
    # ========================================================

    lines = []

    lines.append(
        "=" * 78
    )

    lines.append(
        "HW2.5 PART E - SUSTAINED COMPUTE ANALYSIS"
    )

    lines.append(
        "=" * 78
    )

    lines.append(
        f"GPU label: {args.gpu_label}"
    )

    lines.append(
        f"Telemetry samples: {len(thermal_rows)}"
    )

    lines.append(
        f"Throughput samples: {len(throughput_rows)}"
    )

    lines.append(
        f"Last telemetry timestamp: {experiment_end:.2f} s"
    )

    lines.append("")

    lines.append(
        "FIRST 30 SECONDS"
    )

    lines.append(
        "-" * 78
    )

    lines.append(
        f"Peak throughput: "
        f"{first_30_peak_tflops:.2f} TFLOPS"
    )

    lines.append(
        f"Average throughput: "
        f"{first_30_avg_tflops:.2f} TFLOPS"
    )

    lines.append(
        f"Average SM clock: "
        f"{initial_avg_clock:.2f} MHz"
    )

    lines.append(
        f"Average temperature: "
        f"{mean(first_30_temps):.2f} C"
    )

    lines.append(
        f"Average power: "
        f"{mean(first_30_powers):.2f} W"
    )

    lines.append("")

    lines.append(
        "LAST FIVE MINUTES"
    )

    lines.append(
        "-" * 78
    )

    lines.append(
        f"Average throughput: "
        f"{steady_avg_tflops:.2f} TFLOPS"
    )

    lines.append(
        f"Peak throughput: "
        f"{steady_peak_tflops:.2f} TFLOPS"
    )

    lines.append(
        f"Average SM clock: "
        f"{steady_avg_clock:.2f} MHz"
    )

    lines.append(
        f"Average temperature: "
        f"{steady_avg_temperature:.2f} C"
    )

    lines.append(
        f"Average power: "
        f"{steady_avg_power:.2f} W"
    )

    lines.append(
        f"Average GPU utilization: "
        f"{steady_avg_util:.2f}%"
    )

    lines.append("")

    lines.append(
        "FULL-RUN SUMMARY"
    )

    lines.append(
        "-" * 78
    )

    lines.append(
        f"Maximum temperature: "
        f"{max_temperature:.2f} C"
    )

    lines.append(
        f"Average power: "
        f"{avg_power:.2f} W"
    )

    lines.append(
        f"Maximum power: "
        f"{max_power:.2f} W"
    )

    lines.append(
        f"Configured power limit: "
        f"{nominal_power_limit:.2f} W"
    )

    lines.append(
        f"Throughput change "
        f"(first-30s peak -> final-5min average): "
        f"{throughput_change_percent:.2f}%"
    )

    lines.append(
        f"Clock change "
        f"(first-30s average -> final-5min average): "
        f"{clock_change_percent:.2f}%"
    )

    lines.append("")

    lines.append(
        "INTERPRETATION"
    )

    lines.append(
        "-" * 78
    )

    lines.append(
        interpretation
    )

    lines.append("")

    lines.append(
        "=" * 78
    )

    report = "\n".join(
        lines
    )

    print(
        report
    )

    output_file.write_text(
        report,
        encoding="utf-8"
    )

    # ========================================================
    # Plot 1: SM clock over time
    # ========================================================

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        thermal_times,
        sm_clocks
    )

    plt.xlabel(
        "Elapsed Time (seconds)"
    )

    plt.ylabel(
        "SM Clock (MHz)"
    )

    plt.title(
        f"RTX {args.gpu_label} SM Clock "
        "During 20-Minute Sustained Compute"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        clock_plot,
        dpi=200
    )

    plt.close()

    # ========================================================
    # Plot 2: temperature over time
    # ========================================================

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        thermal_times,
        temperatures
    )

    plt.xlabel(
        "Elapsed Time (seconds)"
    )

    plt.ylabel(
        "GPU Temperature (C)"
    )

    plt.title(
        f"RTX {args.gpu_label} Temperature "
        "During 20-Minute Sustained Compute"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        temperature_plot,
        dpi=200
    )

    plt.close()

    print()
    print(
        "Saved analysis:",
        output_file
    )

    print(
        "Saved clock plot:",
        clock_plot
    )

    print(
        "Saved temperature plot:",
        temperature_plot
    )


if __name__ == "__main__":
    main()