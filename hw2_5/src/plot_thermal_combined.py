import csv
from pathlib import Path

import matplotlib.pyplot as plt


GPU_LABEL = "4090"

INPUT_FILE = Path(
    f"hw2_5/raw/thermal/thermal_{GPU_LABEL}.csv"
)

OUTPUT_FILE = Path(
    f"hw2_5/figures/thermal_clock_temperature_{GPU_LABEL}.png"
)


times = []
clocks = []
temperatures = []


with open(INPUT_FILE, newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        times.append(
            float(row["elapsed_seconds"])
        )

        clocks.append(
            float(row["sm_clock_mhz"])
        )

        temperatures.append(
            float(row["temperature_c"])
        )


fig, axes = plt.subplots(
    2,
    1,
    figsize=(10, 8),
    sharex=True
)


# ------------------------------------------------------------
# GPU clock
# ------------------------------------------------------------

axes[0].plot(
    times,
    clocks
)

axes[0].set_ylabel(
    "SM Clock (MHz)"
)

axes[0].set_title(
    "RTX 4090 Sustained Load: Clock and Temperature"
)

axes[0].grid(
    True,
    alpha=0.3
)


# ------------------------------------------------------------
# GPU temperature
# ------------------------------------------------------------

axes[1].plot(
    times,
    temperatures
)

axes[1].set_xlabel(
    "Elapsed Time (seconds)"
)

axes[1].set_ylabel(
    "Temperature (C)"
)

axes[1].grid(
    True,
    alpha=0.3
)


plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=200
)

plt.close()


print(
    "Combined thermal figure saved:"
)

print(
    OUTPUT_FILE
)
