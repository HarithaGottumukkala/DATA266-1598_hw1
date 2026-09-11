import csv
import numpy as np
import matplotlib.pyplot as plt


INPUT_FILE = "raw/attention/attention_naive_coarse_4090.csv"
OUTPUT_FIGURE = "figures/naive_attention_memory_fit_4090.png"
OUTPUT_TEXT = "raw/attention/naive_attention_memory_fit_4090.txt"


lengths = []
memories = []

with open(INPUT_FILE, newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        if row["status"] == "SUCCESS":
            lengths.append(
                int(row["sequence_length"])
            )

            memories.append(
                float(row["peak_allocated_gib"])
            )


x = np.array(lengths, dtype=float)
y = np.array(memories, dtype=float)


# Fit:
# M(L) = a*L^2 + b*L + c
coefficients = np.polyfit(
    x,
    y,
    2
)

a, b, c = coefficients

predicted = np.polyval(
    coefficients,
    x
)


# R^2 goodness of fit
ss_res = np.sum(
    (y - predicted) ** 2
)

ss_tot = np.sum(
    (y - np.mean(y)) ** 2
)

r_squared = (
    1 - ss_res / ss_tot
    if ss_tot != 0
    else 1.0
)


equation = (
    f"M(L) = "
    f"{a:.12e} * L^2 + "
    f"{b:.12e} * L + "
    f"{c:.12e}"
)


print("=" * 78)
print("HW2.5 PART D4 — NAIVE ATTENTION MEMORY FIT")
print("=" * 78)

print("Input:", INPUT_FILE)
print()

print("Measured points:")

for length, memory in zip(
    lengths,
    memories
):
    print(
        f"L={length:5d}  "
        f"Peak memory={memory:.6f} GiB"
    )

print()

print("Quadratic fit:")
print(equation)

print()
print(
    "Quadratic coefficient a:",
    f"{a:.12e} GiB/token^2"
)

print(
    "Linear coefficient b:",
    f"{b:.12e} GiB/token"
)

print(
    "Constant coefficient c:",
    f"{c:.12e} GiB"
)

print(
    "R^2:",
    f"{r_squared:.12f}"
)


# Save text results
with open(
    OUTPUT_TEXT,
    "w"
) as f:

    f.write(
        "HW2.5 PART D4 — "
        "NAIVE ATTENTION MEMORY FIT\n"
    )

    f.write("=" * 70 + "\n\n")

    for length, memory in zip(
        lengths,
        memories
    ):
        f.write(
            f"L={length}, "
            f"peak_memory_gib={memory}\n"
        )

    f.write("\n")
    f.write(equation + "\n")

    f.write(
        f"Quadratic coefficient a = "
        f"{a:.12e} GiB/token^2\n"
    )

    f.write(
        f"Linear coefficient b = "
        f"{b:.12e} GiB/token\n"
    )

    f.write(
        f"Constant coefficient c = "
        f"{c:.12e} GiB\n"
    )

    f.write(
        f"R^2 = "
        f"{r_squared:.12f}\n"
    )


# Smooth curve for visualization
x_fit = np.linspace(
    min(x),
    max(x),
    300
)

y_fit = np.polyval(
    coefficients,
    x_fit
)


plt.figure(
    figsize=(9, 6)
)

plt.scatter(
    x,
    y,
    label="Measured peak memory"
)

plt.plot(
    x_fit,
    y_fit,
    label="Quadratic fit"
)

plt.xlabel(
    "Sequence Length"
)

plt.ylabel(
    "Peak Allocated GPU Memory (GiB)"
)

plt.title(
    "RTX 4090 Naive Attention: "
    "Peak Memory vs Sequence Length"
)

plt.grid(
    True,
    alpha=0.3
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_FIGURE,
    dpi=200
)

print()
print("Saved text:", OUTPUT_TEXT)
print("Saved figure:", OUTPUT_FIGURE)

print("=" * 78)