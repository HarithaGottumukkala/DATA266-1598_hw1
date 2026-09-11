import csv
import matplotlib.pyplot as plt


NAIVE_FILE = "raw/attention/attention_naive_coarse_4090.csv"
EFFICIENT_FILE = "raw/attention/attention_fused_efficient_coarse_4090.csv"

OUTPUT_CSV = "raw/attention/attention_comparison_4090.csv"
OUTPUT_FIGURE = "figures/attention_speedup_4090.png"


def load_results(path):
    results = {}

    with open(path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row["status"] == "SUCCESS":
                length = int(row["sequence_length"])

                results[length] = {
                    "latency_ms":
                        float(row["median_latency_ms"]),

                    "peak_memory_gib":
                        float(row["peak_allocated_gib"])
                }

    return results


naive = load_results(NAIVE_FILE)
efficient = load_results(EFFICIENT_FILE)


rows = []

print("=" * 80)
print("HW2.5 PART D5 — NAIVE VS MEMORY-EFFICIENT ATTENTION")
print("=" * 80)

print()
print(
    f"{'Length':>8} "
    f"{'Naive ms':>12} "
    f"{'Efficient ms':>14} "
    f"{'Speedup':>10} "
    f"{'Naive GiB':>12} "
    f"{'Efficient GiB':>14}"
)

print("-" * 80)


for length in sorted(
    set(naive.keys()) & set(efficient.keys())
):

    naive_latency = naive[length]["latency_ms"]
    efficient_latency = efficient[length]["latency_ms"]

    naive_memory = naive[length]["peak_memory_gib"]
    efficient_memory = efficient[length]["peak_memory_gib"]

    speedup = (
        naive_latency
        / efficient_latency
    )

    memory_reduction = (
        naive_memory
        / efficient_memory
    )

    print(
        f"{length:8d} "
        f"{naive_latency:12.4f} "
        f"{efficient_latency:14.4f} "
        f"{speedup:9.2f}x "
        f"{naive_memory:12.4f} "
        f"{efficient_memory:14.4f}"
    )

    rows.append({
        "sequence_length":
            length,

        "naive_latency_ms":
            naive_latency,

        "efficient_latency_ms":
            efficient_latency,

        "speedup_x":
            speedup,

        "naive_peak_memory_gib":
            naive_memory,

        "efficient_peak_memory_gib":
            efficient_memory,

        "memory_reduction_x":
            memory_reduction
    })


with open(
    OUTPUT_CSV,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=rows[0].keys()
    )

    writer.writeheader()
    writer.writerows(rows)


lengths = [
    row["sequence_length"]
    for row in rows
]

speedups = [
    row["speedup_x"]
    for row in rows
]


plt.figure(figsize=(9, 6))

plt.plot(
    lengths,
    speedups,
    marker="o"
)

plt.axhline(
    y=1.0,
    linestyle="--"
)

plt.xlabel(
    "Sequence Length"
)

plt.ylabel(
    "Speedup (Naive Latency / Efficient Latency)"
)

plt.title(
    "RTX 4090 Attention Speedup: "
    "Memory-Efficient vs Naive"
)

plt.xticks(lengths)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    OUTPUT_FIGURE,
    dpi=200
)


print()
print("=" * 80)
print("COMPARISON COMPLETE")
print("=" * 80)

print("Saved CSV:", OUTPUT_CSV)
print("Saved figure:", OUTPUT_FIGURE)