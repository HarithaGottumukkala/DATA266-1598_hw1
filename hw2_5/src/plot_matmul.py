import csv
from collections import defaultdict
import matplotlib.pyplot as plt

INPUT = "raw/matmul/matmul_4090.csv"
OUTPUT = "figures/matmul_throughput_4090.png"

data = defaultdict(list)

with open(INPUT, newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        if row["status"] == "SUCCESS":
            precision = row["precision"]
            n = int(row["N"])
            tflops = float(row["achieved_tflops"])
            data[precision].append((n, tflops))

plt.figure(figsize=(9, 6))

for precision in ["FP32", "TF32", "FP16", "BF16"]:
    points = sorted(data[precision])
    sizes = [p[0] for p in points]
    values = [p[1] for p in points]

    plt.plot(
        sizes,
        values,
        marker="o",
        linewidth=2,
        label=precision
    )

plt.xlabel("Matrix size N")
plt.ylabel("Achieved TFLOPS")
plt.title("RTX 4090 Dense Matrix Multiplication Throughput")
plt.xticks([1024, 4096, 8192, 16384])
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(OUTPUT, dpi=200)

print("Plot created successfully.")
print("Saved:", OUTPUT)

for precision in ["FP32", "TF32", "FP16", "BF16"]:
    points = sorted(data[precision])
    best = max(points, key=lambda x: x[1])
    print(
        f"{precision}: peak {best[1]:.3f} TFLOPS "
        f"at N={best[0]}"
    )
