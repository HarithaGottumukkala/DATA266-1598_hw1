"""Regenerates every data figure in the HW12 report from VERIFIED recorded values.
No training or inference is performed. Sources are given beside each value.
Usage (from the hw12 folder, after src/derive_metrics.py):  python3 src/make_figures.py
Writes PNG figures to figures/ (for viewing on GitHub) and vector PDFs to report_source/figures/ (used by main.tex).
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "figures"; OUT.mkdir(parents=True, exist_ok=True)
OUT_PDF = ROOT / "report_source/figures"; OUT_PDF.mkdir(parents=True, exist_ok=True)
S = ROOT / "results/knowledge_distillation"
derived = json.load(open(S / "derived_metrics.json"))

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#555555",
    "axes.linewidth": 0.6, "xtick.color": "#333333", "ytick.color": "#333333",
    "grid.color": "#e3e3e3", "grid.linewidth": 0.5, "axes.grid": True, "axes.axisbelow": True,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]  # validated order
MK = ["o", "s", "^", "D", "v", "P"]
INK, MUTED = "#222222", "#666666"
W1 = 3.45  # IEEE single-column width (in)

def save(fig, name):
    fig.savefig(OUT_PDF / f"{name}.pdf"); fig.savefig(OUT / f"{name}.png"); plt.close(fig)

def hist(p):
    return json.load(open(p))
# ---------------- training histories (seed 1598) ----------------
h = {
    "T=1, α=0.50": hist(S / "temperature_T1_alpha05/training_history.json"),
    "T=2, α=0.50": hist(S / "kd_baseline_T2_alpha05/training_history.json"),
    "T=10, α=0.50": json.load(open(S / "temperature_T10_alpha05/experiment_summary.json"))["history"],
    "T=5, α=0.25": json.load(open(S / "alpha_T5_alpha025/experiment_summary.json"))["history"],
}
# T=5/α=0.50 and T=5/α=0.75: no history file was exported; epoch values come from the
# executed notebook printout (epochs 1-2 printed to two decimals; epoch 3 exact from sweep tables).
val = {k: [r["validation_accuracy"] * 100 for r in v] for k, v in h.items()}
val["T=5, α=0.50"] = [93.93, 94.50, 94.7750]
val["T=5, α=0.75"] = [93.77, 94.50, 94.6583]
order = ["T=1, α=0.50", "T=2, α=0.50", "T=5, α=0.50", "T=10, α=0.50", "T=5, α=0.25", "T=5, α=0.75"]

fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.25))
ep = [1, 2, 3]
for i, k in enumerate(order):
    ls = "--" if k in ("T=5, α=0.50", "T=5, α=0.75") else "-"
    ax[0].plot(ep, val[k], ls, color=C[i], marker=MK[i], ms=4, lw=1.4, label=k)
ax[0].set_xticks(ep); ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Validation accuracy (%)")
ax[0].set_title("(a) Validation accuracy, 12,000 held-out training rows", loc="left", color=INK)
ax[0].legend(ncol=2, frameon=False, loc="lower right")
for i, k in enumerate(["T=1, α=0.50", "T=2, α=0.50", "T=10, α=0.50", "T=5, α=0.25"]):
    ci = order.index(k)
    ax[1].plot(ep, [r["train_ce_loss"] for r in h[k]], "-", color=C[ci], marker=MK[ci], ms=4, lw=1.4,
               label=f"{k} train")
    ax[1].plot(ep, [r["validation_ce_loss"] for r in h[k]], ":", color=C[ci], marker=MK[ci], ms=4,
               lw=1.4, mfc="white", label=f"{k} val.")
ax[1].set_xticks(ep); ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Hard-label cross-entropy")
ax[1].set_title("(b) Training (solid) vs. validation (dotted) CE", loc="left", color=INK)
ax[1].legend(ncol=2, frameon=False, fontsize=5.8, loc="upper right")
save(fig, "fig_training_curves")

# ---------------- sweeps ----------------
ci = derived["val_acc_95ci_halfwidth_pp"]
Ts = [1, 2, 5, 10]; tacc = [94.7167, 94.7000, 94.7750, 94.7083]; tf1 = [0.947245, 0.947056, 0.947808, 0.947146]
As = [0.25, 0.50, 0.75]; aacc = [94.7917, 94.7750, 94.6583]; af1 = [0.947965, 0.947808, 0.946652]
fig, ax = plt.subplots(1, 2, figsize=(W1, 1.85), sharey=True)
for a, xs, ys, f1s, sel, ttl, xl in [
        (ax[0], [str(t) for t in Ts], tacc, tf1, 2, "(a) Temperature, α = 0.50", "Temperature T"),
        (ax[1], [f"{x:.2f}" for x in As], aacc, af1, 0, "(b) CE weight α, T = 5", "α (hard-label weight)")]:
    x = np.arange(len(xs))
    a.errorbar(x, ys, yerr=ci, fmt="none", ecolor="#b5b5b5", elinewidth=0.8, capsize=2)
    a.plot(x, ys, "-", color=C[0], lw=1.2, marker="o", ms=4, label="Accuracy (±95% CI)")
    a.plot(x[sel], ys[sel], marker="*", ms=10, color=C[1], ls="none", label="Selected")
    for xi, yi in zip(x, ys):
        a.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points", xytext=(4, -9), ha="left",
                   fontsize=6, color=MUTED)
    a.set_xticks(x, xs); a.set_xlabel(xl); a.set_title(ttl, loc="left", color=INK)
    a.set_xlim(-0.4, len(xs) - 0.6)
ax[0].set_ylabel("Best validation acc. (%)"); ax[0].set_ylim(94.15, 95.35)
ax[1].legend(frameon=False, loc="lower left", fontsize=6)
save(fig, "fig_sweeps")

# ---------------- teacher soft targets vs temperature (from saved logits) ----------------
st = derived["teacher_soft_targets"]
fig, ax = plt.subplots(1, 2, figsize=(W1, 1.6))
x = np.arange(4)
ent = [st[str(t)]["mean_entropy_over_log4"] for t in Ts]
non = [st[str(t)]["mean_nontarget_mass"] for t in Ts]
for a, y, ttl, yl in [(ax[0], ent, "(a) Mean entropy / log 4", "Normalised entropy"),
                      (ax[1], non, "(b) Mean non-argmax mass", "")]:
    a.bar(x, y, width=0.6, color=C[0], edgecolor="white", linewidth=1)
    for xi, yi in zip(x, y):
        a.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points", xytext=(0, 2), ha="center",
                   fontsize=6, color=INK)
    a.set_xticks(x, [f"T={t}" for t in Ts]); a.set_ylim(0, 1.08); a.set_title(ttl, loc="left", color=INK)
    a.set_ylabel(yl)
fig.subplots_adjust(wspace=0.3)
save(fig, "fig_soft_targets")

# ---------------- per-class + teacher confusion matrix ----------------
names = ["World", "Sports", "Business", "Sci/Tech"]
tea = [94.9474, 98.8421, 91.3684, 92.6842]; stu = [95.4211, 98.8947, 91.2105, 93.1053]
cm = np.array([[1804, 10, 50, 36], [9, 1878, 6, 7], [33, 6, 1736, 125], [28, 9, 102, 1761]])
fig = plt.figure(figsize=(7.1, 2.2))
a = fig.add_axes([0.06, 0.18, 0.40, 0.70]); b = fig.add_axes([0.58, 0.14, 0.27, 0.76])
y = np.arange(4)[::-1]
for yi, t, s in zip(y, tea, stu):
    a.plot([t, s], [yi, yi], color="#c9c9c9", lw=2, zorder=1)
a.scatter(tea, y, s=30, color=C[0], marker="o", zorder=2, label="BERT teacher")
a.scatter(stu, y, s=30, color=C[1], marker="s", zorder=3, label="DistilBERT student (T=5, α=0.25)")
for yi, t, s in zip(y, tea, stu):
    d = s - t
    a.annotate(f"{d:+.2f} pp", (max(t, s), yi), textcoords="offset points", xytext=(6, -2.5),
               fontsize=6.5, color=MUTED)
a.set_yticks(y, names); a.set_xlim(90.5, 100.2); a.set_xlabel("Per-class test accuracy = recall (%)")
a.set_title("(a) Per-class accuracy on 1,900 test articles per class", loc="left", color=INK)
a.legend(frameon=False, loc="upper left", fontsize=6.5); a.grid(axis="y", visible=False)
im = b.imshow(cm / cm.sum(1, keepdims=True) * 100, cmap="Blues", vmin=0, vmax=100)
for i in range(4):
    for j in range(4):
        v = cm[i, j]; frac = v / cm[i].sum()
        b.text(j, i, f"{v}", ha="center", va="center", fontsize=6.5, color="white" if frac > 0.5 else INK)
b.set_xticks(range(4), names, rotation=20); b.set_yticks(range(4), names)
b.set_xlabel("Predicted"); b.set_ylabel("True"); b.grid(False)
b.set_title("(b) Teacher confusion matrix (counts)", loc="left", color=INK)
cb = fig.colorbar(im, ax=b, fraction=0.046, pad=0.03); cb.set_label("Row %", fontsize=6.5); cb.ax.tick_params(labelsize=6)
save(fig, "fig_perclass")

# ---------------- efficiency: relative change, KD and GPTQ (separate panels) ----------------
bench = json.load(open(S / "final_test_evaluation/benchmark_config.json"))["results"]
tb, sb = bench[0], bench[1]
kd = [("Parameters", 66956548 / 109485316 - 1),
      ("p50 latency", sb["p50_latency_ms"] / tb["p50_latency_ms"] - 1),
      ("p95 latency", sb["p95_latency_ms"] / tb["p95_latency_ms"] - 1),
      ("Peak alloc. GPU mem.", sb["peak_gpu_memory_gb"] / tb["peak_gpu_memory_gb"] - 1),
      ("Checkpoint dir. size", sb["checkpoint_size_mb"] / tb["checkpoint_size_mb"] - 1),
      ("Test error rate", (1 - .946579) / (1 - .944605) - 1)]
gq = [("WikiText-2 perplexity", 29.6005 / 26.1023 - 1),
      ("p50 latency", 16.356358 / 7.914175 - 1),
      ("p95 latency", 18.907077 / 10.295235 - 1),
      ("Peak alloc. GPU mem.", 0.148056 / 0.265369 - 1),
      ("Saved weight size", 119.317703 / 238.896339 - 1)]
fig, ax = plt.subplots(1, 2, figsize=(7.1, 1.95))
for a, rows, ttl in [(ax[0], kd, "(a) Distillation: DistilBERT student vs. BERT teacher (Tesla T4)"),
                     (ax[1], gq, "(b) GPTQ 4-bit vs. FP16 OPT-125M (NVIDIA L4)")]:
    lab = [r[0] for r in rows][::-1]; v = [r[1] * 100 for r in rows][::-1]
    cols = [C[2] if x < 0 else C[1] for x in v]
    a.barh(range(len(v)), v, color=cols, height=0.6, edgecolor="white", linewidth=1)
    for i, x in enumerate(v):
        a.annotate(f"{x:+.1f}%", (x, i), textcoords="offset points", xytext=(3 if x >= 0 else -3, -2.5),
                   ha="left" if x >= 0 else "right", fontsize=6.5, color=INK)
    a.axvline(0, color="#555555", lw=0.7); a.set_yticks(range(len(v)), lab)
    a.set_xlabel("Relative change vs. reference (%)  —  negative = lower"); a.grid(axis="y", visible=False)
    a.set_title(ttl, loc="left", color=INK)
ax[0].set_xlim(-64, 15); ax[1].set_xlim(-82, 128)
fig.subplots_adjust(wspace=0.55)
save(fig, "fig_efficiency")
print("PNG figures ->", OUT, "| PDF figures ->", OUT_PDF)
