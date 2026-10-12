"""End-to-end Megatron Core throughput on Grace Blackwell, baseline dispatcher vs HybridEP.
Source: NVIDIA Technical Blog, "Optimizing Communication for MoE Training with Hybrid Expert Parallel"
(Feb 2026), Table 1. Speedup labels are the table's own column."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

groups = ["DSV3\nvs\nDeepEP", "DSV3\nFSDP vs\nall-to-all", "Qwen3\nBF16 vs\nall-to-all", "Qwen3\nMXFP8 vs\nall-to-all"]
base_name = ["vs DeepEP", "vs all-to-all", "vs all-to-all", "vs all-to-all"]
base = [829, 597, 665, 728]
hep = [943, 645, 698, 800]
spd = ["1.14×", "1.08×", "1.05×", "1.10×"]

sc.setup()
fig, ax = sc.figure("right")
w = 0.38
xs = range(len(groups))
ax.bar([x - w / 2 for x in xs], base, w, color=sc.GRAYS[0], edgecolor=sc.INK, linewidth=1.2, hatch=sc.HATCHES[0],
       label="baseline")
ax.bar([x + w / 2 for x in xs], hep, w, color=sc.ACCENT, edgecolor=sc.INK, linewidth=1.2, label="HybridEP")
for x, h, t in zip(xs, hep, spd):
    ax.text(x + w / 2, h + 15, t, ha="center", va="bottom", fontsize=19, fontweight="bold", color=sc.ACCENT)
ax.set_xticks(list(xs))
ax.set_xticklabels(groups)
ax.set_ylim(0, 1080)
ax.set_ylabel("TFLOPS per GPU")
sc.top_legend(ax, ncol=2)
sc.better(ax, "Higher is better", where="upper right")
sc.save(fig, __file__)
