"""Dynamic CP vs packing-only, Llama 13B, GBS 2048, PP=8, CP=8, full recompute (TFLOPS/GPU).

Source: NVIDIA Technical Blog, "Speeding Up Variable-Length Training with Dynamic Context Parallelism and
NVIDIA Megatron Core" (Jan 2026), Table 1. Ratios 289.32/195.88 = 1.48x, 174.39/139.17 = 1.25x.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

groups = ["GitHub", "CommonCrawl"]
series = ["Only packing", "Dynamic CP"]
vals = [[195.88, 139.17], [289.32, 174.39]]

sc.setup()
fig, ax = sc.figure("right")
sc.grouped(ax, groups, series, vals, ours=1, width=0.7)
w = 0.35
for g, (b, d) in enumerate(zip(vals[0], vals[1])):
    ax.text(g - w / 2, b + 6, f"{b:.0f}", ha="center", va="bottom", fontsize=18, color=sc.INK)
    ax.text(g + w / 2, d + 6, f"{d:.0f}", ha="center", va="bottom", fontsize=18, color=sc.ACCENT, fontweight="bold")
    ax.text(g + w / 2, d + 34, f"{d / b:.2f}×", ha="center", va="bottom", fontsize=22, color=sc.ACCENT,
            fontweight="bold")
ax.set_ylim(0, 370)
ax.set_ylabel("TFLOPS per GPU")
sc.better(ax, "Higher is better", where="upper right")
sc.save(fig, __file__)
