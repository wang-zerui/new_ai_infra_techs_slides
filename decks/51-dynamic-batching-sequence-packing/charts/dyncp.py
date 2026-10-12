"""Llama 13B, GBS 2048, PP=8, CP=8, full recompute: packing only vs Dynamic-CP (TFLOPS/GPU).
Source: NVIDIA Technical Blog, "Speeding Up Variable-Length Training with Dynamic Context Parallelism and
NVIDIA Megatron Core" (Jan 28, 2026), Table 1. Speedups 1.48x / 1.25x as stated in the blog."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

groups = ["GitHub", "CommonCrawl"]
series = ["packing only", "packing + Dynamic-CP"]
vals = [[195.88, 139.17], [289.32, 174.39]]
sc.setup()
fig, ax = sc.figure("right")
sc.grouped(ax, groups, series, vals, ours=1, width=0.7)
w = 0.35
for g in range(2):
    for s_ in range(2):
        x = g - 0.35 + w * (s_ + 0.5)
        ax.text(x, vals[s_][g] + 5, f"{vals[s_][g]:.0f}", ha="center", va="bottom", fontsize=18,
                fontweight="bold", color=sc.ACCENT if s_ == 1 else sc.INK)
for g, r in enumerate(["1.48×", "1.25×"]):
    ax.text(g + 0.175, vals[1][g] + 32, r, ha="center", va="bottom", fontsize=22, fontweight="bold", color=sc.ACCENT)
ax.set_ylim(0, 380)
ax.set_ylabel("TFLOPS / GPU")
sc.better(ax, "Higher is better", where="upper right")
sc.save(fig, __file__)
