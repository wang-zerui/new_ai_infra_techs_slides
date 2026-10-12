"""End-to-end training throughput on 256 GB200 GPUs, 4k sequence length (Megatron-Core MoE report,
arXiv 2603.07685, Table 11). Speedup labels are derived ratios. See notes/sources.md."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc  # noqa: E402

groups = ["DeepSeek-V3", "Qwen3-235B"]
series = ["BF16", "MXFP8"]
vals = [[3298, 5100], [4020, 6212]]   # tokens/s/GPU

sc.setup()
fig, ax = sc.plt.subplots(figsize=(6.5, 3.75))
sc.grouped(ax, groups, series, vals, ours=1, width=0.7)
w = 0.35
for g in range(2):
    ax.text(g + w / 2, vals[1][g] + 120, f"{vals[1][g]:,}", ha="center", va="bottom", fontsize=17,
            color=sc.ACCENT, fontweight="bold")
    ax.text(g - w / 2, vals[0][g] + 120, f"{vals[0][g]:,}", ha="center", va="bottom", fontsize=17, color=sc.INK)
    ax.text(g, vals[1][g] + 1150, f"{vals[1][g] / vals[0][g]:.2f}×", ha="center", va="bottom", fontsize=21,
            color=sc.ACCENT, fontweight="bold")
ax.set_ylim(0, 8600)
ax.set_ylabel("Tokens/s per GPU")
sc.better(ax, "Higher is better", where="upper left")
sc.save(fig, __file__)
