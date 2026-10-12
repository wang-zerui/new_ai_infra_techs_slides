"""DeepEP dispatch/combine algorithm bandwidth on DeepSeek's multi-plane fat-tree (MPFT), 4096 tokens per GPU.

Source: Insights into DeepSeek-V3 (ISCA'25), arXiv 2505.09343, Fig. 7 data labels (p.10).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

gpus = ["16", "32", "64", "128"]
dispatch = [42.47, 58.02, 50.58, 45.34]
combine = [43.05, 56.96, 48.54, 41.6]

sc.setup()
fig, ax = sc.figure("right")
w = 0.38
xs = range(len(gpus))
ax.bar([x - w / 2 for x in xs], dispatch, w, color=sc.ACCENT, edgecolor=sc.INK, linewidth=1.2, label="dispatch")
ax.bar([x + w / 2 for x in xs], combine, w, color=sc.ACCENT_LIGHT, edgecolor=sc.INK, linewidth=1.2, label="combine")
for x, d, c in zip(xs, dispatch, combine):
    ax.text(x - w / 2, d + 1, f"{d:.0f}", ha="center", va="bottom", fontsize=17, fontweight="bold", color=sc.ACCENT)
    ax.text(x + w / 2, c + 1, f"{c:.0f}", ha="center", va="bottom", fontsize=17, color=sc.INK)
ax.axhline(40, color=sc.GOOD, lw=2, ls="--", zorder=0, label="40 GB/s")
ax.set_xticks(list(xs))
ax.set_xticklabels(gpus)
ax.set_xlabel("GPUs (8 per node)")
ax.set_ylim(0, 72)
ax.set_ylabel("Algorithm bandwidth\nper GPU (GB/s)")
sc.top_legend(ax, ncol=3)
sc.better(ax, "Higher is better", where="upper right")
sc.save(fig, __file__)
