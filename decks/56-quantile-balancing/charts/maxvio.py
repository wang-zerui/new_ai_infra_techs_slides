"""AvgMaxVio (mean over training batches of max_j load_j / mean load - 1) at 16 and 64 experts.

Source: [BIP] arXiv 2502.15451v2, Table 2 (16 experts, k = 4, 0.3B) and Table 3 (64 experts, k = 8, 1.1B);
Minimind MoE with softmax router; loss-free u = 0.001; BIP shown at T = 14.
Derived: 0.3366 / 0.1275 = 2.64x.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

groups = ["16 experts (top-4)", "64 experts (top-8)"]
series = ["sign-update bias (loss-free)", "assignment solver (BIP)"]
vals = [[0.1275, 0.3366], [0.0547, 0.0529]]
colors = [sc.PROBLEM, sc.ACCENT_LIGHT]

sc.setup()
fig, ax = sc.figure("right")
width = 0.64
w = width / len(series)
tops = []
for si, name in enumerate(series):
    xs = [g - width / 2 + w * (si + 0.5) for g in range(len(groups))]
    ax.bar(xs, vals[si], w, color=colors[si], edgecolor=sc.INK, linewidth=1.2, label=name)
    for x, v in zip(xs, vals[si]):
        ax.text(x, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=19,
                fontweight="bold" if si == 0 else "normal", color=sc.PROBLEM if si == 0 else sc.INK)
        if si == 0:
            tops.append((x, v))
ax.set_xticks(range(len(groups)))
ax.set_xticklabels(groups)
ax.set_ylim(0, 0.5)
ax.set_ylabel("Avg. MaxVio per batch")
sc.top_legend(ax, ncol=2, fontsize=17)
sc.better(ax, "Lower is better", where="upper left")
(x0, v0), (x1, v1) = tops
ax.annotate("", xy=(x1 - 0.17, v1 + 0.03), xytext=(x0 + 0.08, v0 + 0.065),
            arrowprops=dict(arrowstyle="-|>", color=sc.PROBLEM, lw=2.4, linestyle="--"))
ax.text((x0 + x1) / 2 - 0.05, (v0 + v1) / 2 + 0.1, "2.6×", color=sc.PROBLEM, fontsize=24, fontweight="bold",
        ha="center")
sc.save(fig, __file__)
