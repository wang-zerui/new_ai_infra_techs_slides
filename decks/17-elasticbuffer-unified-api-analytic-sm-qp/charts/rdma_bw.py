"""DeepEP V2 RDMA bottleneck bandwidth per GPU (V2 release README performance table, PR #605).
Setup: 8K tokens/batch, hidden 7168, top-8, FP8 dispatch, BF16 combine, CX7 NICs. Logical bandwidth
(includes local-rank traffic). Numbers: notes/sources.md slide 12."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

groups = ["SM90\nEP 8×2", "SM90\nEP 8×4", "SM100\nEP 8×2"]
sms = [12, 6, 12]
dispatch = [90, 61, 90]
combine = [81, 61, 91]

sc.setup()
fig, ax = sc.figure("right")
w = 0.36
xs = range(len(groups))
b1 = ax.bar([x - w / 2 for x in xs], dispatch, w, color=sc.ACCENT, edgecolor=sc.INK, linewidth=1.2, label="Dispatch")
b2 = ax.bar([x + w / 2 for x in xs], combine, w, color=sc.ACCENT_LIGHT, edgecolor=sc.INK, linewidth=1.2, label="Combine")
for bars in (b1, b2):
    for r in bars:
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 1.5, f"{r.get_height():.0f}", ha="center", va="bottom",
                fontsize=18, fontweight="bold", color=sc.INK)
ax.set_xticks(list(xs))
ax.set_xticklabels([f"{g}\n{s} SMs" for g, s in zip(groups, sms)])
ax.set_ylabel("RDMA bandwidth (GB/s)")
ax.set_ylim(0, 115)
sc.top_legend(ax, ncol=2)
sc.better(ax, "Higher is better", where="upper right")
sc.save(fig, __file__)
