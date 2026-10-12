"""Scaling efficiency of KDA context parallelism on 4 GPUs: head-split all-to-all CP vs state-passing CP (KCP).
Source: fla-org/flash-linear-attention PR #691 description, 'KDA' benchmark: 32K tokens, CP=4, 64 heads, head_dim 128,
H800 (triton 3.4, torch 2.8, CUDA 12.8). 'rate' = non-CP time / (CP time x 4):
forward: KCP 7.839/(2.591*4) = 75.65%, all-to-all 7.839/(5.276*4) = 37.15%;
forward+backward: KCP 36.193/(10.490*4) = 86.26%, all-to-all 36.193/(16.415*4) = 55.12%.
Note: FLA's implementation is independent of Moonshot's internal KCP (FLA fla/ops/cp README)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc
groups = ["Forward", "Forward + backward"]
series = ["All-to-all CP (head split)", "State-passing CP (KCP)"]
vals = [[37.15, 55.12], [75.65, 86.26]]
sc.setup()
fig, ax = sc.figure("right")
sc.grouped(ax, groups, series, vals, ours=1)
w = 0.4
for g in range(2):
    ax.text(g - 0.2, vals[0][g] + 1.5, f"{vals[0][g]:.0f}%", ha="center", va="bottom", fontsize=18, color=sc.INK)
    ax.text(g + 0.2, vals[1][g] + 1.5, f"{vals[1][g]:.0f}%", ha="center", va="bottom", fontsize=20, fontweight="bold", color=sc.ACCENT)
ax.axhline(100, color=sc.GRAYS[3], lw=2, ls="--", zorder=0)
ax.text(1.45, 101.5, "ideal", ha="right", va="bottom", fontsize=16, color=sc.GRAYS[3])
ax.set_ylabel("Scaling efficiency, 4 GPUs (%)")
ax.set_ylim(0, 112)
leg = ax.get_legend()
ax.legend(*ax.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, frameon=False, fontsize=17, handlelength=1.4)
sc.better(ax, "Higher is better", "upper left")
sc.save(fig, __file__)
