"""KDA forward time, T=8192, H=96, D=128, fixed length: FLA Triton chunk_kda vs FLA Triton GDN (scalar gate) vs FlashKDA.
Sources: FlashKDA BENCHMARK_H20.md (2026-04-22) and BENCHMARK_GB200.md (2026-05-26), table 'T=8192, H=96, D=128', row 'Fixed'."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc
groups = ["H20 (Hopper)", "GB200 (Blackwell)"]
series = ["Triton KDA", "Triton GDN (scalar gate)", "FlashKDA"]
ms = [[4.8388, 2.3271], [3.1985, 1.2792], [2.6220, 1.0087]]
sc.setup()
fig, ax = sc.figure("right")
sc.grouped(ax, groups, series, ms, ours=2)
w = 0.8 / 3
for g in range(2):
    ax.text(g - 0.4 + w * 2.5, ms[2][g] + 0.08, f"{ms[2][g]:.2f}", ha="center", va="bottom", fontsize=18, fontweight="bold", color=sc.ACCENT)
    ax.text(g - 0.4 + w * 0.5, ms[0][g] + 0.08, f"{ms[0][g]:.2f}", ha="center", va="bottom", fontsize=17, color=sc.INK)
    ax.text(g - 0.4 + w * 1.5, ms[1][g] + 0.08, f"{ms[1][g]:.2f}", ha="center", va="bottom", fontsize=17, color=sc.INK)
ax.set_ylabel("Forward time (ms)")
ax.set_ylim(0, 6.0)
sc.better(ax, "Lower is better", "upper right")
leg = ax.get_legend()
for t in leg.get_texts(): t.set_fontsize(16)
leg.set_bbox_to_anchor((0.5, 1.0)); leg._ncols = 2
ax.legend(*ax.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=16, columnspacing=1.0, handlelength=1.4)
sc.save(fig, __file__)
