"""Analytic HBM traffic per token at one layer boundary, in elements of C (= d), n = 4 streams.

- Plain residual: read 2C, write C = 3C (mHC paper, arXiv 2512.24880, Table 2).
- HC op by op: reads (5n+1)C + n^2 + 2n, writes (3n+1)C + n^2 + 2n -> 34C (+48) at n = 4 (Table 2).
- Mega mHC, logical: reads x (C) + residual (4C), writes new residual (4C) + y (C) = 10C; with the
  scratch traffic the DeepGEMM test counts in normal mode (norm boundary 2C + new-residual re-read 4C) = 16C.
  Mega mHC additionally emits the RMS-normalised next-layer input, which Table 2 does not include.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

sc.setup()
fig, ax = sc.figure("right")
names = ["plain\nresidual", "HC,\nop by op", "Mega mHC"]
xs = [0, 1, 2]
ax.bar(0, 3, 0.6, color=sc.GRAYS[0], edgecolor=sc.INK, linewidth=1.2, hatch=sc.HATCHES[0])
ax.bar(1, 34, 0.6, color=sc.PROBLEM, edgecolor=sc.INK, linewidth=1.2)
ax.bar(2, 10, 0.6, color=sc.ACCENT, edgecolor=sc.INK, linewidth=1.2, label="logical I/O")
ax.bar(2, 6, 0.6, bottom=10, color=sc.ACCENT_LIGHT, edgecolor=sc.INK, linewidth=1.2, hatch="//",
       label="+ scratch I/O")
for x, v, t, c in [(0, 3, "3d", sc.INK), (1, 34, "34d", sc.PROBLEM), (2, 16, "10–16d", sc.ACCENT)]:
    ax.text(x, v + 0.8, t, ha="center", va="bottom", fontsize=20, fontweight="bold", color=c)
ax.set_xticks(xs)
ax.set_xticklabels(names)
ax.set_ylim(0, 40)
ax.set_ylabel("HBM elements per token (× d)")
sc.top_legend(ax, ncol=2)
sc.better(ax, "Lower is better", where="upper right")
sc.save(fig, __file__)
