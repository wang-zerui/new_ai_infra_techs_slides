"""Analytic gradient-path DP traffic per weight per step (bytes), Moonlight's accounting (arXiv 2502.16982 §2.3:
FP32 = 4 B, BF16 = 2 B) applied to DeepSeek-V4 §3.4.1. Gradient path only (parameter all-gather excluded), one ZeRO group.
ZeRO-1 AdamW: FP32 reduce-scatter 4. Distributed Muon (Moonlight): 4 + BF16 gather 2.
V4 dense: owner receives whole matrices by reduce-scatter, no gather: 4, plus padding < 10% (<= 0.4).
V4 MoE: BF16 (stochastic rounding) all-to-all: 2."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

sc.setup()
fig, ax = sc.figure("right")
names = ["ZeRO-1\nAdamW", "Distributed\nMuon", "V4\ndense", "V4\nMoE"]
x = range(4)
w = 0.6
kw = dict(edgecolor=sc.INK, linewidth=1.2, width=w)
ax.bar(0, 4, color=sc.GRAYS[0], hatch="//", label="FP32 sync", **kw)
ax.bar(1, 4, color=sc.GRAYS[0], hatch="//", **kw)
ax.bar(1, 2, bottom=4, color=sc.PROBLEM, label="BF16 gather", **kw)
ax.bar(2, 4, color=sc.GRAYS[0], hatch="//", **kw)
ax.bar(2, 0.4, bottom=4, color="white", hatch="xx", label="padding", **kw)
ax.bar(3, 2, color=sc.ACCENT, label="BF16 sync", **kw)
for xi, v, t, c in [(0, 4, "4 B", sc.INK), (1, 6, "6 B", sc.PROBLEM), (2, 4.4, "≤ 4.4 B", sc.ACCENT), (3, 2, "2 B", sc.ACCENT)]:
    ax.text(xi, v + 0.15, t, ha="center", va="bottom", fontsize=20, fontweight="bold", color=c)
ax.set_xticks(list(x))
ax.set_xticklabels(names)
ax.set_ylabel("Gradient bytes per weight")
ax.set_ylim(0, 7.4)
ax.set_yticks([0, 2, 4, 6])
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=18, frameon=False, handlelength=1.2,
          columnspacing=0.8, handletextpad=0.4, borderaxespad=0.2)
ax.text(0.02, 0.97, "Lower is better", transform=ax.transAxes, ha="left", va="top", fontsize=16,
        fontstyle="italic", color=sc.GOOD)
sc.save(fig, __file__)
