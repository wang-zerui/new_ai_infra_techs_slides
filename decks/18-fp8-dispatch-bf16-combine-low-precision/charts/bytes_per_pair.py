"""Bytes per token-expert pair per pass (dispatch + combine), in units of the hidden size h. Analytic.

BF16 both ways: 2h + 2h = 4h. FP8 dispatch + BF16 combine: h (E4M3) + h/32 (FP32 scale per 128 channels)
+ 2h = 3.03h (V4 rounds this to 3h). Top-k metadata excluded on both sides.
Sources: notes/sources.md "Derived numbers" ([V4] §3.1, [EP-CPP], [MATH], [HE-TEST]).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

names = ["BF16 both ways", "FP8 dispatch\n+ BF16 combine"]
dispatch = [2.0, 1.0]
scales = [0.0, 1 / 32]
combine = [2.0, 2.0]

sc.setup()
fig, ax = sc.figure("right")
xs = [0, 1]
w = 0.56
ax.bar(xs[0], dispatch[0], w, color=sc.GRAYS[1], edgecolor=sc.INK, linewidth=1.2, hatch="//")
ax.bar(xs[1], dispatch[1], w, color=sc.ACCENT, edgecolor=sc.INK, linewidth=1.2)
ax.bar(xs[1], scales[1], w, bottom=dispatch[1], color="#0B2A45", edgecolor=sc.INK, linewidth=0.8)
for x, d, s_, c in zip(xs, dispatch, scales, combine):
    ax.bar(x, c, w, bottom=d + s_, color="#F2F2F2", edgecolor=sc.INK, linewidth=1.2)
# segment labels
ax.text(0, 1.0, "dispatch\n2h", ha="center", va="center", fontsize=18, color=sc.INK,
        bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
ax.text(1, 0.5, "dispatch\nh + h/32", ha="center", va="center", fontsize=18, color="white", fontweight="bold")
ax.text(0, 3.0, "combine\n2h", ha="center", va="center", fontsize=18, color=sc.INK)
ax.text(1, 2.03, "combine\n2h", ha="center", va="center", fontsize=18, color=sc.INK)
ax.text(0, 4.1, "4h", ha="center", va="bottom", fontsize=22, fontweight="bold", color=sc.INK)
ax.text(1, 3.13, "3.03h", ha="center", va="bottom", fontsize=22, fontweight="bold", color=sc.ACCENT)
sc.callout(ax, "−24%", xy=(1.32, 3.05), xytext=(1.62, 3.9), color=sc.ACCENT, size=24)
ax.set_xticks(xs)
ax.set_xticklabels(names)
ax.set_xlim(-0.55, 1.95)
ax.set_ylim(0, 5.3)
ax.set_ylabel("Bytes per token-expert\nper pass (× h)")
sc.better(ax, "Lower is better", where="upper left")
sc.save(fig, __file__)
