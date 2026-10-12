"""Decode speed with MTP speculative decoding, DeepSeek-V3 (arXiv 2412.19437 §5.4.3): acceptance of the MTP-drafted
token 85-90%, measured 1.8x TPS. Ceiling (analytic): Leviathan et al. arXiv 2211.17192 Eq. 1 / Thm 3.8 with one draft
(gamma = 1) and free drafting (c = 0): speedup = 1 + alpha = 1.85-1.90."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

names = ["no MTP", "MTP drafts\n(measured)", "ceiling 1 + α\n(analytic)"]
sc.setup()
fig, ax = sc.figure("right")
ax.bar(0, 1.0, 0.6, color=sc.GRAYS[0], edgecolor=sc.INK, linewidth=1.2, hatch="//")
ax.bar(1, 1.8, 0.6, color=sc.ACCENT, edgecolor=sc.INK, linewidth=1.2)
ax.bar(2, 1.85, 0.6, color="white", edgecolor=sc.INK, linewidth=1.2, linestyle="--")
ax.bar(2, 0.05, 0.6, bottom=1.85, color=sc.ACCENT_LIGHT, edgecolor=sc.INK, linewidth=1.2, linestyle="--")
for x, h, t, c in ((0, 1.0, "1.0×", sc.INK), (1, 1.8, "1.8×", sc.ACCENT), (2, 1.9, "1.85–1.90×", sc.INK)):
    ax.text(x, h + 0.04, t, ha="center", va="bottom", fontsize=20, fontweight="bold", color=c)
ax.set_xticks(range(3))
ax.set_xticklabels(names)
ax.set_ylim(0, 2.3)
ax.set_ylabel("Decode speed\n(× of no MTP)")
sc.better(ax, "Higher is better", where="upper left")
sc.save(fig, __file__)
