"""Analytic multiply-adds per query per attention layer at L = 2^20 (see notes/analytic.py)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "notes"))
import slidechart as sc
from matplotlib.patches import Patch
import analytic as an

sc.setup()
fig, ax = sc.figure("right")
rows = [("V3.2\nDSA layer", an.dsa_v32(an.L), False),
        ("V4-Pro\nCSA layer", an.csa_v4(an.L, 128, 1024), True),
        ("V4-Flash\nCSA layer", an.csa_v4(an.L, 64, 512), True)]
for i, (name, (idx, core), ours) in enumerate(rows):
    idx, core = idx / 1e9, core / 1e9
    dark, light = (sc.ACCENT, sc.ACCENT_LIGHT) if ours else ("#595959", "#D9D9D9")
    ax.bar(i, core, 0.62, color=dark, edgecolor=sc.INK, linewidth=1.2)
    ax.bar(i, idx, 0.62, bottom=core, color=light, edgecolor=sc.INK, linewidth=1.2,
           hatch=None if ours else "//")
    ax.text(i, idx + core + 0.2, f"{idx + core:.1f} G", ha="center", va="bottom", fontsize=20,
            fontweight="bold", color=sc.ACCENT if ours else sc.INK)
ax.set_xticks(range(3)); ax.set_xticklabels([r[0] for r in rows])
ax.set_ylabel("G multiply-adds\nper query per layer")
ax.set_ylim(0, 12.5)
ax.legend(handles=[Patch(facecolor=sc.ACCENT_LIGHT, edgecolor=sc.INK, label="indexer"),
                   Patch(facecolor=sc.ACCENT, edgecolor=sc.INK, label="core attention")],
          loc="upper right", frameon=False, fontsize=18)
sc.better(ax, "Lower is better", "upper left")
sc.callout(ax, "3.9× less", xy=(1.34, 2.0), xytext=(1.62, 6.2))
sc.save(fig, __file__)
