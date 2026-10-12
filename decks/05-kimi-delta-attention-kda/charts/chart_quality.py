"""Quality at equal training budget (1.4T tokens, 48B-A3B MoE, same recipe): MLA vs GDN-H vs Kimi Linear (KDA hybrid).
Sources: Kimi Linear (arXiv 2510.26692) Table 3 (MMLU-Pro, base, 4K) and Table 5 (RULER@128K, long-context average)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc
groups = ["MMLU-Pro\n(4K)", "RULER\n(128K)", "Long-context\naverage"]
series = ["MLA (full attn)", "GDN-H", "Kimi Linear"]
vals = [[47.2, 81.3, 52.2], [47.9, 80.5, 51.2], [51.0, 84.3, 54.5]]
sc.setup()
fig, ax = sc.figure("right")
sc.grouped(ax, groups, series, vals, ours=2)
w = 0.8 / 3
for g, v in enumerate(vals[2]):
    ax.text(g - 0.4 + w * 2.5, v + 1.2, f"{v:.1f}", ha="center", va="bottom", fontsize=18, fontweight="bold", color=sc.ACCENT)
for g, v in enumerate(vals[0]):
    ax.text(g - 0.4 + w * 0.5, v + 1.2, f"{v:.1f}", ha="center", va="bottom", fontsize=16, color=sc.INK)
ax.set_ylabel("Score")
ax.set_ylim(0, 100)
sc.better(ax, "Higher is better", "upper right")
leg = ax.get_legend(); leg.set_bbox_to_anchor((0.5, 1.0))
for txt in leg.get_texts(): txt.set_fontsize(17)
sc.save(fig, __file__)
