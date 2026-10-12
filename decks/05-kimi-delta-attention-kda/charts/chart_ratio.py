"""Hybrid-ratio ablation (KDA:MLA layers) -> validation perplexity.
Source: Kimi Linear (arXiv 2510.26692) Table 1, validation PPL column (same FLOPs budget, 16-layer scaling model)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc
ratios = ["0:1\n(all MLA)", "1:1", "3:1", "7:1", "15:1"]
val_ppl = [5.77, 5.66, 5.65, 5.70, 5.82]
sc.setup()
fig, ax = sc.figure("right")
xs = range(len(ratios))
ax.plot(xs, val_ppl, color=sc.GRAYS[2], lw=2.5, zorder=1)
for x, v in zip(xs, val_ppl):
    ours = ratios[x] == "3:1"
    ax.scatter([x], [v], s=260 if ours else 150, color=sc.ACCENT if ours else sc.GRAYS[1], edgecolor=sc.INK, lw=1.2, zorder=3)
    ax.text(x, v + 0.017, f"{v:.2f}", ha="center", va="bottom", fontsize=19, fontweight="bold",
            color=sc.ACCENT if ours else sc.INK)
ax.set_xticks(list(xs)); ax.set_xticklabels(ratios)
ax.set_xlabel("KDA : MLA layers")
ax.set_ylabel("Validation PPL")
ax.set_ylim(5.58, 5.90)
ax.set_xlim(-0.5, len(ratios) - 0.5)
sc.better(ax, "Lower is better", "upper left")
sc.save(fig, __file__)
