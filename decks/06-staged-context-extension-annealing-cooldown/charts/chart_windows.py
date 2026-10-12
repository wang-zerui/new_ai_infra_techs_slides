"""chart_windows: initial pretraining window vs final (served) window per model, log2 axis.
Sources (notes/sources.md): K2 §2.5 (4K, 128K via YaRN); GLM-4.5 §2.4 (4,096 -> 131,072); K3 §3.4 (8K -> 1M)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc
import numpy as np

models = ["Kimi K2", "GLM-4.5", "Kimi K3"]
bulk = [4, 4, 8]          # K tokens, bulk-pretraining window
final = [128, 128, 1024]  # K tokens, final window
lab = lambda k: f"{k // 1024}M" if k >= 1024 else f"{k}K"

sc.setup()
fig, ax = sc.figure("right")
x = np.arange(len(models)); w = 0.36
b1 = ax.bar(x - w / 2, bulk, w, color=sc.GRAYS[0], edgecolor=sc.INK, linewidth=1.2, hatch="//", label="Initial pretraining window")
b2 = ax.bar(x + w / 2, final, w, color=sc.ACCENT, edgecolor=sc.INK, linewidth=1.2, label="Final window")
ax.set_yscale("log", base=2)
ax.set_ylim(2, 4096)
ax.set_yticks([4, 32, 256, 2048 / 2])
ax.set_yticklabels(["4K", "32K", "256K", "1M"])
ax.minorticks_off()
for r, v in zip(b1, bulk):
    ax.text(r.get_x() + r.get_width() / 2, v * 1.25, lab(v), ha="center", va="bottom", fontsize=20, fontweight="bold", color=sc.INK)
for r, v in zip(b2, final):
    ax.text(r.get_x() + r.get_width() / 2, v * 1.25, lab(v), ha="center", va="bottom", fontsize=20, fontweight="bold", color=sc.ACCENT)
ax.set_xticks(x); ax.set_xticklabels(models)
ax.set_ylabel("Window (tokens, log scale)")
sc.top_legend(ax, ncol=2)
sc.save(fig, __file__)
