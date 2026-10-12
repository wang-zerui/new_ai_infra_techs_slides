"""Total parameters vs base-model MMLU / BBH for DeepSeek-V3-Base, Kimi-K2-Base, GLM-4.5-Base.

Source: GLM-4.5 tech report (arXiv 2508.06471), Table 2 (p.14): total params 671B / 1043B / 355B;
MMLU (EM) 87.2 / 87.8 / 86.1; BBH (EM) 88.4 / 88.7 / 86.2 (GLM-4.5-Base scored with GLM's internal framework).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

models = ["DeepSeek-V3-Base", "Kimi-K2-Base", "GLM-4.5-Base"]
total = [671, 1043, 355]
scores = {"MMLU": [87.2, 87.8, 86.1], "BBH": [88.4, 88.7, 86.2]}
colors = [sc.GRAYS[0], sc.GRAYS[1], sc.ACCENT]
hatches = [sc.HATCHES[0], sc.HATCHES[1], None]

sc.setup()
fig, (a, b) = plt.subplots(1, 2, figsize=(11.8, 3.9), gridspec_kw={"width_ratios": [1, 1.5], "wspace": 0.28})

# (a) total parameters
xs = range(3)
for x, v, c, h in zip(xs, total, colors, hatches):
    a.bar(x, v, 0.62, color=c, edgecolor=sc.INK, linewidth=1.2, hatch=h)
    a.text(x, v + 25, f"{v}B", ha="center", va="bottom", fontsize=20, fontweight="bold",
           color=sc.ACCENT if x == 2 else sc.INK)
a.set_xticks([])
a.set_ylim(0, 1400)
a.set_ylabel("Total parameters (B)")
a.set_xlabel("(a) Model size", labelpad=10)
sc.better(a, "Lower is better", where="upper left")

# (b) base-model scores
w = 0.26
for g, (bench, vals) in enumerate(scores.items()):
    for i, (v, c, h) in enumerate(zip(vals, colors, hatches)):
        x = g + (i - 1) * w
        b.bar(x, v, w * 0.92, color=c, edgecolor=sc.INK, linewidth=1.2, hatch=h)
        b.text(x, v + 1.2, f"{v:.1f}", ha="center", va="bottom", fontsize=17, fontweight="bold",
               color=sc.ACCENT if i == 2 else sc.INK)
b.set_xticks([0, 1])
b.set_xticklabels(list(scores))
b.set_ylim(0, 112)
b.set_ylabel("Base-model score (EM)")
b.set_xlabel("(b) Benchmark score", labelpad=10)
sc.better(b, "Higher is better", where="upper right")

handles = [Patch(facecolor=c, edgecolor=sc.INK, hatch=h, label=m) for m, c, h in zip(models, colors, hatches)]
fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.06))
sc.save(fig, __file__)
