"""Aux-loss dilemma, 1B MoE (arXiv 2408.15664). x = MaxVio_global, y = validation perplexity.

alpha = 1e-3 point and the Loss-Free point are Table 2 (0.72, 9.56) and (0.04, 9.50).
alpha = 1e-2, 1e-4, 0 are digitized from Fig. 2 (raster extracted from the PDF, mapped through the figure's
gridlines; the two Table-2 points are recovered within 0.005): (0.10, 9.69), (2.23, 9.53), (5.07, 9.54).
Writes dilemma.png/pdf (aux-loss sweep only) and dilemma_lf.png/pdf (+ Loss-Free point).
"""
import os
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

alphas = ["α = 0.01", "α = 1e-3", "α = 1e-4", "α = 0"]
maxvio = [0.10, 0.72, 2.23, 5.07]
ppl = [9.69, 9.56, 9.53, 9.54]
LF = (0.04, 9.50)
HERE = os.path.dirname(os.path.abspath(__file__))


def draw(with_lf: bool):
    sc.setup()
    fig, ax = sc.figure("right")
    ax.plot(maxvio, ppl, "--", color=sc.GRAYS[2], lw=1.8, zorder=1)
    ax.scatter(maxvio, ppl, s=150, color=sc.GRAYS[2], edgecolor=sc.INK, linewidth=1.2, zorder=3,
               label="aux loss (α sweep)")
    offs = [(14, -2, "left"), (14, -2, "left"), (0, -30, "center"), (0, 16, "center")]
    for a, x, y, (dx, dy, ha) in zip(alphas, maxvio, ppl, offs):
        ax.annotate(a, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va="bottom" if dy > 0 else ("center" if dy > -10 else "bottom"),
                    fontsize=17, color=sc.INK)
    ax.annotate("perplexity cost", (0.10, 9.69), xytext=(0.30, 9.665), color=sc.PROBLEM, fontsize=18,
                fontweight="bold", ha="left", va="center")
    ax.annotate("imbalance", (5.07, 9.54), xytext=(3.4, 9.59), color=sc.PROBLEM, fontsize=18,
                fontweight="bold", ha="center", va="center")
    if with_lf:
        ax.scatter([LF[0]], [LF[1]], s=520, marker="*", color=sc.ACCENT, edgecolor=sc.INK, linewidth=1.0,
                   zorder=4, label="Loss-Free")
        ax.annotate("", xy=(0.052, 9.503), xytext=(0.62, 9.553),
                    arrowprops=dict(arrowstyle="-|>", color=sc.ACCENT, lw=2.4))
        ax.text(0.062, 9.487, "Loss-Free", color=sc.ACCENT, fontsize=19, fontweight="bold", ha="left", va="center")
    ax.set_xscale("log")
    ax.set_xlim(0.025, 9)
    ax.set_xticks([0.05, 0.1, 0.5, 1, 5])
    ax.set_xticklabels(["0.05", "0.1", "0.5", "1", "5"])
    ax.set_ylim(9.47, 9.72)
    ax.set_yticks([9.50, 9.55, 9.60, 9.65, 9.70])
    ax.set_xlabel("MaxVio$_{global}$ (log scale)")
    ax.set_ylabel("Validation perplexity")
    sc.better(ax, "Lower-left is better", where="upper right")
    name = "dilemma_lf.py" if with_lf else "dilemma.py"
    sc.save(fig, os.path.join(HERE, name))


draw(False)
draw(True)
