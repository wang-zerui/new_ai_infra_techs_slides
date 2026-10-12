"""Downstream accuracy of the 12B model pretrained on 10T tokens, FP8 vs NVFP4 (arXiv 2509.25149, Table 2,
category averages). See notes/sources.md."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc  # noqa: E402

groups = ["General", "Math", "Multilingual", "Code", "Commonsense"]
series = ["FP8", "NVFP4"]
vals = [[68.99, 86.20, 77.93, 59.52, 77.29],
        [69.82, 86.88, 80.24, 56.67, 76.75]]

sc.setup()
fig, ax = sc.plt.subplots(figsize=(10.4, 3.7))
sc.grouped(ax, groups, series, vals, ours=1, width=0.72)
w = 0.36
for g in range(len(groups)):
    ax.text(g - w / 2, vals[0][g] + 1.5, f"{vals[0][g]:.1f}", ha="center", va="bottom", fontsize=16, color=sc.INK)
    ax.text(g + w / 2, vals[1][g] + 1.5, f"{vals[1][g]:.1f}", ha="center", va="bottom", fontsize=16,
            color=sc.ACCENT, fontweight="bold")
ax.set_ylim(0, 105)
ax.set_ylabel("Accuracy (%)")
sc.better(ax, "Higher is better", where="upper left")
sc.save(fig, __file__)
