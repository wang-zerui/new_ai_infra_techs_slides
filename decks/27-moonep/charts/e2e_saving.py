"""End-to-end iteration-time saving of MoonEP vs DeepEP v2, labeled values of MoonEP README figure/e2e_vs_deepep.png:
H20, EP=8, seq 16k, 9 layers, H=7168, E=256, top-8, H'=2048 (sources.md, slide 12). At maxvio 20 DeepEP OOMs."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

MAXVIO = ["0.2", "1", "5", "10", "15", "20"]
saving = [-1.9, -0.5, 3.9, 7.6, 11.1, None]

sc.setup()
fig, ax = sc.figure("right")
xs = range(len(MAXVIO))
for x, v in zip(xs, saving):
    if v is None:
        continue
    col = sc.ACCENT if v > 0 else sc.ACCENT_LIGHT
    ax.bar(x, v, 0.62, color=col, edgecolor=sc.INK, linewidth=1.2)
    ax.text(x, v + (0.4 if v > 0 else -0.4), f"{v:+.1f}%", ha="center", va="bottom" if v > 0 else "top",
            fontsize=18, fontweight="bold", color=sc.ACCENT)
ax.text(5, 1.0, "DeepEP\nOOM", ha="center", va="bottom", fontsize=18, fontweight="bold", color=sc.PROBLEM)
ax.plot([5], [0.3], marker="x", markersize=16, mew=4, color=sc.PROBLEM)
ax.axhline(0, color=sc.INK, linewidth=1.4)
ax.set_xticks(list(xs)); ax.set_xticklabels(MAXVIO)
ax.set_ylim(-4.5, 15)
ax.set_xlabel("Routing imbalance (maxvio)")
ax.set_ylabel("Iteration time saved (%)")
sc.better(ax, "Higher is better", "upper left")
sc.save(fig, __file__)
