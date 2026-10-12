"""chart_span (analytic): run-average attention work per token relative to a 4K-only run.
Attention-score work per token is proportional to the training window (causal attention reads every earlier token).
Token-weighted means from published stage budgets (notes/sources.md, derived 3):
  128K window on every token: 32x; GLM-4.5 staged: (22*1 + 1.0*8 + 0.1*32)/23.1 = 1.44x; K2 staged: (15.9*1 + 0.06*8)/15.96 = 1.03x."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

names = ["128K window\non every token", "GLM-4.5\nstaged", "Kimi K2\nstaged"]
vals = [32.0, 33.2 / 23.1, 16.38 / 15.96]
labels = ["32×", f"{vals[1]:.2f}×", f"{vals[2]:.2f}×"]

sc.setup()
fig, ax = sc.figure("right")
sc.bars(ax, names, vals, ours=[1, 2], problem=0, labels=labels, width=0.58)
ax.set_ylabel("Attention work per token\n(× a 4K-only run)")
ax.set_ylim(0, 39)
sc.better(ax, "Lower is better", where="upper right")
sc.save(fig, __file__)
