"""Kimi K2 sparsity scaling law (arXiv 2507.20534 p.7): to reach validation loss 1.5, sparsity 48 needs
1.69x / 1.39x / 1.15x fewer training FLOPs than sparsity 8 / 16 / 32 (8 activated + 1 shared expert fixed).
Bars = training FLOPs relative to sparsity 48 (= 1.0). Lower is better."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

sc.setup()
fig, ax = sc.figure("right")
names = ["8", "16", "32", "48"]
vals = [1.69, 1.39, 1.15, 1.00]
sc.bars(ax, names, vals, ours=3, labels=["1.69×", "1.39×", "1.15×", "1.0×"])
ax.set_xlabel("Sparsity (routed experts ÷ active)")
ax.set_ylabel("Training FLOPs to\nreach val. loss 1.5")
ax.set_ylim(0, 2.0)
ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
sc.better(ax, "Lower is better", where="upper right")
sc.save(fig, __file__)
