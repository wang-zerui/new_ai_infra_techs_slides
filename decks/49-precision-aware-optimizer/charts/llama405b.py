"""Llama-405B memory by component, BF16 training with Adam (approximate totals, not per GPU).
Source: arXiv 2603.07685, Sec. 3.1.1 (p.12): parameters ~810 GB, optimizer states ~4860 GB, gradients ~1620 GB,
activations (8K sequence) ~5575 GB.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

names = ["BF16\nweights", "Gradients", "Adam\nstate", "Activations\n(8K seq)"]
vals = [810, 1620, 4860, 5575]
labels = ["810", "1,620", "4,860", "5,575"]

sc.setup()
fig, ax = sc.figure("right")
rects = sc.bars(ax, names, vals, ours=[], problem=2, labels=labels)
ax.set_ylabel("Memory, whole model (GB)")
ax.set_ylim(0, 6800)
sc.callout(ax, "6× the weights", xy=(1.68, 4000), xytext=(0.75, 3700), color=sc.PROBLEM, size=20)
sc.save(fig, __file__)
