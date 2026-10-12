"""Speedup over full activation recomputation (TP8CP1), 175B GPT, BF16 + MHA, sequence length 32K.

Source: Megatron-LM context_parallel.md, Fig. 2 ("Speedup of 175B GPT with various TP+CP combinations compared
to full recomputation"). The figure has no value labels; values were pixel-measured from the bar chart
(see notes/sources.md), hence the "≈".
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

names = ["TP8\nfull recompute", "TP32\nno CP", "TP16\nCP2", "TP8\nCP4"]
vals = [1.00, 0.85, 1.15, 1.20]
labels = ["1.0×", "≈0.85×", "≈1.15×", "≈1.20×"]

sc.setup()
fig, ax = sc.figure("right")
sc.bars(ax, names, vals, ours=[2, 3], problem=1, labels=labels)
ax.set_ylim(0, 1.45)
ax.set_yticks([0, 0.5, 1.0])
ax.set_ylabel("Speedup over full\nrecomputation (×)")
sc.better(ax, "Higher is better", where="upper left")
sc.save(fig, __file__)
