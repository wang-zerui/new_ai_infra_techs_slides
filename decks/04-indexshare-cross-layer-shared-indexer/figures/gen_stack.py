"""Slide 2 (status quo): every DSA layer runs its own lightning indexer over all L keys."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 480, 322
s = SVG(W, H, "DSA layer stack, one indexer per layer")
rows = ["layer ℓ+3", "layer ℓ+2", "layer ℓ+1", "layer ℓ"]
y0, rh, gap = 6, 54, 22
for i, name in enumerate(rows):
    y = y0 + i * (rh + gap)
    s.text(0, y + rh / 2 + 6, name, 17, MUTED, anchor="start")
    s.box(92, y, 132, rh, ["Indexer", "all L keys"], PRB_F, PRB, INK, 17, "bold", 2.2, 5)
    s.box(250, y, 84, rh, "Top-k", NEU_F, NEU, INK, 18, "bold", 2, 5)
    s.box(360, y, 118, rh, ["Sparse", "MLA"], NEU_F, NEU, INK, 18, "bold", 2, 5)
    s.arrow(226, y + rh / 2, 247, y + rh / 2)
    s.arrow(336, y + rh / 2, 357, y + rh / 2)
    if i < len(rows) - 1:
        s.arrow(419, y + rh + gap - 2, 419, y + rh + 2)   # hidden state flows upward
s.text(285, H - 6, "× 78 layers in GLM-5", 18, PRB, weight="bold")
s.save(os.path.join(HERE, "fig_stack.svg"))
