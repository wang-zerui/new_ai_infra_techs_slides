"""Slide 2: where DSA lives — V3.1-Terminus -> V3.2 (+DSA) -> successors covered by sibling decks."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 480, 312
s = SVG(W, H, "lineage of DSA")
s.add('<g transform="translate(0,-24)">')


def two(x, y, w, h, l1, l2, new=True, dash=None):
    s.rect(x, y, w, h, ACC_F if new else NEU_F, ACC if new else NEU, 2.5 if new else 2, 6, dash)
    s.text(x + w / 2, y + h / 2 - 4, l1, 19, INK, weight="bold")
    s.text(x + w / 2, y + h / 2 + 19, l2, 17, INK)


two(2, 30, 180, 62, "V3.1-Terminus", "dense MLA", new=False)
two(298, 30, 180, 62, "DeepSeek-V3.2", "MLA + DSA")
s.arrow(184, 61, 295, 61)
s.text(240, 46, "continued", 16, MUTED, style="italic")
s.text(240, 84, "training", 16, MUTED, style="italic")

two(298, 152, 180, 62, "GLM-5", "adopts DSA")
s.arrow(388, 94, 388, 149)
two(2, 152, 180, 62, "DeepSeek-V4", "CSA · deck 01", new=False)
s.path("M330,94 V122 H92 V149", INK, 2.2, "arr")

two(298, 272, 180, 62, "GLM-5.2", "IndexShare · deck 04", new=False)
s.arrow(388, 216, 388, 269)
s.add('</g>')
s.save(os.path.join(HERE, "fig_lineage.svg"))
