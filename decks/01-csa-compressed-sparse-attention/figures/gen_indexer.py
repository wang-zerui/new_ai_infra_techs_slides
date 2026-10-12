"""Mechanism 2: lightning indexer over compressed keys (V4 Eq. 13-17), V4-Pro shapes."""
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svgkit import SVG, ACC, ACC_FILL, ACC_LIGHT, GRAY, GRAY_FILL, INK

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 290
s = SVG(W, H)


def two(x, y, w, h, l1, l2, fill=ACC_FILL, stroke=ACC):
    s.rect(x, y, w, h, fill, stroke, 2.5 if stroke == ACC else 2, 6)
    s.text(x + w / 2, y + h / 2 - 4, l1, 18, "bold", INK)
    s.text(x + w / 2, y + h / 2 + 18, l2, 18, "normal", INK)


# inputs
two(0, 2, 180, 60, "Indexer queries", "64 heads × 128")
two(0, 92, 180, 60, "Indexer keys", "n/4 × 128")
two(0, 196, 180, 60, "Head weights", "64 per query")
# operators
two(226, 92, 170, 60, "ReLU(q · k)", "n/4 × 64")
two(436, 92, 160, 60, "Weighted sum", "n/4 scores")
two(636, 92, 130, 60, "Top-k", "k = 1,024")
two(794, 92, 72, 60, "Block", "ids", "#fff", GRAY)
# wires
s.path("M182,32 H311 V90")
s.arrow(182, 122, 224, 122)
s.arrow(398, 122, 434, 122)
s.path("M182,226 H516 V154")
s.arrow(598, 122, 634, 122)
s.arrow(768, 122, 792, 122)

# illustrative score bars (which blocks survive top-k)
random.seed(7)
vals = [0.35, 0.8, 0.25, 0.55, 0.95, 0.3, 0.2, 0.7, 0.4, 0.15, 0.85, 0.3, 0.5, 0.25]
keep = sorted(range(len(vals)), key=lambda i: -vals[i])[:4]
bx, bw, base, top = 600, 14, 262, 60
for i, v in enumerate(vals):
    h = v * top
    sel = i in keep
    s.rect(bx + i * (bw + 5), base - h, bw, h, ACC if sel else GRAY_FILL, ACC if sel else GRAY, 1.5, 1)
s.arrow(596, base, 868, base, GRAY, 1.5, head=False)
s.text(600, 284, "scores per block · keep top k", 16, "normal", GRAY, "start")
s.save(os.path.join(HERE, "fig_indexer.svg"))
