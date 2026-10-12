"""Mechanism 2: HCA reads every compressed entry plus the raw window; no indexer, no top-k
(contrast with CSA, deck 01). V4 §2.3.2, §2.3.3."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svgkit import SVG, ACC, ACC_FILL, GRAY, GRAY_FILL, INK, RED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 288
s = SVG(W, H)

# --- coverage for one query
s.text(0, 48, "HCA reads", 18, "bold", ACC, "start")
s.text(0, 70, "every entry", 18, "bold", ACC, "start")
X0, BW, G, NB = 150, 66, 6, 8
for i in range(NB):
    s.rect(X0 + i * (BW + G), 34, BW, 40, ACC_FILL, ACC, 2.5, 4)
own = X0 + NB * (BW + G)
s.rect(own, 34, 54, 40, "url(#hatchRed)", RED, 2, 4)
qx = own + 54 + 14
s.arrow(qx, 4, qx, 124, INK, 2, head=False)
s.text(qx - 4, 142, "query", 16, "bold", INK, "end")
n_tok = 16
wx0 = own - 4 * (BW + G) + 30
tw = (own + 54 - wx0) / n_tok
for j in range(n_tok):
    s.rect(wx0 + j * tw + 1, 86, tw - 3, 30, ACC_FILL, ACC, 1.5, 1)
s.text(wx0 - 10, 108, "+ 128 raw tokens", 18, "normal", INK, "end")
s.text(qx - 8, 24, "partial block", 16, "bold", RED, "end")
s.arrow(0, 154, 870, 154, "#BFBFBF", 1.2, dash="5,4", head=False)

# --- pipeline contrast: CSA (gray) vs HCA (accent)
def row(y, name, color, boxes, fill, stroke):
    s.text(0, y + 29, name, 20, "bold", color, "start")
    for (x, w, l1, l2) in boxes:
        if l1 is None:
            continue
        s.rect(x, y, w, 46, fill, stroke, 2.5 if stroke == ACC else 2, 6)
        s.text(x + w / 2, y + 30, l1, 18, "bold", INK)

cols = [(84, 170), (300, 200), (546, 160), (752, 114)]
y1, y2 = 168, 234
row(y1, "CSA", INK, [(84, 170, "n/4 entries", None), (300, 200, "Indexer + top-k", None),
                     (546, 160, "k + 128", None), (752, 114, "MQA", None)], GRAY_FILL, GRAY)
row(y2, "HCA", ACC, [(84, 170, "n/128 entries", None), (300, 200, None, None),
                     (546, 160, "all + 128", None), (752, 114, "MQA", None)], ACC_FILL, ACC)
for y in (y1, y2):
    c = GRAY if y == y1 else ACC
    if y == y1:
        s.arrow(256, y + 23, 298, y + 23, INK)
        s.arrow(502, y + 23, 544, y + 23, INK)
    else:
        s.arrow(256, y + 23, 544, y + 23, ACC, 2.5)
        s.text(400, y + 16, "no selector", 18, "bold", ACC)
    s.arrow(708, y + 23, 750, y + 23, INK if y == y1 else ACC, 2.5 if y == y2 else 2.2)
s.save(os.path.join(HERE, "fig_dense.svg"))
