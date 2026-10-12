"""Mechanism 3: what one query attends to (selected compressed blocks + 128 raw window tokens), then
shared-KV MQA and the grouped output projection (V4 §2.3.1, §2.3.3; V4-Pro shapes)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svgkit import SVG, ACC, ACC_FILL, GRAY, GRAY_FILL, INK, RED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 290
s = SVG(W, H)

# ---- coverage of one query's context (schematic, not to scale)
s.text(0, 52, "Compressed", 18, "bold", INK, "start")
s.text(0, 74, "blocks", 18, "bold", INK, "start")
X0, CW, P = 150, 28, 34
sel = {1, 4, 8, 11, 14}
nblk = 18
for i in range(nblk):
    x = X0 + i * P
    s.rect(x, 40, CW, 36, ACC if i in sel else GRAY_FILL, ACC if i in sel else GRAY, 2, 3)
own_x = X0 + nblk * P
s.rect(own_x, 40, 2 * CW, 36, "url(#hatchRed)", RED, 2, 3)


s.text(0, 128, "Window", 18, "bold", INK, "start")
s.text(0, 150, "128 tokens", 18, "normal", INK, "start")
wx0 = X0 + 12 * P
n_tok = 22
tw = (own_x + 2 * CW - wx0) / n_tok
for j in range(n_tok):
    s.rect(wx0 + j * tw + 1, 118, tw - 3, 30, ACC_FILL, ACC, 1.5, 1)
s.text(wx0 - 14, 142, "…", 20, "bold", GRAY)
qx = own_x + 2 * CW + 14
s.arrow(qx, 158, qx, 8, INK, 2, head=False)
s.text(qx - 4, 178, "query t", 16, "bold", INK, "end")
s.text(qx - 8, 26, "own block · not yet compressed", 16, "bold", RED, "end")
s.arrow(0, 192, 870, 192, "#BFBFBF", 1.2, dash="5,4", head=False)
s.text(X0, 102, "top-k picks: filled", 16, "normal", GRAY, "start")

# ---- concat -> shared-KV MQA -> grouped output projection
y, h = 206, 64


def two(x, w, l1, l2, fill=ACC_FILL, stroke=ACC):
    s.rect(x, y, w, h, fill, stroke, 2.5 if stroke == ACC else 2, 6)
    s.text(x + w / 2, y + h / 2 - 4, l1, 18, "bold", INK)
    s.text(x + w / 2, y + h / 2 + 18, l2, 18, "normal", INK)


two(0, 190, "1,024 + 128", "entries × 512")
two(240, 270, "Shared-KV MQA", "128 heads · K = V")
two(560, 306, "Grouped out-proj", "16 groups → 7,168")
s.arrow(192, y + h / 2, 238, y + h / 2)
s.arrow(512, y + h / 2, 558, y + h / 2)
s.save(os.path.join(HERE, "fig_attend.svg"))
