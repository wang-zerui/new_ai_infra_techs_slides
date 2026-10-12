"""fig_block: what layer 11 attends over. (a) Full AttnRes: the embedding and all 10 earlier outputs.
(b) Block AttnRes with S = 4: b0 = embedding, b1 = v1+..+v4, b2 = v5+..+v8, partial sum b3 = v9+v10 (AR Eq. 5-6)."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
I = "italic"
s = SVG(870, 300, "fig_block: full vs block sources for layer 11")

X0, CW, GAP = 120, 48, 8
cx = [X0 + k * (CW + GAP) for k in range(11)]


def chip(x, y, sub, fill=NEU_F, stroke=NEU, col=INK, lab="v"):
    s.rect(x, y, CW, 36, fill, stroke, 1.6, 4)
    s.rich(x + CW / 2, y + 25, [(lab, col, "bold", I), (sub, col, "bold", "sub")], 18)


def alpha_node(x, y):
    s.circle(x, y, 17, ACC_F, ACC, 2)
    s.text(x, y + 6, "α", 19, ACC, weight="bold", style="italic")
    s.line(x + 18, y, x + 40, y, INK, 2.2, "arr")
    s.rich(x + 46, y + 7, [("h", INK, "bold", I), ("11", INK, "normal", "sub")], 19, anchor="start")


AX = 770
# (a) full
YA, BUSA = 22, 76
s.text(8, YA + 16, "Full", 19, INK, anchor="start", weight="bold")
s.text(8, YA + 38, "11 sources", 16, MUTED, anchor="start")
for k, x in enumerate(cx):
    chip(x, YA, str(k))
    s.line(x + CW / 2, YA + 36, x + CW / 2, BUSA, MUTED, 1.4)
s.line(cx[0] + CW / 2, BUSA, AX - 18, BUSA, MUTED, 1.6, "arrGray")
alpha_node(AX, BUSA)

# (b) block, S = 4
YB, YS, YBB, BUSB = 124, 196, 222, 278
s.text(8, YB + 16, "Block", 19, ACC, anchor="start", weight="bold")
s.text(8, YB + 38, "4 sources", 16, MUTED, anchor="start")
for k, x in enumerate(cx):
    chip(x, YB, str(k))
groups = [(0, 0, "0", False), (1, 4, "1", True), (5, 8, "2", True), (9, 10, "3", True)]
bx = []
for a, b, lab, summed in groups:
    x0, x1 = cx[a] - 4, cx[b] + CW + 4
    if summed:
        s.rect(x0, YB - 6, x1 - x0, 48, "none", ACC, 1.6, 6, dash="5,4")
    mid = (x0 + x1) / 2
    s.line(mid, YB + 42 + 2, mid, YBB - 2, ACC, 1.8, "arrAcc")
    if summed:
        s.text(mid + 8, YS + 4, "Σ", 17, ACC, anchor="start", weight="bold")
    chip(mid - CW / 2 - (10 if lab == "3" else 0), YBB, lab, ACC_F, ACC, ACC, "b")
    bx.append(mid - (10 if lab == "3" else 0))
s.text(bx[3] + CW / 2 + 6, YBB + 24, "partial", 16, ACC, anchor="start", style="italic")
for x in bx:
    s.line(x, YBB + 36, x, BUSB, ACC, 1.6)
s.line(bx[0], BUSB, AX - 18, BUSB, ACC, 1.8, "arrAcc")
alpha_node(AX, BUSB)
s.save(os.path.join(HERE, "fig_block.svg"))
