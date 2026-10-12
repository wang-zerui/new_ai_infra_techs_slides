"""fig_order: the same dot product summed in two orders depending on batch size (split-K vs one SM).
Illustrates [TM] and DeepSeek-V4 §3.3: split heuristics chosen from the batch shape change the reduction order."""
import os
from svgkit import SVG, ACC, PRB, NEU, NEU_F, INK, MUTED, opnode

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 270
s = SVG(W, H, "fig_order")

LW, LH = 62, 32


def leaves(x0, y, gap):
    xs = []
    for i in range(4):
        x = x0 + i * (LW + gap)
        s.box(x, y, LW, LH, f"K{i}", NEU_F, NEU, INK, 18, sw=1.8)
        xs.append(x + LW / 2)
    return xs


# ---- panel (a): small batch -> split-K tree
s.text(205, 22, "Small batch: split-K over 4 SMs", 19, INK, weight="bold")
xa = leaves(28, 44, 46)
for i, x in enumerate(xa):
    s.text(x, 96, f"SM{i}", 15, MUTED)
# level 1 adds
l1 = [((xa[0] + xa[1]) / 2, 136), ((xa[2] + xa[3]) / 2, 136)]
for (cx, cy), (a, b) in zip(l1, ((xa[0], xa[1]), (xa[2], xa[3]))):
    s.path(f"M{a},{104} L{a},{cy} L{cx - 13},{cy}", INK, 2)
    s.path(f"M{b},{104} L{b},{cy} L{cx + 13},{cy}", INK, 2)
    opnode(s, cx, cy)
root = (205, 188)
s.path(f"M{l1[0][0]},{l1[0][1] + 13} L{l1[0][0]},{root[1]} L{root[0] - 13},{root[1]}", INK, 2)
s.path(f"M{l1[1][0]},{l1[1][1] + 13} L{l1[1][0]},{root[1]} L{root[0] + 13},{root[1]}", INK, 2)
opnode(s, *root)
s.line(root[0], root[1] + 13, root[0], 226, INK, 2.2, "arr")
s.box(root[0] - 40, 228, 80, 36, "y", "#FFFFFF", NEU, INK, 20, weight="bold", sw=2)

# divider
s.line(432, 10, 432, 206, "#BFBFBF", 1.2)

# ---- panel (b): large batch -> one SM accumulates K in order
s.text(655, 22, "Large batch: one SM, K in order", 19, INK, weight="bold")
xb = leaves(478, 44, 46)
chain_y = [126, 160, 194]
# first add: K0 + K1
cx = xb[1]
s.line(xb[0], 78, xb[0], chain_y[0], INK, 2)
s.line(xb[0], chain_y[0], cx - 13, chain_y[0], INK, 2)
s.line(xb[1], 78, xb[1], chain_y[0] - 13, INK, 2)
opnode(s, cx, chain_y[0])
# second add: (...) + K2
s.path(f"M{cx},{chain_y[0] + 13} L{cx},{chain_y[1]} L{xb[2] - 13},{chain_y[1]}", INK, 2)
s.line(xb[2], 78, xb[2], chain_y[1] - 13, INK, 2)
opnode(s, xb[2], chain_y[1])
# third add: (...) + K3
s.path(f"M{xb[2]},{chain_y[1] + 13} L{xb[2]},{chain_y[2]} L{xb[3] - 13},{chain_y[2]}", INK, 2)
s.line(xb[3], 78, xb[3], chain_y[2] - 13, INK, 2)
opnode(s, xb[3], chain_y[2])
s.path(f"M{xb[3]},{chain_y[2] + 13} L{xb[3]},{214} L{695},{214} L{695},{226}", INK, 2.2, "arr")
s.box(655, 228, 80, 36, "y′", "#FFFFFF", NEU, INK, 20, weight="bold", sw=2)

# mismatch marker between the two outputs
s.line(250, 250, 650, 250, PRB, 2.4, dash="7,5")
s.text(450, 234, "different rounding: y ≠ y′", 18, PRB, weight="bold", halo=True)
s.save(os.path.join(HERE, "fig_order.svg"))
