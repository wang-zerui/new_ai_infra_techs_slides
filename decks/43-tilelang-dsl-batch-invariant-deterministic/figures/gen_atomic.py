"""fig_atomic: sparse-attention backward, gradient of one KV token (DeepSeek-V4 §3.3).
(a) atomicAdd: contributions land in arrival order; (b) one accumulation buffer per SM, then a global
deterministic summation in a fixed order. Schematic: 4 SMs."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED, badge

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 252
s = SVG(W, H, "fig_atomic")

YS = [44, 94, 144, 194]
SH = 36

# ---- (a) atomicAdd
s.text(10, 22, "(a) atomicAdd into one buffer", 19, INK, anchor="start", weight="bold")
tgt = (300, 116, 110, 44)
s.box(*tgt, "dKV[j]", PRB_F, PRB, INK, 18, weight="bold", sw=2)
arrival = [3, 1, 4, 2]
for i, y in enumerate(YS):
    s.box(20, y, 96, SH, f"SM{i}", NEU_F, NEU, INK, 17, sw=1.8)
    x0, y0 = 118, y + SH / 2
    x1, y1 = tgt[0] - 2, tgt[1] + 8 + i * 9.3
    s.line(x0, y0, x1, y1, PRB, 2, "arrPrb")
    mx, my = x0 + (x1 - x0) * 0.45, y0 + (y1 - y0) * 0.45
    badge(s, mx, my, arrival[i], PRB, r=11)
s.text(355, 196, "order =", 17, PRB, weight="bold")
s.text(355, 218, "arrival time", 17, PRB, weight="bold")

s.line(436, 8, 436, 244, "#BFBFBF", 1.2)

# ---- (b) per-SM buffers + fixed-order sum
s.text(450, 22, "(b) Per-SM buffers + ordered sum", 19, ACC, anchor="start", weight="bold")
sumbox = (716, 96, 146, 56)
for i, y in enumerate(YS):
    s.box(452, y, 84, SH, f"SM{i}", NEU_F, NEU, INK, 17, sw=1.8)
    s.line(538, y + SH / 2, 574, y + SH / 2, ACC, 2, "arrAcc")
    s.box(576, y, 84, SH, f"buf{i}", ACC_F, ACC, INK, 17, sw=1.8)
    s.line(662, y + SH / 2, sumbox[0] - 2, sumbox[1] + 10 + i * 12, ACC, 2, "arrAcc")
s.box(*sumbox, ["sum buffers", "in SM order"], ACC_F, ACC, INK, 17, sw=2)
s.line(sumbox[0] + sumbox[2] / 2, sumbox[1] + sumbox[3] + 2, sumbox[0] + sumbox[2] / 2, 194, ACC, 2.2, "arrAcc")
s.box(739, 196, 100, 40, "dKV[j]", "#FFFFFF", ACC, INK, 18, weight="bold", sw=2)
s.save(os.path.join(HERE, "fig_atomic.svg"))
