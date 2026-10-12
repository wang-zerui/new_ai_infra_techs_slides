"""fig_param: mHC dynamic parameterization ([mHC] eq. 7-8, 10-19; [V4] eq. 3-8, §3.3). x_l (4 x C) is flattened
and RMS-normalized, one projection phi (nC x (n^2 + 2n)) yields 24 numbers (n = 4): 4 for H_pre, 4 for H_post,
16 for H_res; each = alpha * dynamic + static bias; then sigma, 2 sigma, Sinkhorn. C = 7168 for V4-Pro."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, NEU, NEU_F, INK, MUTED, sup_text

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 286, "fig_param")
# x_l: 4 lanes
for i in range(4):
    s.rect(4, 82 + i * 20, 84, 14, NEU_F, NEU, 1.4, 2)
sup_text(s, 46, 70, "x", "l", 21, INK, sub=True)
s.text(46, 186, "4 × 7168", 17, MUTED)
s.line(96, 120, 146, 120, INK, 2.2, "arr")
s.text(121, 104, "flatten", 16, MUTED)
# flattened, normalized vector
s.rect(150, 112, 150, 16, NEU_F, NEU, 1.4, 2)
s.text(225, 100, "1 × 28,672", 17, INK, weight="bold")
s.text(225, 152, "RMSNorm", 17, MUTED)
s.line(304, 120, 338, 120, INK, 2.2, "arr")
# projection
s.box(342, 84, 118, 72, "", ACC_F, ACC, sw=2.4)
s.text(401, 116, "× φ", 24, ACC, weight="bold")
s.text(401, 142, "28,672 × 24", 16, ACC, weight="bold")
s.line(464, 120, 494, 120, INK, 2.2, "arr")
# 24 outputs: 4 | 4 | 16
SX, SY, CH = 498, 22, 10
segs = [(0, 4, "4"), (4, 8, "4"), (8, 24, "16")]
for a, b, lab in segs:
    s.rect(SX, SY + a * CH, 22, (b - a) * CH, ACC_F, ACC, 2, 0)
    for k in range(a + 1, b):
        s.line(SX, SY + k * CH, SX + 22, SY + k * CH, "#9DC3E6", 1)
    s.text(SX - 6, SY + (a + b) / 2 * CH + 6, lab, 16, ACC, anchor="end", weight="bold")
# branches
rows = [(SY + 2 * CH, "σ", "pre", "1 × 4"), (SY + 6 * CH, "2σ", "post", "4 × 1"), (SY + 16 * CH, "Sinkhorn", "res", "4 × 4")]
for y, op, name, shape in rows:
    s.line(SX + 24, y, 584, y, INK, 2, "arr")
    w = 112
    s.box(586, y - 17, w, 34, op, "#FFFFFF", ACC, color=ACC, weight="bold", size=18, sw=2)
    s.line(586 + w + 2, y, 728, y, INK, 2, "arr")
    sup_text(s, 732, y + 7, "H", name, 21, INK, anchor="start")
    s.text(800, y + 7, shape, 17, MUTED, anchor="start")
s.text(4, 238, "each output = α · dynamic part + static bias b", 17, INK, anchor="start")
s.text(4, 262, "α gates start near zero (0.01 in the mHC runs)", 17, MUTED, anchor="start")
s.save(os.path.join(HERE, "fig_param.svg"))
