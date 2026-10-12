"""fig_overview: one Stable LatentMoE layer at K3 shapes (K3 Eq. 11, Table 1; HF modeling_kimi_linear.py).
Router and shared experts see the full 7168-wide x; routed experts work in the 3584-wide latent;
RMSNorm sits between the weighted sum and W_up; every GLU uses SiTU."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 330, "fig_overview: Stable LatentMoE layer")
I = "italic"


def rbox(x, y, w, h, lines, fill, stroke, size=19):
    s.rect(x, y, w, h, fill, stroke, 2, 5)
    n = len(lines)
    step = size * 1.2
    y0 = y + h / 2 - step * (n - 1) / 2 + 0.35 * size
    for i, segs in enumerate(lines):
        s.rich(x + w / 2, y0 + i * step, segs, size)


CY = 176
# input and bus
s.rich(18, CY + 8, [("x", INK, "bold", I)], 24)
s.text(28, CY + 34, "7168", 16, MUTED)
s.line(34, CY, 58, CY, INK, 2.2)
s.line(58, 62, 58, 292, INK, 2.2)
for y in (62, CY, 292):
    s.line(58, y, 78, y, INK, 2.2, "arr")

# router (existing) + QB bias (new, deck 56)
rbox(80, 36, 300, 52, [[("router + ", INK, "normal"), ("QB bias", ACC, "bold")],
                       [("top-16 of 896 → ", INK, "normal"), ("p", INK, "normal", I), ("i", INK, "normal", "sub")]],
     NEU_F, NEU, 18)
# W_down (new)
rbox(80, CY - 28, 118, 56, [[("W", ACC, "bold"), ("↓", ACC, "bold")], [("7168 → 3584", INK, "normal")]],
     ACC_F, ACC, 18)
s.badge(80, CY - 28, 1)
s.line(198 + 2, CY, 236 - 1, CY, INK, 2.2, "arr")
# routed experts (latent), stack effect
EX, EY, EW, EH = 238, CY - 44, 250, 88
for off in (8, 4):
    s.rect(EX + off, EY - off, EW, EH, "#FFFFFF", "#A6A6A6", 1.4, 5)
rbox(EX, EY, EW, EH, [[("16 latent experts", INK, "bold")], [("3584 → 3072 → 3584", INK, "normal")],
                      [("SiTU-GLU", ACC, "bold")]], NEU_F, NEU, 18)
s.badge(EX + EW - 2, EY + EH - 2, 3)
# routing weights into the experts
s.line(330, 88 + 2, 330, EY - 8 - 1, INK, 2.2, "arr")
# weighted sum
SX = 530
s.line(EX + EW + 10, CY, SX - 19, CY, INK, 2.2, "arr")
s.circle(SX, CY, 18, "#FFFFFF", NEU, 2)
s.text(SX, CY + 7, "Σ", 20, INK, weight="bold")
s.rich(SX, CY - 28, [("u", INK, "normal", I)], 19)
# RMSNorm (new)
rbox(560, CY - 28, 108, 56, [[("RMSNorm", ACC, "bold")], [("3584", INK, "normal")]], ACC_F, ACC, 18)
s.badge(560, CY - 28, 2)
s.line(SX + 19, CY, 560 - 1, CY, INK, 2.2, "arr")
# W_up (new)
rbox(688, CY - 28, 118, 56, [[("W", ACC, "bold"), ("↑", ACC, "bold")], [("3584 → 7168", INK, "normal")]],
     ACC_F, ACC, 17)
s.line(668 + 2, CY, 688 - 1, CY, INK, 2.2, "arr")
# merge with shared branch
PX = 832
s.line(806 + 2, CY, PX - 15, CY, INK, 2.2, "arr")
s.circle(PX, CY, 14, "#FFFFFF", INK, 2)
s.text(PX, CY + 7, "+", 20, INK, weight="bold")
s.line(PX, CY - 15, PX, 108, INK, 2.2, "arr")
s.rich(PX, 98, [("y", INK, "bold", I)], 22)
# shared experts (full width)
rbox(80, 266, 600, 52, [[("2 shared experts, full width: 7168 → 2 × 3072 → 7168 · ", INK, "normal"),
                         ("SiTU-GLU", ACC, "bold")]], NEU_F, NEU, 18)
s.path(f"M 680 292 H {PX} V {CY + 15}", INK, 2.2, "arr")
s.save(os.path.join(HERE, "fig_overview.svg"))
