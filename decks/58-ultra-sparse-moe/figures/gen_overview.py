"""fig_overview: one DeepSeek-V4-Pro MoE layer. Router picks 6 of 384 routed experts (grid), the shared expert
bypasses the router, outputs are gate-weighted and summed. Config numbers from V4 §4.2.1; 66M = 3*7168*3072."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 292
s = SVG(W, H, "fig_overview: V4-Pro MoE layer, 6 of 384 routed + 1 shared")

# token and router
s.box(4, 106, 82, 50, "Token", NEU_F, NEU)
s.line(88, 131, 110, 131, INK, 2.2, "arr")
s.box(112, 98, 92, 66, ["Router", "top-6"], "#FFFFFF", NEU)
s.line(206, 131, 228, 131, INK, 2.2, "arr")

# grid of 384 routed experts
GX, GY, P, SQ, NC, NR = 232, 38, 15, 12, 32, 12
lit = {(1, 7), (3, 22), (5, 13), (7, 28), (9, 4), (10, 18)}
s.text(GX, 26, "384 routed experts · 66M weights each", 19, INK, anchor="start", weight="bold")
for r in range(NR):
    for c in range(NC):
        x, y = GX + c * P, GY + r * P
        if (r, c) in lit:
            s.rect(x - 1, y - 1, SQ + 2, SQ + 2, ACC, ACC, 1.2, 1.5)
        else:
            s.rect(x, y, SQ, SQ, NEU_F, "#A6A6A6", 1, 1.5)
GR = GX + NC * P - 3          # grid right edge
s.rect(668, 12, 14, 14, ACC, ACC, 1.2, 1.5)
s.text(690, 25, "picked for this token", 17, INK, anchor="start")
# sum node
CX, CY, R = 776, 131, 17
s.line(GR + 6, CY, CX - R - 2, CY, ACC, 2.6, "arrAcc")
s.add(f'<circle cx="{CX}" cy="{CY}" r="{R}" fill="#FFFFFF" stroke="{ACC}" stroke-width="2.4"/>')
s.line(CX - 9, CY, CX + 9, CY, ACC, 2.6); s.line(CX, CY - 9, CX, CY + 9, ACC, 2.6)
s.text(CX + 8, 78, "gate-weighted", 18, ACC, weight="bold")
s.text(CX + 8, 99, "sum", 18, ACC, weight="bold")
s.line(CX + R + 2, CY, 806, CY, INK, 2.2, "arr")
s.text(812, CY + 6.5, "out", 19, INK, anchor="start", weight="bold")

# shared expert: bypasses the router
SY = 236
s.box(GX, SY, 250, 46, "Shared expert · always on", ACC_F, ACC, color=ACC, weight="bold")
s.path(f"M45,158 V{SY + 23} H{GX - 3}", INK, 2.2, "arr")
s.path(f"M{GX + 252},{SY + 23} H{CX} V{CY + R + 3}", ACC, 2.4, "arrAcc")
s.save(os.path.join(HERE, "fig_overview.svg"))
