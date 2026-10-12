"""fig_combine: HybridEP combine with fused, hierarchical reduction for token t (home GPU A0; experts on
B0, B1, B3 in domain B and A1 in domain A).

Domain B pre-reduces the partial outputs on B0 (same local index as A0), the RDMA warp group sends one partial
sum to A0 with IBGDA, and A0's reduce warps add the domain-A partials and TMA-store row t in original order.
Sources: arXiv 2603.07685 Sec. 4.2.2 p.31; NVIDIA blog (Feb 2026) Fig. 3 text; Hybrid-EP_Implementation.md
Sec. 8.2.
"""
import os
from svgkit import SVG, ACC, ACC_F, PRB, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 256, "fig_combine: hierarchical fused reduction")


def sigma(cx, cy):
    s.add(f'<circle cx="{cx}" cy="{cy}" r="25" fill="{ACC_F}" stroke="{ACC}" stroke-width="2.4"/>')
    s.text(cx, cy + 8.5, "Σ", 25, ACC, weight="bold")


# ---- domain B
s.text(166, 20, "NVLink domain B", 17, INK, weight="bold")
s.rect(4, 32, 324, 214, "#FFFFFF", NEU, 2, 6)
src = [("B1", 52), ("B3", 116), ("B0", 180)]
SX, SW, SH = 20, 130, 40
CB = (262, 136)
CA = (650, 136)
for name, y in src:
    s.box(SX, y, SW, SH, f"{name} expert out", NEU_F, NEU, size=16)
    s.line(SX + SW + 2, y + SH / 2, CB[0] - 26, CB[1] + (y + SH / 2 - CB[1]) * 0.22, NEU, 2.2, "arr")
sigma(*CB)
s.text(CB[0], CB[1] + 50, "pre-reduce", 16, ACC, weight="bold")
s.text(CB[0], CB[1] + 70, "on B0", 16, ACC)

# ---- domain A
s.text(702, 20, "NVLink domain A", 17, INK, weight="bold")
s.rect(540, 32, 326, 214, "#FFFFFF", NEU, 2, 6)
s.box(560, 192, 130, 40, "A1 expert out", NEU_F, NEU, size=16)
s.line(CA[0] - 25, 190, CA[0] - 12, CA[1] + 24, NEU, 2.2, "arr")
sigma(*CA)
s.text(CA[0], CA[1] - 40, "final reduce on A0", 16, ACC, weight="bold")
s.box(734, 114, 120, 44, "row t", "#FFFFFF", ACC, color=INK, size=17, weight="bold", sw=2.4)
s.line(CA[0] + 27, CA[1], 732, CA[1], ACC, 2.6, "arrAcc")
s.text(706, CA[1] - 10, "TMA", 16, ACC, weight="bold")
s.text(794, 182, "original order", 16, INK)
# ---- RDMA hop
CA = (650, 136)
s.line(CB[0] + 27, CB[1], CA[0] - 28, CA[1], ACC, 2.8, "arrAcc")
s.text((CB[0] + CA[0]) / 2, CB[1] - 12, "IBGDA", 17, ACC, weight="bold")
s.text((CB[0] + CA[0]) / 2, CB[1] + 26, "one partial sum", 16, INK)

s.save(os.path.join(HERE, "fig_combine.svg"))
