"""fig_router: where the router GEMM sits in an MoE layer, and why its precision matters (near-tied top-k)."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED, rich

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 480, 340
s = SVG(W, H, "fig_router: x (BF16) and W (FP32) -> router GEMM -> FP32 scores -> top-k -> experts")

# row A: operands
s.box(8, 8, 214, 58, ["hidden states x", "BF16 · M × K"], NEU_F, NEU, size=18)
s.box(258, 8, 214, 58, ["router weight W", "FP32 · N × K"], NEU_F, NEU, size=18)
# row B: router GEMM
s.box(8, 102, 464, 52, "router GEMM   x · Wᵀ", ACC_F, ACC, color=ACC, weight="bold", size=20)
s.line(115, 66, 115, 100, INK, 2.2, "arr")
s.line(365, 66, 365, 100, INK, 2.2, "arr")
# row C: scores of one token as a mini bar chart
s.text(8, 186, "scores of one token", 18, INK, anchor="start")
s.rect(300, 172, 18, 18, ACC, ACC, 2, 2)
s.text(326, 187, "selected (k = 2)", 17, INK, anchor="start")
s.line(240, 154, 240, 194, INK, 2.2, "arr")
base = 316
vals = [38, 92, 55, 30, 88, 70, 24, 86]   # schematic heights, not data
x0, bw, gap = 14, 38, 14
for i, v in enumerate(vals):
    x = x0 + i * (bw + gap)
    top3 = i in (1, 4)
    tied = i == 7
    fill = ACC if top3 else (PRB_F if tied else NEU_F)
    stroke = ACC if top3 else (PRB if tied else NEU)
    s.rect(x, base - v, bw, v, fill, stroke, 2, 2, dash="5,3" if tied else None)
    s.text(x + bw / 2, base + 20, f"e{i}", 16, MUTED)
s.line(6, base, x0 + 8 * (bw + gap) - gap + 8, base, INK, 1.5)
# annotations
xk = x0 + 4 * (bw + gap) + bw / 2
xt = x0 + 7 * (bw + gap) + bw / 2
s.text(xk, base - vals[4] - 10, "k-th", 17, ACC, weight="bold")
s.text(xt, base - vals[7] - 12, "(k+1)-th", 17, PRB, weight="bold")
s.line(xk + bw / 2 + 2, base - vals[4], xt - bw / 2 - 2, base - vals[7], PRB, 1.6, dash="4,3")
s.save(os.path.join(HERE, "fig_router.svg"))
