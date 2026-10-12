"""fig_chunked: chunked stream overlap — one GEMM cut into per-rank kernels, comm on a side stream."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 282, "fig_chunked: chunked GEMM + side-stream collective, with sync points and idle slots")

X0 = 156
YC, YK, HROW = 54, 128, 46

s.text(142, YC + HROW / 2 + 6.3, "compute", 18, INK, anchor="end")
s.text(142, YK + HROW / 2 + 6.3, "comm", 18, INK, anchor="end")
s.text(142, YK + HROW / 2 + 26, "(side stream)", 15, MUTED, anchor="end")

CW, GAP, COMM = 104, 26, 86
x = X0
for k in range(4):
    s.box(x, YC, CW, HROW, f"GEMM {k + 1}", NEU_F, NEU, size=18)
    # idle slot while the dependent chunk communication finishes
    s.rect(x + CW, YC, GAP, HROW, "url(#hatchPrb)", PRB, 1.6, 2)
    s.rect(x + CW, YK, COMM, HROW, PRB_F, PRB, 2.2, 3)
    s.text(x + CW + COMM / 2, YK + HROW / 2 + 6.3, "chunk", 17, PRB, weight="bold")
    s.line(x + CW, YC + HROW, x + CW, YK, MUTED, 1.6, dash="4,3")
    x += CW + GAP + 40

# legend
s.rect(700, 14, 26, 20, "url(#hatchPrb)", PRB, 1.6, 2)
s.text(734, 30, "stalled", 17, INK, anchor="start")

# annotations
s.text(X0, 212, "Each chunk is a small GEMM", 18, INK, anchor="start")
s.text(X0, 240, "Timing between streams is not controllable", 18, PRB, anchor="start", weight="bold")

# time axis
s.line(X0 - 6, 192, 858, 192, INK, 2, "arr")
s.text(854, 216, "time", 18, INK, anchor="end")
s.save(os.path.join(HERE, "fig_chunked.svg"))
