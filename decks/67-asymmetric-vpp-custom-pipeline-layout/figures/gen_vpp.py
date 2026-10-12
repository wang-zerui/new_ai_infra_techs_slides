"""fig_vpp: uniform interleaved assignment, 24 layers, PP 4, VPP 2 ([MC] Sec. 7.5 example).

Stage s = chunk * PP + rank holds layers 3s..3s+2; data flows s0 -> s1 -> ... -> s7.
"""
import os
from svgkit import SVG, ACC, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(480, 266, "fig_vpp: 24 layers on PP4 x VPP2, interleaved")
X0, W, G = 92, 80, 16
Y = [44, 156]
H = 54
for p in range(4):
    s.text(X0 + p * (W + G) + W / 2, 28, f"GPU {p}", 17, INK, weight="bold")
for v in range(2):
    s.text(X0 - 12, Y[v] + H / 2 + 6, f"chunk {v}", 17, INK, anchor="end")
    for p in range(4):
        st = v * 4 + p
        x = X0 + p * (W + G)
        s.box(x, Y[v], W, H, f"L{3 * st}–{3 * st + 2}", NEU_F, NEU, size=18)
        if p < 3:
            s.line(x + W + 1, Y[v] + H / 2, x + W + G - 1, Y[v] + H / 2, INK, 2, "arr")
# wrap from GPU3 chunk0 to GPU0 chunk1
x3 = X0 + 3 * (W + G) + W / 2
x0 = X0 + W / 2
ym = (Y[0] + H + Y[1]) / 2
s.path(f"M{x3:.1f},{Y[0] + H + 2} V{ym:.1f} H{x0:.1f} V{Y[1] - 2}", ACC, 2.2, "arrAcc")
s.text((x0 + x3) / 2, ym - 8, "next chunk", 16, ACC, weight="bold")
s.text(X0, 250, "24 layers · PP 4 · VPP 2 · 3 layers per chunk", 16, MUTED, anchor="start", style="italic")
s.save(os.path.join(HERE, "fig_vpp.svg"))
