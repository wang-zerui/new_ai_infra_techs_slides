"""fig_route: how one MoE layer chooses experts (DeepSeek-V3 eqs. 12-16; DeepSeek-V4 §2.1 uses Sqrt(Softplus)).
Every input to the top-K choice comes from the current parameters: backbone (hidden state), router centroids, bias."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 480, 334
s = SVG(W, H, "fig_route")

MX, MW, BH = 180, 290, 46
rows = [
    (6, "hidden state u", "backbone θ"),
    (88, "s = √softplus(uᵀe)", "router e"),
    (170, "top-6 of s + b", "bias b"),
]
for y, lab, par in rows:
    s.box(MX, y, MW, BH, lab, "#FFFFFF", NEU, INK, 19, sw=2)
    s.box(4, y + 5, 140, BH - 10, par, NEU_F, NEU, INK, 17, sw=1.6)
    s.line(146, y + BH / 2, MX - 2, y + BH / 2, INK, 2, "arr")
for (y0, _, _), (y1, _, _) in zip(rows, rows[1:]):
    s.line(MX + MW / 2 - 50, y0 + BH + 2, MX + MW / 2 - 50, y1 - 2, INK, 2.2, "arr")
# outputs: indices from top-K; gate values from s over the selected experts
s.box(MX, 258, 136, 70, ["expert", "indices"], ACC_F, ACC, ACC, 19, weight="bold", sw=2.4)
s.box(MX + 154, 258, 136, 70, ["gate", "values"], "#FFFFFF", NEU, INK, 19, sw=2)
s.line(MX + 68, 218, MX + 68, 256, ACC, 2.4, "arrAcc")
s.path(f"M{MX + MW - 20},{136} L{MX + MW - 20},{148} L{W - 6},{148} L{W - 6},{240} L{MX + 222},{240} L{MX + 222},{256}",
       INK, 2, "arr")
s.text(4, 300, "which experts", 17, ACC, anchor="start", weight="bold")
s.text(4, 320, "each token visits", 17, ACC, anchor="start", weight="bold")
s.save(os.path.join(HERE, "fig_route.svg"))
