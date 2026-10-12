"""fig_where: QK-Norm acts inside the forward pass; QK-Clip (Kimi K2) acts in the optimizer step and needs the
per-head max logit S_max from the forward pass (arXiv 2507.20534 Sec. 2.1). QK-Clip deep dive: deck 65.
"""
import os
from svgkit import SVG, rich, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 262
s = SVG(W, H, "fig_where: forward-pass op vs optimizer hook")
L1, L2, BH = (24, 92), (176, 244), 68


def row(y, items):
    xs = []
    x = 124
    for w, lab, fill, st, col, wt, dash in items:
        s.box(x, y[0], w, BH, lab, fill, st, col, size=18, weight=wt, dash=dash, sw=2.4 if fill == ACC_F else 2)
        xs.append((x, w))
        x += w + 28
    for (x0, w0), (x1, _) in zip(xs, xs[1:]):
        s.line(x0 + w0 + 2, (y[0] + y[1]) / 2, x1 - 2, (y[0] + y[1]) / 2, INK, 2.2, "arr")
    return xs


s.text(0, 64, "forward", 20, INK, anchor="start", weight="bold")
s.text(0, 216, "optimizer", 20, INK, anchor="start", weight="bold")
s.line(0, 134, W, 134, "#BFBFBF", 1.2, dash="4,4")
f = row(L1, [
    (110, ["q, k", "proj"], NEU_F, NEU, INK, "normal", None),
    (124, "QK-Norm", ACC_F, ACC, ACC, "bold", None),
    (90, "RoPE", NEU_F, NEU, INK, "normal", None),
    (160, ["attention", "kernel"], NEU_F, NEU, INK, "normal", None),
    (110, "loss", NEU_F, NEU, INK, "normal", None),
])
s.badge(f[1][0] + 2, L1[0] + 2, 3, r=12, size=16)
o = row(L2, [
    (110, "gradients", NEU_F, NEU, INK, "normal", None),
    (124, ["Muon", "update"], NEU_F, NEU, INK, "normal", None),
    (220, ["QK-Clip: rescale", "Wq and Wk"], "#FFFFFF", NEU, INK, "normal", "6,4"),
    (110, ["next", "weights"], NEU_F, NEU, INK, "normal", None),
])
# S_max signal from the attention kernel to QK-Clip
ax = f[3][0] + f[3][1] / 2
s.line(ax, L1[1] + 2, ax, L2[0] - 2, MUTED, 2, "arrGray", dash="6,4")
s.text(ax + 8, 158, "per-head max logit", 17, MUTED, anchor="start", style="italic")
s.text(o[2][0] + o[2][1] / 2, L2[1] + 16, "deck 65", 15, MUTED, style="italic")
s.save(os.path.join(HERE, "fig_where.svg"))
