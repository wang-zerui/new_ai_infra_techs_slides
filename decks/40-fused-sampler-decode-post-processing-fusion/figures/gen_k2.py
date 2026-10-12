"""fig_k2: K2 stage2_kernel on one row. Left: merged, sorted candidates with softmax over the top-k survivors
(AFTER_TOPK); top-p keeps tokens whose preceding mass < p (here the first five). Right: Gumbel-max over the
survivors, key = log p + g, argmax wins. Bar heights / keys are schematic, not data."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 298
s = SVG(W, H, "fig_k2")

# ---- left: sorted candidates
s.text(8, 22, "sorted candidates of one row", 18, INK, anchor="start", weight="bold")
p = [0.38, 0.22, 0.14, 0.10, 0.07, 0.04, 0.025, 0.015, 0.01, 0.008, 0.006, 0.005]
BASE, SCALE, BW, BG, X0 = 236, 420, 28, 9, 20
for i, v in enumerate(p):
    x = X0 + i * (BW + BG)
    h = max(4, v * SCALE)
    if i < 5:
        fill, stroke = ACC, ACC
    elif i < 9:
        fill, stroke = NEU_F, NEU
    else:
        fill, stroke = "url(#hatchGray)", NEU
    s.rect(x, BASE - h, BW, h, fill, stroke, 1.5, 1)
    s.text(x + BW / 2, BASE + 18, f"c{i + 1}", 15, MUTED)
s.line(X0 - 6, BASE, X0 + 12 * (BW + BG), BASE, INK, 1.5)
# cut lines
xp = X0 + 5 * (BW + BG) - BG / 2
xk = X0 + 9 * (BW + BG) - BG / 2
s.line(xp, 52, xp, BASE + 4, ACC, 2, dash="6,4")
s.text(xp + 6, 64, "top-p cut", 16, ACC, anchor="start", weight="bold")
s.line(xk, 104, xk, BASE + 4, MUTED, 2, dash="6,4")
s.text(xk + 6, 116, "top-k cut", 16, MUTED, anchor="start", weight="bold")
s.text(X0 + 1 * (BW + BG), 288, "softmax over the top-k · prefix sum by warp scan", 16, INK, anchor="start")

# arrow between panels
s.line(470, 150, 520, 150, INK, 2.5, "arr")

# ---- right: Gumbel-max
RX = 536
s.text(RX, 22, "Gumbel-max over survivors", 18, INK, anchor="start", weight="bold")
keys = [196, 150, 232, 110, 172]   # schematic
for i, k in enumerate(keys):
    y = 44 + i * 40
    win = i == 2
    s.text(RX + 22, y + 22, f"c{i + 1}", 16, MUTED, anchor="end")
    s.rect(RX + 30, y + 4, k, 28, ACC if win else ACC_F, ACC, 1.5, 2)
    s.text(RX + 40, y + 24, "log p + g", 16, "#FFFFFF" if win else ACC, anchor="start")
s.text(RX + 30 + keys[2] + 8, 44 + 2 * 40 + 24, "argmax", 17, ACC, anchor="start", weight="bold")
s.text(RX, 288, "g = Gumbel noise per candidate", 16, MUTED, anchor="start", style="italic")
s.save(os.path.join(HERE, "fig_k2.svg"))
