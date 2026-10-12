"""Slide 4 (villain): dense MLA — every query row of the causal matrix reads every cached latent."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 290
s = SVG(W, H, "dense causal attention")

# ---- left panel: causal query x key matrix (16 x 16)
n, c, g = 16, 13, 2
x0, y0 = 58, 28
for r in range(n):
    for k in range(r + 1):
        s.rect(x0 + k * (c + g), y0 + r * (c + g), c, c, PRB_F, PRB, 1.1, 1)
# highlighted last row = one decode step
last_y = y0 + (n - 1) * (c + g)
s.rect(x0 - 3, last_y - 3, n * (c + g) + 4, c + 6, "none", PRB, 2.6, 2)
s.text(x0 + n * (c + g) / 2, 19, "cached keys 1 … t", 18, INK)
s.add(f'<text x="{x0 - 14}" y="{y0 + n * (c + g) / 2}" text-anchor="middle" font-size="18" fill="{INK}" '
      f'transform="rotate(-90 {x0 - 14} {y0 + n * (c + g) / 2})">query positions</text>')
s.text(x0 + n * (c + g) / 2, 284, "prefill: all rows · decode: one row", 17, MUTED, style="italic")

# ---- right panel: one decode step expanded
qx, qy, qw, qh = 440, 14, 280, 58
s.box(qx, qy, qw, qh, ["Query token t", "128 heads"], "#FFFFFF", INK, INK, 19, "bold")
cells_x0, cw, cg, cy = 420, 26, 7, 182
xs = [cells_x0 + i * (cw + cg) for i in range(11)]
xs.append(xs[-1] + cw + 46)
for i, x in enumerate(xs):
    s.rect(x, cy, cw, 44, PRB_F, PRB, 2, 3)
    s.line(qx + qw / 2, qy + qh + 2, x + cw / 2, cy - 2, PRB, 1.6, "arrPrb")
s.text(xs[10] + cw + 23, cy + 29, "…", 22, PRB, weight="bold")
s.text(xs[0] + cw / 2, cy + 64, "1", 17, MUTED)
s.text(xs[-1] + cw / 2, cy + 64, "t", 17, MUTED, style="italic")
s.text((xs[0] + xs[-1] + cw) / 2, cy + 64, "cached latents · 576-d each", 18, INK)
s.text(qx + qw + 14, 38, "QK + PV", 19, PRB, anchor="start", weight="bold")
s.text(qx + qw + 14, 62, "on all t", 19, PRB, anchor="start", weight="bold")
# connector from highlighted row to the expansion
s.path(f"M{x0 + n * (c + g) + 2},{last_y + c / 2} C 380,{last_y + c / 2} 380,{cy + 22} {cells_x0 - 4},{cy + 22}",
       MUTED, 1.8, "arrGray", dash="5,4")
s.save(os.path.join(HERE, "fig_dense.svg"))
