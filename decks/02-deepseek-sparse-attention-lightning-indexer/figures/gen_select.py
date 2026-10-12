"""Slide 9 (mechanism 2): top-k turns each causal row into k reads; MQA mode lets one gathered latent serve all heads.
Selection pattern is illustrative (k = 4 per row here; DSA uses k = 2,048)."""
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 290
s = SVG(W, H, "top-k sparse MLA in MQA mode")
SCORED = "#F4F8FC"

# ---- left: causal matrix, k = 4 selected per row
n, c, g, K = 16, 13, 2, 4
x0, y0 = 58, 28
random.seed(11)
rows = []
for r in range(n):
    cand = list(range(r + 1))
    sel = set(cand) if len(cand) <= K else set(random.sample(cand[:-1], K - 1)) | {r}
    rows.append(sel)
    for k in range(r + 1):
        on = k in sel
        s.rect(x0 + k * (c + g), y0 + r * (c + g), c, c, ACC if on else SCORED, ACC if on else "#9DB7D5", 1.1, 1)
last_y = y0 + (n - 1) * (c + g)
s.rect(x0 - 3, last_y - 3, n * (c + g) + 4, c + 6, "none", ACC, 2.6, 2)
s.text(x0 + n * (c + g) / 2, 20, "cached keys 1 … t", 18, INK)
s.add(f'<text x="{x0 - 14}" y="{y0 + n * (c + g) / 2}" text-anchor="middle" font-size="18" fill="{INK}" '
      f'transform="rotate(-90 {x0 - 14} {y0 + n * (c + g) / 2})">query positions</text>')
s.rect(206, 62 - 12, 13, 13, ACC, ACC, 1, 1); s.text(225, 62, "read", 17, INK, anchor="start")
s.rect(206, 88 - 12, 13, 13, SCORED, "#9DB7D5", 1, 1); s.text(225, 88, "scored only", 17, INK, anchor="start")

# ---- right: one row expanded
cells_x0, cw, cg, cy = 420, 26, 7, 228
sel_last = sorted(rows[-1])
xs = [cells_x0 + i * (cw + cg) for i in range(11)]
xs.append(xs[-1] + cw + 46)
idx = [0, 2, 5, 9, 11]          # which drawn cells are selected (last = the newest token)
gx, gy, gw, gh = 500, 134, 250, 50
for i, x in enumerate(xs):
    on = i in idx
    s.rect(x, cy, cw, 40, ACC if on else "#FFFFFF", ACC if on else NEU, 2, 3)
    if on:
        s.line(x + cw / 2, cy - 2, gx + 20 + idx.index(i) * 52, gy + gh + 3, ACC, 2, "arrAcc")
s.text(xs[10] + cw + 23, cy + 27, "…", 22, NEU, weight="bold")
s.text((xs[0] + xs[-1] + cw) / 2, cy + 60, "MLA latent cache row · 576-d entries", 18, INK)
s.rect(gx, gy, gw, gh, ACC_F, ACC, 2.5, 6)
s.text(gx + gw / 2, gy + 21, "gathered latents", 19, INK, weight="bold")
s.text(gx + gw / 2, gy + 42, "k × 576", 17, INK)
# stacked query heads
for d in (8, 4, 0):
    s.rect(gx + d, 14 - d + 8, gw, 50, "#FFFFFF" if d else NEU_F, NEU, 2, 6)
s.text(gx + gw / 2, 46, "128 query heads", 19, INK, weight="bold")
s.text(gx + gw / 2, 66, "of token t", 17, INK)
s.arrow(gx + gw / 2, gy - 2, gx + gw / 2, 78, ACC, 2.6)
s.text(gx + gw / 2 + 14, 104, "each latent read once", 17, ACC, anchor="start", weight="bold")
s.text(gx + gw / 2 + 14, 124, "serves all 128 heads", 17, ACC, anchor="start", weight="bold")
s.path(f"M{x0 + n * (c + g) + 2},{last_y + c / 2} C 380,{last_y + c / 2} 380,{cy + 20} {cells_x0 - 4},{cy + 20}",
       MUTED, 1.8, "arrGray", dash="5,4")
s.save(os.path.join(HERE, "fig_select.svg"))
