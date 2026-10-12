"""Slide 6 (insight): adjacent layers' indexers pick mostly the same tokens; distant layers do not.
Schematic: 28 token positions, k = 8 per layer; the measured adjacent-layer overlap is 0.7-1.0 (IndexCache App. A)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 262
s = SVG(W, H, "cross-layer top-k overlap (schematic)")
LIGHT = "#9DC3E6"
N = 24
base = {1, 4, 7, 10, 13, 16, 19, 22}
rows = [
    ("layer ℓ", base),
    ("layer ℓ+1", (base - {16}) | {15}),
    ("layer ℓ+2", (base - {7}) | {8}),
    ("layer ℓ+3", (base - {16, 22}) | {15, 20}),
]
far = ("distant layer", {0, 3, 7, 9, 12, 15, 18, 21})
x0, cw, cg = 150, 22, 4
y, rh, rg = 8, 30, 12


def row(label, sel, y, muted=False):
    s.text(0, y + 21, label, 18, MUTED if muted else INK, anchor="start")
    for c in range(N):
        x = x0 + c * (cw + cg)
        if c in sel:
            fill = ACC if c in base else LIGHT
            s.rect(x, y, cw, rh, fill, ACC, 1.4, 2)
        else:
            s.rect(x, y, cw, rh, "#FFFFFF", "#C8C8C8", 1.2, 2)


for i, (lab, sel) in enumerate(rows):
    row(lab, sel, y + i * (rh + rg))
yf = y + 4 * (rh + rg) + 20
s.text(x0 + N * (cw + cg) / 2, yf - 10, "⋮", 22, MUTED)
row(far[0], far[1], yf, muted=True)
# legend + bracket
lx = x0 + N * (cw + cg) + 10
s.line(lx, y + 2, lx + 8, y + 2, ACC, 2); s.line(lx + 8, y + 2, lx + 8, y + 4 * (rh + rg) - rg - 2, ACC, 2)
s.line(lx, y + 4 * (rh + rg) - rg - 2, lx + 8, y + 4 * (rh + rg) - rg - 2, ACC, 2)
s.text(lx + 16, y + 74, "mostly", 18, ACC, anchor="start", weight="bold")
s.text(lx + 16, y + 96, "shared", 18, ACC, anchor="start", weight="bold")
ly = H - 10
s.rect(x0, ly - 14, 16, 16, ACC, ACC, 1.2, 2); s.text(x0 + 22, ly, "also picked by layer ℓ", 17, INK, anchor="start")
s.rect(x0 + 250, ly - 14, 16, 16, LIGHT, ACC, 1.2, 2); s.text(x0 + 272, ly, "newly picked", 17, INK, anchor="start")
s.text(x0 + 410, ly, "schematic · k = 8 of 24", 16, MUTED, anchor="start", style="italic")
s.save(os.path.join(HERE, "fig_overlap.svg"))
