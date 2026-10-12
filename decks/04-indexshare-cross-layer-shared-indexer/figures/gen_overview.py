"""Slide 7 (mechanism 1): F layer runs the indexer and fills the top-k cache; the next S layers reuse it."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 276
s = SVG(W, H, "IndexCache / IndexShare: F and S layers")
rows = [("layer ℓ+3", "S"), ("layer ℓ+2", "S"), ("layer ℓ+1", "S"), ("layer ℓ", "F")]
y0, rh, gap = 6, 48, 16
XI, WI = 128, 160      # indexer slot
XT, WT = 314, 96       # top-k slot
XC, WC = 446, 150      # cache column
XM, WM = 690, 170      # sparse MLA
bus_x = XC + WC / 2
f_y = y0 + 3 * (rh + gap)
for i, (name, role) in enumerate(rows):
    y = y0 + i * (rh + gap)
    cy = y + rh / 2
    s.text(0, cy - 3, name, 17, INK, anchor="start")
    s.text(0, cy + 18, "Full" if role == "F" else "Shared", 16, ACC if role == "F" else MUTED, anchor="start",
           weight="bold" if role == "F" else "normal")
    s.box(XM, y, WM, rh, "Sparse MLA", NEU_F, NEU, INK, 18, "bold", 2, 5)
    if i < len(rows) - 1:
        s.arrow(XM + WM / 2, y + rh + gap - 2, XM + WM / 2, y + rh + 2)
    if role == "F":
        s.box(XI, y, WI, rh, "Indexer", ACC_F, ACC, INK, 18, "bold", 2.5, 5)
        s.box(XT, y, WT, rh, "Top-k", ACC_F, ACC, INK, 18, "bold", 2.5, 5)
        s.box(XC, y, WC, rh, ["write", "top-k cache"], "#FFFFFF", ACC, ACC, 17, "bold", 2.5, 5)
        s.arrow(XI + WI + 2, cy, XT - 3, cy)
        s.arrow(XT + WT + 2, cy, XC - 3, cy)
        s.arrow(XC + WC + 2, cy, XM - 3, cy, ACC, 2.4)
    else:
        s.box(XI, y, WI + WT + (XT - XI - WI), rh, "no indexer · no top-k", "#FFFFFF", "#A6A6A6", MUTED, 17,
              "normal", 1.6, 5, dash="5,4")
        s.line(bus_x, cy, XM - 3, cy, ACC, 2.4, "arrAcc")
        s.text((bus_x + XM) / 2, cy - 7, "reuse", 16, ACC, weight="bold")
# vertical bus from the cache up through the S rows
top_cy = y0 + rh / 2
s.line(bus_x, f_y - 2, bus_x, top_cy, ACC, 2.4)
s.text(XC + WC / 2, f_y + rh + 20, "overwritten at the next F layer", 16, MUTED, style="italic")
s.save(os.path.join(HERE, "fig_overview.svg"))
