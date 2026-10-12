"""fig_batchinv: what tests/test_mega_mhc.py checks for batch invariance (duplicated tokens -> identical rows)."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED, GOOD

W, H = 450, 300
s = SVG(W, H, "batch invariance check")
rows = ["0", "1", "63", "64", "…", "511", "512", "T−1"]
dup = {"0", "63", "64", "511", "512", "T−1"}
y0, rh = 40, 30
s.text(60, 26, "input rows", 16, MUTED, style="italic")
s.text(386, 26, "output rows", 16, MUTED, style="italic")
for i, r in enumerate(rows):
    y = y0 + i * rh
    if r == "…":
        s.text(60, y + 20, "…", 18, INK)
        s.text(386, y + 20, "…", 18, INK)
        continue
    hl = r in dup
    for x in (20, 346):
        s.rect(x, y + 3, 80, rh - 6, ACC if hl else "#FFFFFF", ACC if hl else NEU, 1.4, 2)
    s.text(12, y + 21, r, 15, INK, anchor="end") if False else None
    s.text(60, y + 20, r, 15, "#FFFFFF" if hl else INK, weight="bold" if hl else "normal")
s.box(166, 112, 116, 64, ["Mega mHC", "one call"], fill=ACC_F, stroke=ACC, sizes=[18, 15],
      weights=["bold", "normal"], colors=[ACC, INK])
s.line(100, 144, 165, 144, INK, 2.2, "arr")
s.line(282, 144, 345, 144, INK, 2.2, "arr")
s.text(225, 222, "one token placed", 16, ACC, weight="bold")
s.text(225, 242, "at 6 positions", 16, ACC, weight="bold")
s.text(225, 276, "→ bitwise identical rows", 17, GOOD, weight="bold")
s.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_batchinv.svg"))
