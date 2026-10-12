"""Insight: entries per layer for a 2^20-token context, bars drawn to linear scale."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svgkit import SVG, ACC, ACC_FILL, GRAY, GRAY_FILL, INK, RED, RED_FILL

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 220
s = SVG(W, H)
X0, FULL = 250, 616
rows = [("Raw tokens", "1,048,576 per layer", 2 ** 20, RED_FILL, RED, INK),
        ("CSA entries", "262,144 per layer", 2 ** 18, GRAY_FILL, GRAY, INK),
        ("HCA entries", "8,192 per layer", 2 ** 13, ACC, ACC, ACC)]
y = 10
for name, sub, n, fill, stroke, col in rows:
    s.text(0, y + 22, name, 20, "bold", col, "start")
    s.text(0, y + 46, sub, 18, "normal", col, "start")
    w = FULL * n / 2 ** 20
    s.rect(X0, y + 6, max(w, 5), 40, fill, stroke, 2, 2)
    if n == 2 ** 13:
        s.text(X0 + 20, y + 33, "read in full · no selector", 18, "bold", ACC, "start")
    y += 72
s.text(X0, 216, "bars to linear scale", 16, "normal", GRAY, "start")
s.save(os.path.join(HERE, "fig_scale.svg"))
