"""Status quo: a per-token KV cache is a layers x tokens grid; narrower entries do not change the count."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svgkit import SVG, RED, RED_FILL, GRAY, INK

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 474, 226
s = SVG(W, H)
X0, Y0, CW, CH, GX, GY = 96, 50, 24, 22, 5, 8
rows = ["layer 0", "layer 1", "layer 2", None, "layer 60"]
cols = 11
s.text(X0, 22, "tokens 1 … L", 18, "bold", INK, "start")
s.arrow(X0 + 130, 16, X0 + cols * (CW + GX) + 40, 16, INK, 2.2)
y = Y0
for r in rows:
    if r is None:
        s.text(X0 + 5 * (CW + GX), y + 14, "⋮", 22, "bold", GRAY)
        y += 26
        continue
    s.text(X0 - 10, y + 17, r, 18, "normal", INK, "end")
    for c in range(cols):
        s.rect(X0 + c * (CW + GX), y, CW, CH, RED_FILL, RED, 1.8, 2)
    s.text(X0 + cols * (CW + GX) + 6, y + 18, "…", 20, "bold", RED, "start")
    y += CH + GY
s.text(X0, y + 22, "one entry per token per layer", 18, "bold", RED, "start")
s.save(os.path.join(HERE, "fig_before.svg"))
