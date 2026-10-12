"""Insight figure: DSA scores every token; CSA scores every 4-token block and keeps recent tokens raw."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svgkit import SVG, ACC, ACC_FILL, GRAY, GRAY_FILL, INK, RED, RED_FILL

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 252
s = SVG(W, H)
X0, CW, P = 214, 36, 41          # 16 token slots: 214 .. 865
ROW1, ROW2 = 56, 162             # cell tops

# row 1: DSA (token granularity)
s.text(0, ROW1 + 6, "DSA (V3.2)", 20, "bold", INK, "start")
s.text(0, ROW1 + 32, "scores 16 of 16", 18, "normal", RED, "start")
for t in range(16):
    x = X0 + t * P
    s.rect(x, ROW1, CW, 36, GRAY_FILL, GRAY, 2, 3)
    s.arrow(x + CW / 2, ROW1 - 2, x + CW / 2, ROW1 - 34, RED, 2.2)

# row 2: CSA (block granularity + raw window)
s.text(0, ROW2 + 6, "CSA (V4)", 20, "bold", ACC, "start")
s.text(0, ROW2 + 32, "scores 4 blocks", 18, "normal", ACC, "start")
for b in range(4):
    x = X0 + b * 4 * P
    w = 4 * P - (P - CW)
    s.rect(x, ROW2, w, 36, ACC_FILL, ACC, 2.5, 4)
    s.text(x + w / 2, ROW2 + 25, "4-token block", 18, "bold", ACC)
    s.arrow(x + w / 2, ROW2 - 2, x + w / 2, ROW2 - 34, ACC, 2.5)
for t in range(8, 16):
    x = X0 + t * P
    s.rect(x, ROW2 + 48, CW, 26, "#fff", ACC, 2, 3)
s.text(X0 + 8 * P - 12, ROW2 + 68, "+ recent tokens kept raw", 18, "normal", INK, "end")
s.arrow(0, 112, 870, 112, "#BFBFBF", 1.2, dash="5,4", head=False)
s.save(os.path.join(HERE, "fig_insight.svg"))
