"""Mechanism 1: heavy, non-overlapping pooling. Each 128-token block -> one 512-d entry via a per-channel
softmax over its 128 token projections plus a learned positional bias (V4 Eq. 20-23)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svgkit import SVG, ACC, ACC_FILL, GRAY, GRAY_FILL, INK

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 286
s = SVG(W, H)
CW, GAP, TY = 26, 5, 186


def block(x0, first, idx):
    xs = []
    x = x0
    for j in range(5):
        s.rect(x, TY, CW, 40, GRAY_FILL, GRAY, 2, 3); xs.append(x + CW / 2); x += CW + GAP
    s.text(x + 14, TY + 27, "…", 22, "bold", GRAY); x += 34
    for j in range(5):
        s.rect(x, TY, CW, 40, GRAY_FILL, GRAY, 2, 3); xs.append(x + CW / 2); x += CW + GAP
    x1 = x - GAP
    cx = (x0 + x1) / 2
    s.text(cx, TY + 66, f"tokens {first} … {first + 127}", 18, "normal", INK)
    for xc in xs:
        s.arrow(xc, TY - 2, cx, 128, ACC, 2, head=False)
    s.circle(cx, 112, 16, "#fff", ACC, 2.5)
    s.text(cx, 118.5, "Σ", 20, "bold", ACC)
    s.arrow(cx, 95, cx, 72, ACC, 2.5)
    s.rect(cx - 90, 14, 180, 56, ACC_FILL, ACC, 2.5, 6)
    s.text(cx, 38, f"Entry {idx}", 18, "bold", INK)
    s.text(cx, 60, "one 512-d vector", 16, "normal", INK)
    return x1


x_end = block(16, 0, 0)
block(x_end + 70, 128, 1)
s.arrow(x_end + 35, TY - 6, x_end + 35, TY + 46, "#BFBFBF", 1.5, dash="5,4", head=False)
s.text(x_end + 35, TY + 72, "no overlap", 16, "bold", GRAY)
s.circle(732, 30, 14, "#fff", ACC, 2.5)
s.text(732, 36, "Σ", 18, "bold", ACC)
s.text(754, 36, "softmax over", 18, "normal", INK, "start")
s.text(754, 58, "128 weights", 18, "normal", INK, "start")
s.text(754, 80, "+ bias", 18, "normal", INK, "start")
s.save(os.path.join(HERE, "fig_pool.svg"))
