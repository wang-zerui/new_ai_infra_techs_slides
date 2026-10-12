"""Mechanism 3: heterogeneous KV cache for the interleaved stack (V4 §3.5.1, Fig. 6), V4-Pro layer order.
Right panel bar widths are proportional to bytes per 128-token block (576 B per entry, 64 B per FP4 indexer key)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svgkit import SVG, ACC, ACC_FILL, ACC_LIGHT, GRAY, GRAY_FILL, INK

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 278
s = SVG(W, H)
layers = [("L0", "HCA"), ("L1", "HCA"), ("L2", "CSA"), ("L3", "HCA"), None, ("L60", "CSA")]
RY0, RH, RG = 80, 28, 8

# ---- left: state cache (fixed per request)
s.text(0, 22, "State cache", 20, "bold", INK, "start")
s.text(0, 46, "fixed size per request", 18, "normal", GRAY, "start")
y = RY0
for c in layers:
    if c is None:
        s.text(170, y + 14, "⋮", 20, "bold", GRAY); y += 22; continue
    s.text(0, y + 20, c[0], 18, "normal", INK, "start")
    s.rect(46, y, 170, RH, ACC_FILL, ACC, 2, 3)
    s.text(131, y + 20, "window 128", 16, "bold", ACC)
    s.rect(222, y, 92, RH, "url(#hatchGray)", GRAY, 2, 3)
    s.text(268, y + 20, "tail", 16, "bold", INK)
    y += RH + RG

# ---- right: block cache (one block = lcm(4, 128) = 128 tokens)
BX = 382
s.text(BX, 22, "Block cache", 20, "bold", INK, "start")
s.text(BX, 46, "1 block = 128 tokens · N blocks grow with context", 18, "normal", GRAY, "start")
scale = 248 / (32 * 576)      # px per byte
# stacked copies behind (blocks 1..N)
for off in (12, 6):
    s.rect(BX + 92 + off, RY0 - 6 - off, 372, y - RY0 + 2, "#fff", "#BFBFBF", 1.5, 4)
s.rect(BX + 92, RY0 - 6, 372, y - RY0 + 2, "#fff", GRAY, 1.5, 4)
y = RY0
for c in layers:
    if c is None:
        s.text(BX + 240, y + 14, "⋮", 20, "bold", GRAY); y += 22; continue
    li, k = c
    s.text(BX, y + 20, f"{li} {k}", 18, "bold" if k == "HCA" else "normal", ACC if k == "HCA" else INK, "start")
    x = BX + 104
    if k == "CSA":
        wm, wi = 32 * 576 * scale, 32 * 64 * scale
        s.rect(x, y, wm, RH, GRAY_FILL, GRAY, 2, 2)
        s.text(x + wm / 2, y + 20, "32 main entries", 16, "normal", INK)
        s.rect(x + wm + 4, y, wi, RH, "#fff", GRAY, 2, 2)
        s.text(x + wm + wi + 12, y + 20, "32 idx", 16, "normal", INK, "start")
    else:
        wh = 576 * scale
        s.rect(x, y, wh, RH, ACC, ACC, 2, 1)
        s.text(x + wh + 10, y + 20, "1 entry", 16, "bold", ACC, "start")
    y += RH + RG
s.save(os.path.join(HERE, "fig_layout.svg"))
