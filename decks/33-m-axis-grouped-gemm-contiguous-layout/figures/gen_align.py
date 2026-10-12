"""fig_align_1/2: packed segments let one M tile straddle experts; aligning each segment to BLOCK_M = 128 rows
gives every tile exactly one expert (build: step 1 = packed only, step 2 = both)."""
import os
from tilekit import SVG, seg, pad, hlines, outline, mono, INK, MUTED, ACC, PRB, GOOD

HERE = os.path.dirname(os.path.abspath(__file__))
BM, TH, Y0, NT = 128, 46, 44, 4
px = TH / BM


def column(s, x, w, parts):
    """parts: list of (expert or None for pad, rows)."""
    y = Y0
    for e, rows in parts:
        h = rows * px
        if e is None:
            pad(s, x, y, w, h)
        else:
            seg(s, x, y, w, h, e, f"E{e}" if h >= 20 else None, size=17, halo=True)
            if h < 20:
                s.text(x + w / 2, y + h / 2 + 5.5, f"E{e}", 15, INK, weight="bold")
        y += h
    hlines(s, x, 0, w, [Y0 + TH * k for k in range(1, NT)], sw=1.8)
    outline(s, x, Y0, w, TH * NT)
    for k in range(NT + 1):
        mono(s, x - 8, Y0 + TH * k + 5, str(BM * k), 15, MUTED, anchor="end")


def build(step):
    s = SVG(480, 290, f"fig_align_{step}: packed vs aligned contiguous layout")
    XL, XR, W = 70, 318, 128
    s.text(XL + W / 2, 26, "(a) packed", 18, INK, weight="bold")
    column(s, XL, W, [(0, 300), (1, 70), (2, 142)])
    # the straddling tile
    s.rect(XL - 3, Y0 + 2 * TH, W + 6, TH, "none", PRB, 3.2, rx=0)
    s.text(XL + W / 2, Y0 + NT * TH + 24, "tile 2 mixes 3 experts", 16, PRB, weight="bold")
    s.text(XL + W / 2, Y0 + NT * TH + 44, "which weight to load?", 16, PRB)
    if step >= 2:
        s.text(XR + W / 2, 26, "(b) aligned to 128", 18, INK, weight="bold")
        column(s, XR, W, [(0, 300), (None, 84), (1, 70), (None, 58)])
        s.text(XR + W / 2, Y0 + NT * TH + 24, "one expert per tile", 16, ACC, weight="bold")
        s.text(XR + W / 2, Y0 + NT * TH + 44, "zero pad rows (index −1)", 16, ACC)
        s.text((XL + W + XR) / 2 - 16, Y0 + NT * TH / 2 + 10, "⇒", 30, MUTED)
    s.save(os.path.join(HERE, f"fig_align_{step}.svg"))


for st in (1, 2):
    build(st)
