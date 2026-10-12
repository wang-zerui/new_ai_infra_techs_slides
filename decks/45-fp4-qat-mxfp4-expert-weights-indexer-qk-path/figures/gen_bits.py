"""Insight: E2M1 vs E4M3 bit layouts, and 1x32 FP4 sub-blocks (own E8M0 scale) inside a 128x128 FP8 block."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svgkit import SVG, ACC, ACC_FILL, ACC_LIGHT, GRAY, GRAY_FILL, INK

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 282
s = SVG(W, H)
CW = 36


def bits(y, name, fields):
    s.text(0, y + 26, name, 20, "bold", INK, "start")
    x = 112
    for kind, extra in fields:
        fill = {"S": "#fff", "E": ACC_LIGHT if not extra else ACC, "M": GRAY_FILL}[kind]
        s.rect(x, y, CW, 40, fill, ACC if kind == "E" else GRAY, 2, 2)
        s.text(x + CW / 2, y + 27, kind, 18, "bold", "#fff" if extra else INK)
        x += CW
    return x


bits(30, "FP4", [("S", 0), ("E", 0), ("E", 0), ("M", 0)])
s.text(0, 92, "E2M1", 18, "normal", GRAY, "start")
bits(130, "FP8", [("S", 0), ("E", 0), ("E", 0), ("E", 1), ("E", 1), ("M", 0), ("M", 0), ("M", 0)])
s.text(0, 192, "E4M3", 18, "normal", GRAY, "start")
s.path("M220,178 V190 H292 V178", ACC, 2.5, head=False)
s.text(256, 214, "+2 exponent bits", 18, "bold", ACC)
s.text(0, 270, "S sign · E exponent · M mantissa", 16, "normal", GRAY, "start")

# right: 128 x 128 FP8 block of 1 x 32 FP4 sub-blocks (schematic: 8 rows shown)
BX, BY, BW, BH = 466, 24, 236, 236
shades = ["#EAF1F8", "#DEEBF7", "#C5DAEE", "#9DC3E6"]
rows = 8
rh = BH / rows
sw = BW / 4
for r in range(rows):
    for c in range(4):
        s.rect(BX + c * sw, BY + r * rh, sw, rh, shades[(r * 3 + c * 2) % 4], "#7F9FBF", 1, 0)
s.rect(BX, BY, BW, BH, "none", ACC, 3.5, 0)
s.text(BX + BW + 16, BY + 22, "128 × 128", 18, "bold", INK, "start")
s.text(BX + BW + 16, BY + 44, "FP8 block", 18, "normal", INK, "start")
s.text(BX + BW + 16, BY + 66, "one scale", 18, "normal", INK, "start")
s.path(f"M{BX + BW + 10},{BY + BH - 2 * rh + rh / 2} H{BX + BW - sw / 2}", INK, 1.8, head=True)
s.text(BX + BW + 16, BY + 170, "1 × 32 FP4", 18, "bold", INK, "start")
s.text(BX + BW + 16, BY + 192, "sub-blocks", 18, "normal", INK, "start")
s.text(BX + BW + 16, BY + 214, "own E8M0 scale", 18, "normal", INK, "start")
s.text(BX, BY + BH + 20, "shade = sub-block scale", 16, "normal", GRAY, "start")
s.save(os.path.join(HERE, "fig_bits.svg"))
