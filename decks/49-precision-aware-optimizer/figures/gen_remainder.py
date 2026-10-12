"""fig_remainder: FP32 master weight = BF16 weight (high 16 bits) + int16 remainder (low 16 bits).
TE adam.cu AdamFunctorMasterParamRemainder: undo the round-up if the remainder is negative,
rejoin, update in FP32, split again with round-to-nearest."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 262, "fig_remainder: 32-bit layout and the per-step cycle")

# ---------------- bit layout
X0, BIT, Y, H = 150, 22.0, 70, 46
segs = [(1, "s", NEU_F, NEU), (8, "exponent 8", NEU_F, NEU), (7, "mantissa 7", NEU_F, NEU),
        (16, "low mantissa 16", ACC_F, ACC)]
x = X0
for n, lab, f, st in segs:
    w = n * BIT
    s.rect(x, Y, w, H, f, st, 2.2, 0)
    s.text(x + w / 2, Y + H / 2 + 6, lab, 16, ACC if st == ACC else INK, weight="bold" if st == ACC else "normal")
    x += w
s.text(X0 - 12, Y + H / 2 + 6, "FP32 master", 18, INK, anchor="end", weight="bold")
xm = X0 + 16 * BIT


def brace_top(x1, x2, y, lab, col):
    s.path(f"M{x1 + 2},{y} q0,-8 8,-8 L{(x1 + x2) / 2 - 8},{y - 8} q8,0 8,-8 q0,8 8,8 L{x2 - 10},{y - 8} "
           f"q8,0 8,8", col, 1.8)
    s.text((x1 + x2) / 2, y - 24, lab, 17, col, weight="bold")


brace_top(X0, xm, Y - 4, "BF16 weight for the GEMMs", INK)
brace_top(xm, X0 + 32 * BIT, Y - 4, "int16 remainder in optimizer state", ACC)

# ---------------- per-step cycle
steps = [("stored", "BF16 w + int16 r"), ("rejoin", "undo round-up if r < 0"),
         ("Adam step", "on the exact FP32 value"), ("split", "w rounded to nearest")]
BW, BH, Y2 = 192, 58, 168
for i, (t, sub) in enumerate(steps):
    bx = 8 + i * (BW + 26)
    new = i in (1, 3)
    s.rect(bx, Y2, BW, BH, ACC_F if new else "#FFFFFF", ACC if new else NEU, 2, 4)
    s.text(bx + BW / 2, Y2 + 24, t, 17, ACC if new else INK, weight="bold")
    s.text(bx + BW / 2, Y2 + 46, sub, 15, INK)
    if i:
        s.line(bx - 24, Y2 + BH / 2, bx - 2, Y2 + BH / 2, INK, 2, "arr")
s.path(f"M{8 + 3 * (BW + 26) + BW / 2},{Y2 + BH + 2} L{8 + 3 * (BW + 26) + BW / 2},{Y2 + BH + 18} "
       f"L{8 + BW / 2},{Y2 + BH + 18} L{8 + BW / 2},{Y2 + BH + 4}", MUTED, 1.8, "arrGray", dash="5,4")
s.save(os.path.join(HERE, "fig_remainder.svg"))
