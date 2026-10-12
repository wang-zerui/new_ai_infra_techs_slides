"""fig_naive: a small Adam step on a weight stored in BF16 rounds back to w (BF16 spacing s, step < s/2);
the same step on an FP32 master weight lands on a representable value."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 262, "fig_naive: BF16 grid vs FP32 grid around w")

X0, X1 = 190, 840
XW = 560                  # position of w
S = 230                   # BF16 spacing in pt
D = 52                    # the update (< S/2)


def axis(y, label, sub):
    s.text(18, y + 6, label, 19, INK, anchor="start", weight="bold")
    s.text(18, y + 28, sub, 15, MUTED, anchor="start")
    s.line(X0, y, X1, y, NEU, 2)


# ---- BF16 row
Y1 = 64
axis(Y1, "BF16 weight", "8 significant bits")
for k, lab in ((-1, "w − s"), (0, "w"), (1, "w + s")):
    x = XW + k * S
    s.line(x, Y1 - 14, x, Y1 + 14, INK, 2.4)
    s.text(x, Y1 + 36, lab, 17, INK)
s.add(f'<circle cx="{XW}" cy="{Y1}" r="7" fill="{INK}"/>')
s.path(f"M{XW - 4},{Y1 - 22} L{XW - D + 6},{Y1 - 22}", PRB, 2.4, "arrPrb")
s.text(XW - D / 2, Y1 - 30, "−Δ", 17, PRB, weight="bold")
s.add(f'<circle cx="{XW - D}" cy="{Y1}" r="6" fill="#FFFFFF" stroke="{PRB}" stroke-width="2.2"/>')
s.path(f"M{XW - D + 2},{Y1 + 10} Q{XW - D / 2},{Y1 + 30} {XW - 6},{Y1 + 12}", PRB, 2.2, "arrPrb")
s.text(XW - S / 2 - 30, Y1 + 62, "Δ < s/2: rounds back to w", 17, PRB, weight="bold")

# ---- FP32 row
Y2 = 178
axis(Y2, "FP32 master", "24 significant bits")
for k in range(-12, 13):
    x = XW + k * (S / 12)
    if X0 < x < X1:
        s.line(x, Y2 - 8, x, Y2 + 8, MUTED, 1.2)
s.add(f'<circle cx="{XW}" cy="{Y2}" r="7" fill="{INK}"/>')
s.text(XW, Y2 + 32, "w", 17, INK)
xn = XW - (S / 12) * round(D / (S / 12))
s.path(f"M{XW - 4},{Y2 - 22} L{xn + 6},{Y2 - 22}", ACC, 2.4, "arrAcc")
s.text((XW + xn) / 2, Y2 - 30, "−Δ", 17, ACC, weight="bold")
s.add(f'<circle cx="{xn}" cy="{Y2}" r="7" fill="{ACC}"/>')
s.text(xn, Y2 + 32, "w − Δ", 17, ACC, weight="bold")
s.text(XW - S / 2 - 30, Y2 + 62, "the step is kept", 17, ACC, weight="bold")
s.save(os.path.join(HERE, "fig_naive.svg"))
