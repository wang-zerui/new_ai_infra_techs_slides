"""fig_dual: one expert's dual objective vs its threshold beta = -b (schematic). Slope = target load q minus
current load (K3 App. C, Eq. 27). Sign update = fixed steps (red); QB = one jump to the exact minimizer (accent)."""
import os
from svgkit import SVG, ACC, PRB, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 500, 300
s = SVG(W, H, "fig_dual: convex piecewise-linear dual, sign steps vs QB jump (schematic)")

OX, OY, XE, YT = 34, 236, 488, 14
q = 2
M = [70, 112, 150, 184, 214, 243, 262, 392]       # token margins in px (schematic)


def L(x):                                          # q*x + sum max(0, m - x)
    return q * x + sum(max(0.0, m - x) for m in M)


xs = [OX + 10] + sorted(M) + [XE - 14]
vals = [L(x) for x in xs]
lo, hi = min(vals), max(vals)
Y0, Y1 = 196, 40                                    # pixel range for the curve


def Y(x):
    return Y0 - (L(x) - lo) / (hi - lo) * (Y0 - Y1)


# axes
s.line(OX, OY, XE, OY, INK, 2, "arr")
s.line(OX, OY, OX, YT, INK, 2, "arr")
s.text(XE - 2, OY + 26, "expert threshold β = −b", 18, INK, anchor="end")
s.add(f'<text x="{OX - 12}" y="{(OY + YT) / 2 + 6}" text-anchor="middle" font-size="18" fill="{INK}" '
      f'transform="rotate(-90 {OX - 12} {(OY + YT) / 2 + 6})">dual objective</text>')
# token margins as ticks on the x axis
for m in M:
    s.line(m, OY - 7, m, OY + 1, MUTED, 2)
s.text(OX + 14, OY - 14, "token margins", 16, MUTED, anchor="start", style="italic")

# curve
d = "M " + " L ".join(f"{x:.1f} {Y(x):.1f}" for x in xs)
s.path(d, INK, 2.6)

# minimizer: flat notch between the (q+1)-th and q-th largest margins
xstar = sorted(M, reverse=True)[q]                 # (q+1)-th largest margin
s.line(xstar, Y(xstar) + 4, xstar, OY - 9, ACC, 2, dash="5,4")
s.circle(xstar, Y(xstar), 6, ACC, ACC, 1)

# sign steps: fixed gamma, overshooting around the notch (red arcs above the curve)
gam = 68
pts = [60]
for _ in range(5):
    x = pts[-1]
    slope = q - sum(1 for m in M if m > x)
    pts.append(x + gam if slope < 0 else (x - gam if slope > 0 else x))
for a, b in zip(pts[:-1], pts[1:]):
    ya, yb = Y(a), Y(b)
    cx, cy = (a + b) / 2, min(ya, yb) - 26
    s.path(f"M {a:.1f} {ya - 4:.1f} Q {cx:.1f} {cy:.1f} {b:.1f} {yb - 5:.1f}", PRB, 2.2, "arrPrb")
for x in pts:
    s.circle(x, Y(x), 4, PRB, PRB, 1)
s.text(296, 118, "sign steps: fixed γ", 18, PRB, anchor="start", weight="bold")

# QB: one jump from the start to the minimizer (accent arc, high)
a, b = pts[0], xstar
s.path(f"M {a:.1f} {Y(a) - 8:.1f} C {a + 30:.1f} {YT + 6:.1f} {b - 40:.1f} {YT + 6:.1f} {b:.1f} {Y(b) - 9:.1f}",
       ACC, 2.8, "arrAcc")
s.text(b + 14, 40, "QB: one exact jump", 18, ACC, anchor="start", weight="bold")
s.text(b + 14, 62, "to the (q+1)-th margin", 18, ACC, anchor="start")
s.save(os.path.join(HERE, "fig_dual.svg"))
