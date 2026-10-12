"""fig_timeline: illustrative job whose node count changes between restarts (multiples of 32, as K2 allows)."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, NEU, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(480, 330, "fig_timeline")
X0, X1, Y0, YT = 62, 470, 262, 46
def y(n):
    return Y0 - (Y0 - YT) * n / 96
# grid + y labels
for n in (32, 64, 96):
    s.line(X0, y(n), X1, y(n), "#D9D9D9", 1.2, dash="5,5")
    s.text(X0 - 10, y(n) + 6, str(n), 18, INK, anchor="end")
s.line(X0, Y0, X1, Y0, INK, 2, "arr")
s.line(X0, Y0, X0, YT - 14, INK, 2, "arr")
s.text(X1, Y0 + 24, "time", 17, MUTED, anchor="end", style="italic")
s.text(X0 - 10, YT - 22, "nodes", 18, INK, anchor="start", weight="bold")

segs = [(X0 + 4, 186, 64), (206, 326, 32), (346, 462, 96)]
for xa, xb, n in segs:
    s.line(xa, y(n), xb, y(n), ACC, 4)
# checkpoints
def ckpt(x, n, yy=None):
    yy = y(n) if yy is None else yy
    s.add(f'<rect x="{x - 6:.1f}" y="{yy - 6:.1f}" width="12" height="12" fill="#FFFFFF" stroke="{ACC}" '
          f'stroke-width="2.4" transform="rotate(45 {x:.1f} {yy:.1f})"/>')
for x in (100, 140, 176):
    ckpt(x, 64)
for x in (240, 280, 318):
    ckpt(x, 32)
for x in (380, 420, 455):
    ckpt(x, 96)
# failure
fx, fy = 192, y(64)
s.line(fx - 9, fy - 9, fx + 9, fy + 9, PRB, 3.5)
s.line(fx - 9, fy + 9, fx + 9, fy - 9, PRB, 3.5)
s.text(fx - 4, fy - 20, "node failure", 17, PRB, weight="bold")
# restarts
s.line(196, fy + 12, 206, y(32) - 6, MUTED, 1.6, "arrGray", dash="4,3")
s.line(332, y(32) - 4, 346, y(96) + 6, MUTED, 1.6, "arrGray", dash="4,3")
s.text(266, y(32) + 30, "restart smaller", 17, INK)
s.text(404, y(96) + 30, "restart larger", 17, INK)
# legend
ckpt(76, 0, yy=Y0 + 44)
s.text(92, Y0 + 50, "checkpoint", 17, INK, anchor="start")
s.text(X1, Y0 + 50, "illustrative", 15, MUTED, anchor="end", style="italic")
s.save(os.path.join(HERE, "fig_timeline.svg"))
