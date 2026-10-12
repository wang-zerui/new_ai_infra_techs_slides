"""fig_knapsack: V4 dense-parameter buckets (§3.4.1). Whole matrices packed onto the ranks of a capped ZeRO group
(knapsack, <= 5 matrices per rank), each bucket padded to the largest; extra DP groups repeat the same buckets and
recompute Newton-Schulz. Matrix sizes are illustrative."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 290
s = SVG(W, H, "fig_knapsack")
U, X0, LH = 36, 50, 42
buckets = [[("M1", 9), ("M8", 3)], [("M2", 7), ("M4", 5)], [("M3", 6), ("M9", 3), ("M10", 2)],
           [("M5", 5), ("M6", 4), ("M7", 3)]]
mx = max(sum(n for _, n in b) for b in buckets)
lanes = [42 + i * 52 for i in range(4)]
s.text(4, 24, "whole matrices per rank (illustrative sizes)", 18, MUTED, anchor="start", style="italic")
for i, (b, y) in enumerate(zip(buckets, lanes)):
    s.text(4, y + LH / 2 + 6.5, f"R{i}", 19, INK, anchor="start", weight="bold")
    x = X0
    for name, n in b:
        s.box(x, y, n * U, LH, name, ACC_F, ACC, color=ACC, weight="bold", size=18, sw=1.8, rx=2)
        x += n * U
    tot = sum(n for _, n in b)
    if tot < mx:
        s.add(f'<rect x="{x}" y="{y}" width="{(mx - tot) * U}" height="{LH}" fill="url(#hatchPrb)" '
              f'stroke="{PRB}" stroke-width="1.6"/>')
# bracket for the ZeRO group
bx = X0 + mx * U + 12
s.path(f"M{bx},{lanes[0]} h10 V{lanes[-1] + LH} h-10", INK, 2)
s.text(bx + 20, (lanes[0] + lanes[-1] + LH) / 2 - 4, "capped", 18, INK, anchor="start", weight="bold")
s.text(bx + 20, (lanes[0] + lanes[-1] + LH) / 2 + 18, "ZeRO group", 18, INK, anchor="start", weight="bold")
# extra DP groups
gx = 656
for k in range(2, -1, -1):
    s.rect(gx + k * 8, 86 - k * 8, 190, 136, "#FFFFFF" if k else NEU_F, "#A6A6A6", 1.6, 4, dash="6,4" if k else None)
s.text(gx + 95, 128, "extra DP groups", 18, MUTED, weight="bold")
s.text(gx + 95, 154, "same buckets", 18, MUTED)
s.text(gx + 95, 180, "NS recomputed", 18, MUTED)
s.line(bx + 134, 150, gx - 4, 150, MUTED, 2, "arrGray")
# legend
s.add(f'<rect x="{X0}" y="{H - 30}" width="30" height="20" fill="url(#hatchPrb)" stroke="{PRB}" stroke-width="1.4"/>')
s.text(X0 + 40, H - 14, "padding to the largest bucket", 18, INK, anchor="start")
s.save(os.path.join(HERE, "fig_knapsack.svg"))
