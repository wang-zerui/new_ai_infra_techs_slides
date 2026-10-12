"""fig_warmup: why the overlap schedule runs one extra warmup micro-batch.

Plain 1F1B on a stage whose step pairs F_k with B_k (e.g. the last stage): the pair is dependent.
With one extra warmup forward, every step pairs F_{k+1} with B_k (independent); only the first F
and the last B stay exposed (combined_1f1b.py; schedules.get_pp_rank_microbatches).
"""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 280, "fig_warmup: plain 1F1B pairs are dependent; +1 warmup micro-batch decouples them")
X0, FW, BW = 166, 62, 108
Y1, Y2, H = 70, 168, 46

s.text(152, Y1 + H / 2 + 6.3, "plain 1F1B", 18, INK, anchor="end", weight="bold")
s.text(152, Y2 + H / 2 + 6.3, "+1 warmup", 18, ACC, anchor="end", weight="bold")

x = X0
for k in range(4):
    fx = x
    s.box(x, Y1, FW, H, f"F{k}", NEU_F, NEU); x += FW
    bx = x
    s.box(x, Y1, BW, H, f"B{k}", NEU_F, NEU); x += BW
    # dependency F_k -> B_k (same micro-batch)
    x1, x2 = fx + FW / 2, bx + BW / 2
    s.path(f"M{x1:.1f},{Y1 - 2} C{x1:.1f},{Y1 - 34} {x2:.1f},{Y1 - 34} {x2:.1f},{Y1 - 3}", PRB, 2.2, "arrPrb")
s.add(f'<text x="{X0}" y="24" font-size="17" font-weight="bold" fill="{PRB}">B<tspan dy="4" font-size="15">k</tspan>'
      f'<tspan dy="-4"> needs the output of F</tspan><tspan dy="4" font-size="15">k</tspan>'
      f'<tspan dy="-4"> of the same micro-batch</tspan></text>')

x = X0
s.box(x, Y2, FW, H, "F0", PRB_F, PRB, color=PRB); s.text(x + FW / 2, Y2 + H + 26, "exposed", 17, PRB, weight="bold")
x += FW + 8
for k in range(3):
    w = 136
    s.box(x, Y2, w, H, f"F{k + 1} + B{k}", ACC_F, ACC, color=ACC, weight="bold")
    x += w + 8
s.box(x, Y2, BW, H, "B3", PRB_F, PRB, color=PRB); s.text(x + BW / 2, Y2 + H + 26, "exposed", 17, PRB, weight="bold")

s.line(X0 - 4, 252, 862, 252, INK, 2, "arr")
s.text(858, 276, "time", 18, INK, anchor="end")
s.save(os.path.join(HERE, "fig_warmup.svg"))
