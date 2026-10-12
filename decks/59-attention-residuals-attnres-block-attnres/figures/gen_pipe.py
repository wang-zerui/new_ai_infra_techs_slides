"""fig_pipe: blocks sent per pipeline transition, interleaved PP with P = 4 ranks and V = 2 virtual stages, one new
block per chunk (N_p = 1). Naive: chunk j receives all j accumulated blocks (AR Eq. 7). Cached: the first virtual
stage is unchanged, later transitions send only the ~P new blocks (AR Eq. 8, Fig. 3). Schematic."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 336, "fig_pipe: naive vs cached cross-stage block transfer")
P, V = 4, 2
X0, BW, GAP, BH = 122, 62, 34, 36
ranks = [f"R{k % P}" for k in range(P * V)]
xs = [X0 + k * (BW + GAP) for k in range(P * V)]


def row(yb, counts, fill, stroke, title, sub, col):
    s.text(8, yb + 16, title, 19, col, anchor="start", weight="bold")
    s.text(8, yb + 38, sub, 15, MUTED, anchor="start")
    for k, x in enumerate(xs):
        s.box(x, yb, BW, BH, ranks[k], NEU_F, NEU, size=17)
    for j in range(1, P * V):
        gx = xs[j] - GAP / 2
        s.line(xs[j - 1] + BW + 2, yb + BH / 2, xs[j] - 2, yb + BH / 2, INK, 1.8, "arr")
        for c in range(counts[j - 1]):
            s.rect(gx - 7, yb - 8 - (c + 1) * 13, 14, 11, fill, stroke, 1.4, 1)


naive = list(range(1, P * V))                       # 1..7
cached = [j if j < P else P for j in range(1, P * V)]  # 1,2,3,4,4,4,4
row(126, naive, PRB_F, PRB, "naive", "full history", PRB)
row(258, cached, ACC_F, ACC, "cached", "new only", ACC)
# virtual-stage split
xm = (xs[P - 1] + BW + xs[P]) / 2
for a, b, lab in ((xs[0], xs[P - 1] + BW, "virtual stage 1"), (xs[P], xs[-1] + BW, "virtual stage 2")):
    s.path(f"M {a} 302 V 308 H {b} V 302", MUTED, 1.4)
    s.text((a + b) / 2, 330, lab, 17, MUTED, style="italic")
s.text(xs[-1] + BW, 16, "chip = one block sent", 15, MUTED, anchor="end", style="italic")
s.save(os.path.join(HERE, "fig_pipe.svg"))
