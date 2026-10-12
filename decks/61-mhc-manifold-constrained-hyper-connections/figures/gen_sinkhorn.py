"""fig_sinkhorn: Sinkhorn-Knopp on a made-up 4x4 input ([mHC] eq. 9; [V4] eq. 8): M0 = exp(H~), then
M_t = T_r(T_c(M_{t-1})). Values are computed here; t_max = 20 as in mHC / V4."""
import math, os
from svgkit import SVG, ACC, ACC_F, PRB, NEU, INK, MUTED, GOOD

HERE = os.path.dirname(os.path.abspath(__file__))
Ht = [[1.0, -0.5, 0.2, -1.0], [0.3, 0.8, -0.6, 0.1], [-0.4, 0.2, 0.9, -0.2], [0.0, -0.8, 0.4, 0.7]]
M0 = [[math.exp(v) for v in r] for r in Ht]

def step(M):
    cs = [sum(M[i][j] for i in range(4)) for j in range(4)]
    M = [[M[i][j] / cs[j] for j in range(4)] for i in range(4)]
    return [[v / sum(r) for v in r] for r in M]

M1 = step(M0)
M20 = M0
for _ in range(20):
    M20 = step(M20)

s = SVG(870, 286, "fig_sinkhorn")
CS = 44

def panel(x0, y0, M, title):
    s.text(x0 + 2 * CS, y0 - 14, title, 19, INK, weight="bold")
    mx = max(max(r) for r in M)
    for i in range(4):
        for j in range(4):
            v = M[i][j]
            a = 0.12 + 0.75 * v / mx
            s.add(f'<rect x="{x0 + j * CS}" y="{y0 + i * CS}" width="{CS}" height="{CS}" fill="{ACC}" '
                  f'fill-opacity="{a:.2f}" stroke="#FFFFFF" stroke-width="1.5"/>')
            s.text(x0 + j * CS + CS / 2, y0 + i * CS + CS / 2 + 6, f"{v:.2f}", 16,
                   "#FFFFFF" if a > 0.55 else INK)
    # row sums (right) and column sums (below)
    for i in range(4):
        r = sum(M[i])
        ok = abs(r - 1) < 0.005
        s.text(x0 + 4 * CS + 8, y0 + i * CS + CS / 2 + 6, f"{r:.2f}", 16, GOOD if ok else PRB, anchor="start",
               weight="bold")
    for j in range(4):
        c = sum(M[i][j] for i in range(4))
        ok = abs(c - 1) < 0.005
        s.text(x0 + j * CS + CS / 2, y0 + 4 * CS + 22, f"{c:.2f}", 16, GOOD if ok else PRB, weight="bold")

Y0 = 48
panel(10, Y0, M0, "exp(H̃)  (toy)")
panel(318, Y0, M1, "after 1 iteration")
panel(626, Y0, M20, "after 20 iterations")
for xa in (244, 552):
    s.line(xa, Y0 + 2 * CS, xa + 58, Y0 + 2 * CS, ACC, 2.6, "arrAcc")
s.text(273, Y0 + 2 * CS - 12, "Tc, Tr", 17, ACC, weight="bold")
s.text(581, Y0 + 2 * CS - 12, "×19", 17, ACC, weight="bold")
s.text(10, 280, "right: row sums · below: column sums · red ≠ 1, green = 1", 17, MUTED, anchor="start", style="italic")
s.save(os.path.join(HERE, "fig_sinkhorn.svg"))
