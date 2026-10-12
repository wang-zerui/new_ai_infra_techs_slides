"""Depth-mixing matrices M (AttnRes §6.2): input to layer l = sum_i M[l, i] v_i, v_0 = embedding.
(a) standard residual: all-ones lower-triangular (row sums grow with l).
(b) AttnRes: softmax weights from a learned per-layer query, each row sums to 1. Toy alpha values (illustrative)."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
I = "italic"
N = 6
ALPHA = [[1.00], [.30, .70], [.15, .15, .70], [.40, .05, .05, .50], [.10, .55, .05, .05, .25],
         [.20, .05, .05, .05, .15, .50]]


def shade(a):
    if a >= .6:
        return "#2E75B6", "#FFFFFF"
    if a >= .3:
        return "#9DC3E6", INK
    if a >= .1:
        return "#DEEBF7", INK
    return "#F4F8FC", MUTED


def grid(s, x0, y0, cell, mode, rowsum=False, title=None):
    if title:
        s.text(x0 + N * cell / 2, y0 - 34, title, 18, INK, weight="bold")
    # column headers
    for i in range(N):
        lab = "emb" if i == 0 else str(i)
        s.text(x0 + i * cell + cell / 2, y0 - 8, lab, 15, MUTED)
    for l in range(N):
        s.text(x0 - 8, y0 + l * cell + cell / 2 + 5.5, f"layer {l + 1}", 15, MUTED, anchor="end")
        for i in range(N):
            x, y = x0 + i * cell, y0 + l * cell
            if i > l:
                s.rect(x, y, cell, cell, "#FFFFFF", "#E0E0E0", 1, 0)
                continue
            if mode == "ones":
                s.rect(x, y, cell, cell, NEU_F, NEU, 1.2, 0)
                s.text(x + cell / 2, y + cell / 2 + 6, "1", 17, INK, weight="bold")
            else:
                a = ALPHA[l][i]
                f, tc = shade(a)
                s.rect(x, y, cell, cell, f, ACC, 1.2, 0)
                t = "1" if a == 1 else f"{a:.2f}".lstrip("0")
                s.text(x + cell / 2, y + cell / 2 + 5.5, t, 15, tc, weight="bold")
        if rowsum:
            col = PRB if mode == "ones" else ACC
            val = str(l + 1) if mode == "ones" else "1"
            s.text(x0 + N * cell + 26, y0 + l * cell + cell / 2 + 6, val, 18, col, weight="bold")
    if rowsum:
        s.text(x0 + N * cell + 26, y0 - 8, "Σ", 17, PRB if mode == "ones" else ACC, weight="bold")


# fig_matrix_1: standard residual alone (slide 4, right 55%)
s = SVG(420, 300, "fig_matrix_1: standard residual depth-mixing matrix")
grid(s, 110, 52, 36, "ones", rowsum=True)
s.text(110 + 3 * 36, 22, "source", 16, MUTED, style="italic")
s.rich(110, 296, [("row sums grow 1, 2, …, L", PRB, "bold")], 18, anchor="start")
s.save(os.path.join(HERE, "fig_matrix_1.svg"))

# fig_matrix_2: (a) vs (b) (insight slide, full width)
s = SVG(870, 300, "fig_matrix_2: residual = linear attention with weights 1; AttnRes = softmax attention")
grid(s, 110, 60, 34, "ones", rowsum=True, title="(a) standard residual")
grid(s, 560, 60, 34, "soft", rowsum=True, title="(b) AttnRes")
s.text(110 + 3 * 34, 290, "fixed 1s: depth-wise linear attention", 17, PRB, weight="bold")
s.text(560 + 3 * 34, 290, "softmax over sources, learned query", 17, ACC, weight="bold")
s.save(os.path.join(HERE, "fig_matrix_2.svg"))
