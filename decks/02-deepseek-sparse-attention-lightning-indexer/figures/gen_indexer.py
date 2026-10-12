"""Slide 8 (mechanism 1): lightning indexer for one query token t, as tensor shapes (build in 2 steps).

Step 1: q (64 heads x 128) @ K^T (128 x L) -> ReLU logits (64 x L).
Step 2: w_t^T (1 x 64) @ ReLU logits -> scores I_t (1 x L) -> top-k ids.
Cell values are illustrative (seeded), not data.
"""
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 288
COLS, ROWS = 25, 8
LX, LY, CW, CH = 300, 74, 16, 10          # logits block origin and cell size
random.seed(3)
logit = [[random.gauss(0, 1) for _ in range(COLS)] for _ in range(ROWS)]
w = [random.uniform(0.3, 1.0) for _ in range(ROWS)]
score = [sum(w[r] * max(logit[r][c], 0) for r in range(ROWS)) for c in range(COLS)]
top = set(sorted(range(COLS), key=lambda c: -score[c])[:5])


def blend(a, b, t):
    a = [int(a[i:i + 2], 16) for i in (1, 3, 5)]; b = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02X}" for x, y in zip(a, b))


def common(s):
    # keys (cached), K^T: 128 x L
    s.rect(LX, 10, COLS * CW, 48, ACC_F, ACC, 2.2, 3)
    s.rich(LX + COLS * CW / 2, 40, [("Indexer keys K", ""), ("I", "sup"), ("  ·  128 × L  ·  FP8 cache", "")], 18)
    # query block q: 64 x 128
    s.rect(244, LY, 48, ROWS * CH, ACC_F, ACC, 2.2, 3)
    s.rich(232, LY + 30, [("q", "b"), ("I", "sup"), ("t", "sub")], 22, anchor="end")
    s.text(232, LY + 56, "64 × 128", 18, INK, anchor="end")
    # logits after ReLU: 64 x L (8 x 25 illustrative cells)
    mx = max(max(r) for r in logit)
    for r in range(ROWS):
        for c in range(COLS):
            v = max(logit[r][c], 0) / mx
            fill = "#FFFFFF" if v == 0 else blend("#DEEBF7", "#2E75B6", v)
            s.rect(LX + c * CW, LY + r * CH, CW, CH, fill, "#BFBFBF", 0.6, 0)
    s.rect(LX, LY, COLS * CW, ROWS * CH, "none", NEU, 2, 0)
    s.text(LX + COLS * CW + 14, LY + 30, "ReLU logits", 19, INK, anchor="start", weight="bold")
    s.text(LX + COLS * CW + 14, LY + 54, "64 × L", 18, INK, anchor="start")
    s.text(LX + COLS * CW + 14, LY + 76, "zeros = white", 16, MUTED, anchor="start", style="italic")
    s.rich(435, 270, [("I", "b"), ("t,s", "sub"), (" = Σ", ""), ("j", "sub"), ("  w", ""), ("t,j", "sub"),
                      (" · ReLU( q", ""), ("I", "sup"), ("t,j", "sub"), (" · k", ""), ("I", "sup"), ("s", "sub"),
                      (" )", "")], 23)


def step2(s):
    sy = 196
    # w_t^T: 1 x 64, left of the score strip (GEMM layout)
    s.rect(212, sy, 80, 22, ACC_F, ACC, 2.2, 3)
    s.rich(200, sy + 17, [("w", "b"), ("t", "sub")], 22, anchor="end")
    s.text(252, sy + 46, "1 × 64", 17, INK)
    smax = max(score)
    for c in range(COLS):
        sel = c in top
        fill = ACC if sel else blend("#FFFFFF", "#9DC3E6", score[c] / smax)
        s.rect(LX + c * CW, sy, CW, 22, fill, NEU if not sel else ACC, 0.8, 0)
    s.rect(LX, sy, COLS * CW, 22, "none", NEU, 2, 0)
    s.arrow(LX + COLS * CW / 2, LY + ROWS * CH + 3, LX + COLS * CW / 2, sy - 3)
    s.text(LX + COLS * CW / 2 - 10, sy - 14, "weighted sum over 64 heads", 17, INK, anchor="end")
    s.text(LX + COLS * CW / 2, sy + 44, "scores Iₜ · 1 × L", 18, INK)
    s.rect(736, sy - 16, 130, 54, ACC_F, ACC, 2.5, 6)
    s.text(801, sy + 6, "Top-k", 19, INK, weight="bold")
    s.text(801, sy + 28, "2,048 ids", 17, INK)
    s.arrow(LX + COLS * CW + 3, sy + 11, 733, sy + 11)


for step in (1, 2):
    s = SVG(W, H, f"lightning indexer step {step}")
    common(s)
    if step == 2:
        step2(s)
    s.save(os.path.join(HERE, f"fig_indexer_{step}.svg"))
