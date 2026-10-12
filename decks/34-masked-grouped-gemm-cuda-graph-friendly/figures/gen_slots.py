"""fig_slots_1/2: across two decode steps, the contiguous layout changes its total M and every expert's offset;
the masked layout keeps G fixed slots of max_m rows, only masked_m changes (build: 1 = contiguous, 2 = both)."""
import os
from tilekit import SVG, seg, pad, outline, INK, MUTED, ACC, PRB

HERE = os.path.dirname(os.path.abspath(__file__))
BM, MAXM = 128, 512
PX = 200 / (3 * MAXM)
Y0, CW = 44, 58
steps = [[300, 60, 200], [120, 250, 90]]


def contiguous(x, counts):
    y = Y0
    for e, m in enumerate(counts):
        al = -(-m // BM) * BM
        seg(s, x, y, CW, m * PX, e, None, sw=1)
        pad(s, x, y + m * PX, CW, (al - m) * PX)
        if e == 1:
            s.line(x - 4, y, x + CW + 4, y, PRB, 2.4)
        y += al * PX
    outline(s, x, Y0, CW, y - Y0, INK, 1.8)
    return y


def masked(x, counts):
    for e, m in enumerate(counts):
        y = Y0 + e * MAXM * PX
        s.rect(x, y, CW, MAXM * PX, "#FFFFFF", "#BFBFBF", 1, rx=0)
        seg(s, x, y, CW, m * PX, e, None, sw=1)
        s.line(x - 3, y + m * PX, x + CW + 3, y + m * PX, ACC, 2.6)
        s.line(x, y, x + CW, y, INK, 1.8)
    outline(s, x, Y0, CW, 3 * MAXM * PX, INK, 1.8)


def build(step):
    global s
    s = SVG(480, 290, f"fig_slots_{step}: contiguous offsets move, masked slots stay")
    s.text(108, 24, "(a) contiguous", 18, INK, weight="bold")
    xs = [52, 136]
    ends = [contiguous(x, c) for x, c in zip(xs, steps)]
    for x, lab, c, yend in zip(xs, ["step t", "step t+1"], steps, ends):
        s.text(x + CW / 2, Y0 + 220, lab, 15, MUTED)
        tot = sum(-(-m // BM) * BM for m in c)
        s.text(x + CW / 2, yend + 20, f"ΣM = {tot}", 15, PRB, weight="bold")
    s.text(108, Y0 + 242, "shape and offsets move", 16, PRB, weight="bold")
    if step >= 2:
        s.line(240, 14, 240, 284, "#999999", 1)
        s.text(364, 24, "(b) masked", 18, ACC, weight="bold")
        xm = [296, 380]
        for x, c in zip(xm, steps):
            masked(x, c)
        for x, lab in zip(xm, ["step t", "step t+1"]):
            s.text(x + CW / 2, Y0 + 220, lab, 15, MUTED)
        s.text(366, Y0 + 242, "same shape and pointers", 16, ACC, weight="bold")
        for e in range(3):
            s.text(288, Y0 + (e + 0.5) * MAXM * PX + 5, f"E{e}", 15, INK, anchor="end", weight="bold")
    s.save(os.path.join(HERE, f"fig_slots_{step}.svg"))


for st in (1, 2):
    build(st)
