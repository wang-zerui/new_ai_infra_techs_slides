"""fig_fold_1 / fig_fold_2: the 64 GPUs of one PP stage (TP4 CP2 DP8 PP4 = 256 GPUs, [MC] Sec. 3.3.3).

Rank r = tp + 4*cp + 8*dp (order tp-cp-ep-dp-pp): row = DP replica, columns = TP4 x CP2.
(a) attention: 16 TP groups (row segments of 4).
(b) MoE when EP must be a sub-group of DP: EP groups are the 8 DP columns, ETP = TP = 4, EDP = 2.
(c) MoE folded: ETP1, one EP64 group over all 64 ranks, EDP1.
"""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
CELL, STEP, CPGAP = 18, 21, 6
GW = 8 * STEP - (STEP - CELL) + CPGAP      # grid width
GH = 8 * STEP - (STEP - CELL)
TOP = 44
PX = [32, 338, 644]


def cx(c):
    return c * STEP + (CPGAP if c >= 4 else 0)


def grid(s, x0, fill, stroke):
    for r in range(8):
        for c in range(8):
            s.rect(x0 + cx(c), TOP + r * STEP, CELL, CELL, fill, stroke, 1.2, 2)


def panel_attn(s):
    x0 = PX[0]
    s.text(x0 + GW / 2, 26, "attention", 19, INK, weight="bold")
    grid(s, x0, NEU_F, "#A6A6A6")
    for r in range(8):            # all 16 TP groups
        for h in range(2):
            s.rect(x0 + cx(4 * h) - 2, TOP + r * STEP - 2, 4 * STEP - (STEP - CELL) + 4, CELL + 4, "none", NEU, 2, 3)
    s.text(x0 + GW / 2, TOP + GH + 28, "TP4 · CP2 · DP8", 18, INK)
    s.text(x0 + GW / 2, TOP + GH + 52, "row = one DP replica", 16, MUTED, style="italic")


def panel_before(s):
    x0 = PX[1]
    s.text(x0 + GW / 2, 26, "MoE: EP ⊆ DP", 19, INK, weight="bold")
    grid(s, x0, NEU_F, "#A6A6A6")
    for c in range(8):            # all 8 EP groups = DP columns
        s.rect(x0 + cx(c) - 2, TOP - 2, CELL + 4, GH + 4, "none", PRB, 2, 3)
    s.text(x0 + GW / 2, TOP + GH + 28, "ETP4 · EP8 · EDP2", 18, INK)
    s.text(x0 + GW / 2, TOP + GH + 52, "EP ≤ DP = 8", 18, PRB, weight="bold")


def panel_fold(s):
    x0 = PX[2]
    s.text(x0 + GW / 2, 26, "MoE: folded", 19, ACC, weight="bold")
    grid(s, x0, ACC_F, ACC)
    s.rect(x0 - 4, TOP - 4, GW + 8, GH + 8, "none", ACC, 3, 5)
    s.text(x0 + GW / 2, TOP + GH + 28, "ETP1 · EP64 · EDP1", 18, INK)
    s.text(x0 + GW / 2, TOP + GH + 52, "one EP64 group", 18, ACC, weight="bold")


def arrow(s, x1, x2, label, color, marker):
    y = TOP + GH / 2
    s.line(x1, y, x2, y, color, 2.4, marker)
    s.text((x1 + x2) / 2, y - 12, label, 17, color, weight="bold")


for step in (1, 2):
    s = SVG(870, 274, f"fig_fold_{step}: per-PP-stage process-group grids")
    PX[0], PX[1] = (170, 500) if step == 1 else (32, 338)
    panel_attn(s)
    panel_before(s)
    arrow(s, PX[0] + GW + 8, PX[1] - 10, "inherits", MUTED, "arrGray")
    if step == 2:
        panel_fold(s)
        arrow(s, PX[1] + GW + 8, PX[2] - 10, "fold", ACC, "arrAcc")
    s.save(os.path.join(HERE, f"fig_fold_{step}.svg"))
