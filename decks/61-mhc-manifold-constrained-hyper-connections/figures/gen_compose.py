"""fig_compose_1 / _2: product of the residual mappings across 60 sublayers ([mHC] eq. 2, 4; §4.1).
Residual: I...I = I (stable, no lane mixing). HC: unconstrained product drifts (gain unbounded).
mHC (build): every factor doubly stochastic -> product doubly stochastic, spectral norm <= 1."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED, GOOD, sup_text

HERE = os.path.dirname(os.path.abspath(__file__))

def row(s, y, label, kind):
    h = 54
    lab_col = {"res": INK, "hc": PRB, "mhc": ACC}[kind]
    s.text(4, y + h / 2 + 7, label, 21, lab_col, anchor="start", weight="bold")
    fill, stroke, tc = {"res": (NEU_F, NEU, INK), "hc": (PRB_F, PRB, PRB), "mhc": (ACC_F, ACC, ACC)}[kind]
    xs = [120, 204, 330]
    subs = ["60", "59", "1"]
    for x, sb in zip(xs, subs):
        s.rect(x, y, 70, h, fill, stroke, 2, 4)
        if kind == "res":
            s.text(x + 35, y + h / 2 + 8, "I", 22, tc, weight="bold")
        else:
            sup_text(s, x + 35, y + h / 2 + 7, "H", sb, 22, tc, sub=True)
    s.text(302, y + h / 2 + 7, "⋯", 24, MUTED)
    s.text(422, y + h / 2 + 8, "=", 26, INK, weight="bold")
    res = {"res": "I", "hc": "drifts", "mhc": "doubly stoch."}[kind]
    s.rect(446, y, 150, h, fill, stroke, 2.6, 4)
    s.text(521, y + h / 2 + 7, res, 20, tc, weight="bold")
    v = {"res": [("✓", "stable", GOOD), ("✗", "no lane mixing", PRB)],
         "hc": [("✓", "lanes mix", GOOD), ("✗", "gain unbounded", PRB)],
         "mhc": [("✓", "lanes mix", GOOD), ("✓", "spectral norm ≤ 1", GOOD)]}[kind]
    for i, (mark, txt, c) in enumerate(v):
        yy = y + 21 + i * 26
        s.text(616, yy, mark, 20, c, anchor="start", weight="bold")
        s.text(640, yy, txt, 19, INK, anchor="start")

def draw(n, name):
    s = SVG(870, 262, name)
    s.text(4, 20, "product of residual mappings across depth (60 sublayers in the 27B model)", 18, MUTED, anchor="start", style="italic")
    row(s, 36, "Residual", "res")
    row(s, 110, "HC", "hc")
    if n == 2:
        row(s, 184, "mHC", "mhc")
    s.save(os.path.join(HERE, name + ".svg"))

draw(1, "fig_compose_1")
draw(2, "fig_compose_2")
