"""fig_overview: Quantile Balancing inside one training step (K3 §2.3.3 + App. D).
Per micro-batch: Top-(k+1) -> routes + cutoffs -> required biases scatter-added into a local histogram.
Per step: all-reduce counts -> per-expert quantile at rank q -> mean-centred bias for the next step."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 300, "fig_overview: QB data flow in one training step")


def rbox(x, y, w, h, segs_lines, fill, stroke, size=19):
    s.rect(x, y, w, h, fill, stroke, 2, 5)
    n = len(segs_lines)
    step = size * 1.2
    y0 = y + h / 2 - step * (n - 1) / 2 + 0.35 * size
    for i, segs in enumerate(segs_lines):
        s.rich(x + w / 2, y0 + i * step, segs, size)


I = "italic"
# lane labels and separator
s.text(8, 22, "every micro-batch, forward", 17, MUTED, anchor="start", style="italic")
s.line(8, 182, 862, 182, "#A6A6A6", 1.5, dash="6,5")
s.text(8, 204, "once per step", 17, MUTED, anchor="start", style="italic")

# A router (existing)
rbox(10, 74, 150, 60, [[("router", INK, "normal")],
                       [("s", INK, "bold", I), (" = σ(W", INK, "normal"), ("r", INK, "normal", "sub"),
                        ("x)", INK, "normal")]], NEU_F, NEU)
# B Top-(k+1) (new)
rbox(200, 74, 200, 60, [[("Top-(k+1)", ACC, "bold")], [("of ", INK, "normal"), ("s", INK, "bold", I),
                        (" + ", INK, "normal"), ("b", INK, "bold", I), ("(t)", INK, "normal")]], ACC_F, ACC)
s.badge(200, 74, 1)
s.line(160 + 2, 104, 200 - 1, 104, INK, 2.2, "arr")
# C dispatch (existing)
rbox(470, 30, 390, 46, [[("first k → dispatch to experts", INK, "normal")]], NEU_F, NEU)
# D cutoff + local histogram (new)
rbox(470, 112, 150, 58, [[("cutoff ", INK, "normal"), ("α", INK, "bold", I), ("i", INK, "normal", "sub")]],
     ACC_F, ACC)
rbox(690, 112, 170, 58, [[("local histogram", INK, "normal")], [("n × B counts", INK, "normal")]], ACC_F, ACC, size=18)
s.badge(838, 112, 3)
s.path("M 402 92 H 436 V 53 H 469", INK, 2.2, "arr")
s.path("M 402 116 H 436 V 141 H 469", INK, 2.2, "arr")
s.line(622, 141, 689, 141, INK, 2.2, "arr")
s.rich(655, 128, [("α", INK, "normal", I), ("i", INK, "normal", "sub"), (" − ", INK, "normal"),
                  ("s", INK, "normal", I), ("ij", INK, "normal", "sub")], 17)

# bottom lane (right to left)
rbox(690, 222, 170, 52, [[("all-reduce counts", INK, "normal")]], ACC_F, ACC)
rbox(400, 222, 250, 52, [[("quantile at rank ", INK, "normal"), ("q", INK, "normal", I),
                          (" → ", INK, "normal"), ("b̂", INK, "bold", I), ("j", INK, "normal", "sub")]],
     ACC_F, ACC)
s.badge(400, 222, 2)
rbox(130, 222, 230, 52, [[("b", INK, "bold", I), ("(t+1)", INK, "normal"), (" = ", INK, "normal"),
                          ("b̂", INK, "bold", I), (" − mean", INK, "normal")]], ACC_F, ACC)
s.line(775, 170 + 2, 775, 222 - 1, INK, 2.2, "arr")
s.line(690 - 2, 248, 650 + 1, 248, INK, 2.2, "arr")
s.line(400 - 2, 248, 360 + 1, 248, INK, 2.2, "arr")
# back to Top-(k+1) for the next step
s.path("M 245 222 V 136", ACC, 2.4, "arrAcc")
s.text(253, 172, "next step", 17, ACC, anchor="start", style="italic")
s.save(os.path.join(HERE, "fig_overview.svg"))
