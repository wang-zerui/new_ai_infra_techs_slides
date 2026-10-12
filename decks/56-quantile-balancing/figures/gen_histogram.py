"""fig_histogram: K3 App. D histogram estimator. Every rank scatter-adds required biases r = alpha - s of all its
micro-batches into a local n x B count matrix (no communication); one all-reduce sums the counts; each expert's
bias is read where the cumulative count first reaches ceil(q), with linear interpolation inside that bin."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 292, "fig_histogram: local histograms, one all-reduce, quantile read-out")
I = "italic"


def hgrid(x, y, w=74, h=52, cols=6, rows=4):
    s.rect(x, y, w, h, ACC_F, ACC, 2, 2)
    for c in range(1, cols):
        s.line(x + c * w / cols, y, x + c * w / cols, y + h, ACC, 0.9)
    for r in range(1, rows):
        s.line(x, y + r * h / rows, x + w, y + r * h / rows, ACC, 0.9)


# ---------------- left: ranks x micro-batches -> local histograms
rows = [(46, "rank 0"), (176, "rank R−1")]
for y, name in rows:
    s.text(6, y - 10, name, 18, INK, anchor="start", weight="bold")
    for k, lab in enumerate(["mb 1", "mb 2", "⋯"]):
        bx = 6 + k * 62
        if lab == "⋯":
            s.text(bx + 22, y + 32, "⋯", 22, INK)
        else:
            s.box(bx, y, 56, 44, lab, NEU_F, NEU, size=17)
    s.line(166, y + 22, 214, y + 22, INK, 2.2, "arr")
    hgrid(216, y - 4)
s.text(110, 132, "⋮", 26, INK)
s.text(200, 122, "scatter-add", 16, MUTED, style="italic")
s.text(200, 142, "no comm", 16, MUTED, style="italic")
s.text(253, 250, "local n × B", 17, ACC)

# ---------------- middle: one all-reduce
AX, AY, AW, AH = 330, 74, 132, 136
s.rect(AX, AY, AW, AH, ACC_F, ACC, 2, 5)
s.text(AX + AW / 2, AY + 52, "all-reduce", 19, ACC, weight="bold")
s.text(AX + AW / 2, AY + 78, "int counts", 18, INK)
s.text(AX + AW / 2, AY + 102, "once / step", 18, INK)
s.badge(AX, AY, 3)
s.path(f"M 292 {46 + 22} H 310 V {AY + 34} H {AX - 1}", INK, 2.2, "arr")
s.path(f"M 292 {176 + 22} H 310 V {AY + AH - 34} H {AX - 1}", INK, 2.2, "arr")
s.line(AX + AW + 2, AY + AH / 2, 500, AY + AH / 2, INK, 2.2, "arr")

# ---------------- right: one expert's global histogram and the quantile read-out
HX, HY0, BW = 512, 236, 24
heights = [6, 14, 26, 42, 60, 76, 86, 82, 70, 54, 40, 26, 16, 8]
sel = 3                                         # first bin whose cumulative count reaches ceil(q)
s.text(HX, 30, "expert j, global counts", 18, INK, anchor="start", weight="bold")
for i, h in enumerate(heights):
    x = HX + i * BW
    if i < sel:
        s.rect(x, HY0 - h, BW, h, ACC_F, ACC, 1.6, 0)
    elif i == sel:
        s.rect(x, HY0 - h, BW, h, "#FFFFFF", PRB, 2.4, 0)
    else:
        s.rect(x, HY0 - h, BW, h, "#FFFFFF", "#A6A6A6", 1.6, 0)
s.line(HX - 4, HY0, HX + len(heights) * BW + 10, HY0, INK, 2, "arr")
xq = HX + sel * BW + 0.55 * BW                  # linear interpolation inside the bin
s.line(xq, HY0 - 118, xq, HY0 + 2, PRB, 2.4, dash="5,4")
s.rich(xq + 6, HY0 - 124, [("b̂", PRB, "bold", I), ("j", PRB, "bold", "sub")], 21, anchor="start")
s.text(HX + sel * BW + 30, 66, "cumulative count", 17, INK, anchor="start")
s.rich(HX + sel * BW + 30, 88, [("first reaches ⌈", INK, "normal"), ("q", INK, "normal", I),
                                      ("⌉", INK, "normal")], 17, anchor="start")
s.rich(HX - 2, HY0 + 22, [("b", MUTED, "normal", I), ("min", MUTED, "normal", "sub"), (" − 1", MUTED, "normal")],
       17, anchor="start")
s.rich(HX + len(heights) * BW + 4, HY0 + 22, [("b", MUTED, "normal", I), ("max", MUTED, "normal", "sub"),
                                              (" + 1", MUTED, "normal")], 17, anchor="end")
s.rich(HX + len(heights) * BW / 2, HY0 + 48, [("required bias ", INK, "normal"), ("r", INK, "normal", I),
                                              (" = ", INK, "normal"), ("α", INK, "normal", I),
                                              (" − ", INK, "normal"), ("s", INK, "normal", I)], 18)
s.save(os.path.join(HERE, "fig_histogram.svg"))
