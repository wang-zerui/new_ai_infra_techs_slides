"""fig_dcp_problem / fig_dcp_fix: packed variable-length micro-batches under static CP2 vs Dynamic CP
(two GPUs; redrawn after the example in arXiv 2603.07685 Fig. 36-38 / the Dynamic-CP blog)."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))


def tri(s, x, y, w, h):
    s.add(f'<path d="M{x:.1f},{y:.1f} L{x:.1f},{y + h:.1f} L{x + w:.1f},{y + h:.1f} z" fill="{PRB_F}" '
          f'stroke="{PRB}" stroke-width="1.6"/>')


def panel_work(s, x0):
    s.text(x0 + 4, 24, "(a) Equal tokens but unequal work", 20, INK, anchor="start", weight="bold")
    L, k = 280, 0.3
    xs = x0 + 16
    # micro-batch X: one long sequence
    s.text(xs, 60, "micro-batch X", 16, MUTED, anchor="start", style="italic")
    s.box(xs, 68, L, 28, "A", NEU_F, NEU, size=17)
    tri(s, xs, 96, L, L * k)
    s.text(xs + L + 12, 150, "attention", 16, PRB, anchor="start", style="italic")
    s.text(xs + L + 12, 168, "work", 16, PRB, anchor="start", style="italic")
    # micro-batch Y: four short sequences
    s.text(xs, 212, "micro-batch Y", 16, MUTED, anchor="start", style="italic")
    w = L / 4
    for i, n in enumerate("BCDE"):
        s.box(xs + i * w, 220, w, 28, n, NEU_F, NEU, size=17)
        tri(s, xs + i * w, 248, w, w * k)


def panel_cp(s, x0, title, dynamic):
    s.text(x0 + 4, 24, title, 20, INK, anchor="start", weight="bold")
    gx = x0 + 70
    c0, c0w = gx, 130
    c1 = gx + c0w + 20
    c1w = 180
    y0, y1, H = 76, 166, 50
    s.text(gx - 10, y0 + H / 2 + 6, "GPU0", 17, INK, anchor="end", weight="bold")
    s.text(gx - 10, y1 + H / 2 + 6, "GPU1", 17, INK, anchor="end", weight="bold")
    s.text(c0 + c0w / 2, 62, "micro-batch 0", 16, MUTED)
    s.text(c1 + c1w / 2, 62, "micro-batch 1", 16, MUTED)
    # micro-batch 0: long A needs CP2 in both modes
    s.box(c0, y0, c0w, H, ["A", "part 1"], NEU_F, NEU, size=16)
    s.box(c0, y1, c0w, H, ["A", "part 2"], NEU_F, NEU, size=16)
    s.line(c0 + c0w / 2, y0 + H + 4, c0 + c0w / 2, y1 - 4, ACC, 2.4, "arrAcc", start_marker="arrAcc")
    s.text(c0 + c0w / 2, 266, "CP2", 18, ACC, weight="bold")
    # micro-batch 1: short B and C
    if not dynamic:
        bw = (c1w - 10) / 2
        for i, n in enumerate("BC"):
            x = c1 + i * (bw + 10)
            s.box(x, y0, bw, H, [n, "part 1"], NEU_F, NEU, size=16)
            s.box(x, y1, bw, H, [n, "part 2"], NEU_F, NEU, size=16)
            s.line(x + bw / 2, y0 + H + 4, x + bw / 2, y1 - 4, PRB, 2.4, "arrPrb", start_marker="arrPrb")
        s.text(c1 + c1w / 2, 256, "CP2 with", 17, PRB, weight="bold")
        s.text(c1 + c1w / 2, 276, "needless KV traffic", 17, PRB, weight="bold")
    else:
        s.box(c1, y0, c1w, H, ["B", "whole sequence"], ACC_F, ACC, ACC, 16, "bold")
        s.box(c1, y1, c1w, H, ["C", "whole sequence"], ACC_F, ACC, ACC, 16, "bold")
        s.text(c1 + c1w / 2, 266, "CP1 + CP1", 18, ACC, weight="bold")


s = SVG(870, 290, "fig_dcp_problem: (a) equal-token packs with unequal attention work, (b) static CP2")
panel_work(s, 0)
s.line(424, 12, 424, 282, "#BFBFBF", 1.2)
panel_cp(s, 436, "(b) Static CP2", False)
s.save(os.path.join(HERE, "fig_dcp_problem.svg"))

s = SVG(870, 290, "fig_dcp_fix: (b) static CP2 vs (c) Dynamic CP")
panel_cp(s, 0, "(b) Static CP2", False)
s.line(434, 12, 434, 282, "#BFBFBF", 1.2)
panel_cp(s, 446, "(c) Dynamic CP", True)
s.save(os.path.join(HERE, "fig_dcp_fix.svg"))
