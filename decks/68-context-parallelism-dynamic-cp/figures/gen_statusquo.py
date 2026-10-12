"""fig_statusquo: (a) full recomputation repeats the forward; (b) more TP shrinks GEMMs below TP comm time."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 282, "fig_statusquo: two panels, schematic timelines")

# ---------------- panel (a): full recomputation
s.text(10, 24, "(a) Full recomputation", 20, INK, anchor="start", weight="bold")
Y, H = 104, 50
s.text(64, Y + H / 2 + 6.3, "GPU", 18, INK, anchor="end", weight="bold")
s.box(76, Y, 104, H, "forward", NEU_F, NEU)
s.rect(180, Y, 112, H, "url(#hatchPrb)", PRB, 2, 3)
s.text(236, Y + H / 2 + 6.3, "recompute", 18, PRB, weight="bold", halo=True)
s.box(292, Y, 120, H, "backward", NEU_F, NEU)
s.text(128, Y + H + 26, "drop", 16, MUTED, style="italic")
s.text(128, Y + H + 46, "activations", 16, MUTED, style="italic")
s.text(236, Y + H + 26, "forward", 16, PRB, style="italic")
s.text(236, Y + H + 46, "again", 16, PRB, style="italic")
s.line(76, 252, 410, 252, INK, 2, "arr")
s.text(410, 276, "time", 18, INK, anchor="end")

# divider
s.line(438, 14, 438, 276, "#BFBFBF", 1.2)

# ---------------- panel (b): more tensor parallelism
X0 = 540
s.text(456, 24, "(b) More tensor parallelism", 20, INK, anchor="start", weight="bold")


def lanes(y, tag, gemm, comm=50, n=3):
    s.text(468, y - 10, tag, 18, INK, anchor="start", weight="bold")
    s.text(X0 - 8, y + 22, "compute", 16, INK, anchor="end")
    s.text(X0 - 8, y + 61, "comm", 16, INK, anchor="end")
    ag_end, g_end, g_start = X0, X0, X0
    for k in range(n):
        ag_start = X0 if k == 0 else max(ag_end, g_start)
        ag_end = ag_start + comm
        st = max(ag_end, g_end)
        idle = st - g_end if k > 0 else 0
        exposed = idle > 0
        s.rect(ag_start, y + 40, comm, 30, PRB_F if exposed else "#FFFFFF", PRB if exposed else NEU, 1.8, 3)
        s.text(ag_start + comm / 2, y + 61, "AG", 16, PRB if exposed else INK)
        if idle > 0:
            s.rect(g_end, y, idle, 32, "url(#hatchPrb)", PRB, 1.5, 0)
        s.rect(st, y, gemm, 32, NEU_F, NEU, 1.8, 3)
        if gemm >= 60:
            s.text(st + gemm / 2, y + 22, "GEMM", 16, INK)
        g_start, g_end = st, st + gemm
    return g_end


lanes(62, "TP8", 88)
e = lanes(166, "TP32", 22, n=4)
s.text(e + 14, 184, "GPU waits", 16, PRB, anchor="start", style="italic")
s.text(e + 14, 204, "on comm", 16, PRB, anchor="start", style="italic")
s.line(X0, 252, 860, 252, INK, 2, "arr")
s.text(860, 276, "time", 18, INK, anchor="end")
s.save(os.path.join(HERE, "fig_statusquo.svg"))
