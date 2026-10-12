"""fig_hook: one MoE layer under expert parallelism; the hot expert's GPU sets the pace, the others idle."""
import os
from svgkit import SVG, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(480, 300, "fig_hook: per-GPU expert compute in one MoE layer; imbalance = idle GPUs")

X0, XEND = 96, 436
LANE_Y = [50, 104, 158, 212]
LH = 40
work = [XEND, 280, 240, 312]          # bar end x per GPU (schematic, not data)

s.text(X0, 32, "expert compute in one MoE layer", 16, MUTED, anchor="start", style="italic")
for i, (y, xe) in enumerate(zip(LANE_Y, work)):
    s.text(84, y + LH / 2 + 6.3, f"GPU {i}", 18, INK, anchor="end")
    if i == 0:
        s.box(X0, y, xe - X0, LH, "E0 · hot expert", PRB_F, PRB, color=PRB, weight="bold")
    else:
        s.box(X0, y, xe - X0, LH, f"E{i}", NEU_F, NEU)
        s.rect(xe, y, XEND - xe, LH, "url(#hatchGray)", "#A6A6A6", 1.5, 4)
        s.text((xe + XEND) / 2, y + LH / 2 + 6.3, "idle", 17, MUTED, weight="bold", halo=True)
# the layer cannot combine until the slowest expert finishes
s.line(XEND + 6, 42, XEND + 6, 262, PRB, 2.2, dash="6,4")
s.text(XEND + 2, 284, "combine waits", 17, PRB, anchor="end", weight="bold")
s.line(X0, 272, 300, 272, INK, 2, "arr")
s.text(X0, 292, "time", 16, MUTED, anchor="start", style="italic")
s.save(os.path.join(HERE, "fig_hook.svg"))
