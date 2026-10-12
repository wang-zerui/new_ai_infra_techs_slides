"""fig_moe_layer: the five stages of one DeepSeek-V4 MoE layer under expert parallelism (DeepSeek-V4 Fig. 5)."""
import os
from svgkit import SVG, ACC, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(480, 330, "fig_moe_layer: dispatch -> L1 -> SwiGLU+FP8 cast -> L2 -> combine")

X, BW, BH, GAP = 170, 300, 46, 20
stages = [
    (["Dispatch all-to-all"], "comm", "FP8 tokens out"),
    (["Linear-1 GEMM"], "comp", "gate + up"),
    (["SwiGLU + FP8 cast"], "comp", ""),
    (["Linear-2 GEMM"], "comp", "down"),
    (["Combine all-to-all"], "comm", "BF16 results back"),
]
ys = []
for i, (lab, kind, sub) in enumerate(stages):
    y = 6 + i * (BH + GAP)
    ys.append(y)
    if kind == "comm":
        s.box(X, y, BW, BH, lab, "#FFFFFF", NEU, INK, 19, weight="bold", sw=2, dash="7,4")
    else:
        s.box(X, y, BW, BH, lab, NEU_F, NEU, INK, 19, sw=2)
    if i < len(stages) - 1:
        s.line(X + BW / 2, y + BH + 2, X + BW / 2, y + BH + GAP - 1, INK, 2.2, "arr")

# side annotations (what moves / what computes)
s.text(X + BW / 2 + 12, ys[0] + BH + 15, "FP8 tokens", 15, MUTED, anchor="start", style="italic")
s.text(X + BW / 2 + 12, ys[3] + BH + 15, "BF16 results", 15, MUTED, anchor="start", style="italic")

# left brackets: network vs tensor cores
def bracket(y0, y1, label, color):
    xb = X - 16
    s.path(f"M{xb + 8},{y0} L{xb},{y0} L{xb},{y1} L{xb + 8},{y1}", color, 2)
    s.text(xb - 8, (y0 + y1) / 2 + 6, label, 18, color, anchor="end", weight="bold")

bracket(ys[0] + 4, ys[0] + BH - 4, "network", MUTED)
bracket(ys[1] + 4, ys[3] + BH - 4, "compute", INK)
bracket(ys[4] + 4, ys[4] + BH - 4, "network", MUTED)
s.save(os.path.join(HERE, "fig_moe_layer.svg"))
