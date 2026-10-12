"""fig_layer_1 / fig_layer_2: one merged layer pair on two CUDA streams.

Order follows TransformerLayerSchedulePlan.run (Megatron-LM main):
  comm: combine_bwd | dispatch_fwd -> dispatch_bwd | combine_fwd
  comp: pre_dispatch_fwd (attn) | mlp_bwd -> mlp_bwd_dw -> mlp_fwd | pre_dispatch_bwd (attn)
Durations are illustrative (time units), not measurements.
"""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
U = 1.3                 # pt per time unit
X0 = 160
YC, YN, H = 70, 150, 50


def blk(s, t0, t1, lane, label, kind, emph=False, dash=None):
    y = YC if lane == "c" else YN
    if kind == "F":
        fill, stroke, color = ACC_F, ACC, ACC
    elif kind == "B":
        fill, stroke, color = NEU_F, NEU, INK
    else:  # ghost
        fill, stroke, color = "#FFFFFF", "#A6A6A6", MUTED
    s.box(X0 + t0 * U, y, (t1 - t0) * U, H, label, fill, ACC if emph else stroke, color=color,
          weight="bold" if emph else "normal", sw=3 if emph else 2, dash=dash)


def frame(s):
    s.rect(160, 12, 26, 18, ACC_F, ACC, 2, 3)
    s.text(194, 27, "F = forward of micro-batch n+1", 17, INK, anchor="start")
    s.rect(500, 12, 26, 18, NEU_F, NEU, 2, 3)
    s.text(534, 27, "B = backward of micro-batch n", 17, INK, anchor="start")
    s.text(148, YC + H / 2 + 6.3, "compute", 18, INK, anchor="end", weight="bold")
    s.text(148, YN + H / 2 + 6.3, "comm", 18, INK, anchor="end", weight="bold")
    s.text(148, YN + H / 2 + 26, "(a2a stream)", 15, MUTED, anchor="end", style="italic")
    s.line(X0 - 4, 228, 862, 228, INK, 2, "arr")
    s.text(858, 254, "time", 18, INK, anchor="end")


# ---- build 1: B/mlp not split; B/dispatch waits for the whole backward MLP
s = SVG(870, 262, "fig_layer_1: merged layer pair without W/D split")
frame(s)
blk(s, 0, 100, "c", "F/attn", "F")
blk(s, 100, 300, "c", "B/mlp", "B")
blk(s, 300, 350, "c", "F/mlp", "F")
s.rect(X0 + 350 * U, YC, 30 * U, H, "url(#hatchPrb)", PRB, 2, 3)
s.text(X0 + 365 * U, YC - 8, "wait", 18, PRB, weight="bold")
blk(s, 380, 540, "c", "B/attn", "B")
blk(s, 0, 80, "n", "B/combine", "B")
blk(s, 100, 180, "n", "F/dispatch", "F")
blk(s, 300, 380, "n", "B/dispatch", "B")
blk(s, 380, 460, "n", "F/combine", "F")
# dependencies that create the wait
s.line(X0 + 300 * U, YC + H + 2, X0 + 300 * U, YN - 1, PRB, 2.4, "arrPrb")
s.line(X0 + 380 * U, YN - 2, X0 + 380 * U, YC + H + 1, PRB, 2.4, "arrPrb")
s.save(os.path.join(HERE, "fig_layer_1.svg"))

# ---- build 2: W/D split (--delay-wgrad-compute)
s = SVG(870, 262, "fig_layer_2: merged layer pair with W/D split; attention wgrad delayed")
frame(s)
blk(s, 0, 100, "c", "F/attn", "F")
blk(s, 100, 200, "c", "D/mlp", "B")
blk(s, 200, 300, "c", "W/mlp", "B", emph=True)
blk(s, 300, 350, "c", "F/mlp", "F")
blk(s, 350, 430, "c", "D/attn", "B")
blk(s, 430, 520, "c", "W/attn", "B", emph=True)
blk(s, 0, 80, "n", "B/combine", "B")
blk(s, 100, 180, "n", "F/dispatch", "F")
blk(s, 200, 280, "n", "B/dispatch", "B")
blk(s, 350, 430, "n", "F/combine", "F")
blk(s, 430, 520, "n", "next pair", "G", dash="6,4")
s.line(X0 + 200 * U, YC + H + 2, X0 + 200 * U, YN - 1, ACC, 2.4, "arrAcc")
s.line(X0 + 540 * U, 52, X0 + 540 * U, YN + H + 6, MUTED, 1.8, dash="5,5")
s.text(X0 + 540 * U, 46, "without split", 16, MUTED, anchor="end", style="italic")
s.save(os.path.join(HERE, "fig_layer_2.svg"))
