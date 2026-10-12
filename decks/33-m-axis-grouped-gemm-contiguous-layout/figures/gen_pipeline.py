"""fig_pipeline: who produces the contiguous layout. (a) DeepEP V1 normal dispatch + a separate permute/pad kernel
(SGLang, 2025); (b) DeepEP V2 dispatch writes the aligned, zero-padded layout and psum offsets directly, and both
expert GEMMs (and the backward) reuse it."""
import os
from tilekit import SVG, seg, pad, outline, mono, EXP, INK, MUTED, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 290, "fig_pipeline: dispatch -> aligned layout -> two grouped GEMMs")


def chip(x, y, aligned=True, w=40, h=52):
    """Small A-matrix icon: three expert segments, with or without alignment padding."""
    if aligned:
        parts = [(0, 14), (None, 6), (1, 8), (None, 4), (2, 14), (None, 6)]
    else:
        parts = [(1, 7), (0, 9), (2, 8), (0, 6), (1, 7), (2, 9), (0, 6)]
    yy = y
    for e, hh in parts:
        if e is None:
            pad(s, x, yy, w, hh)
        else:
            seg(s, x, yy, w, hh, e, None, sw=1)
        yy += hh
    outline(s, x, y, w, h, INK, 1.6)


def arrow(x1, x2, y, col=INK):
    s.line(x1 + 2, y, x2 - 3, y, col, 2.2, "arrAcc" if col == ACC else "arr")


# ---------- (a) before
YA = 64
s.text(14, 30, "(a) DeepEP V1 normal dispatch + permute kernel", 18, INK, anchor="start", weight="bold")
s.box(14, YA, 128, 44, "dispatch", NEU_F, NEU, INK, 17)
arrow(142, 160, YA + 22)
chip(162, YA - 4, aligned=False)
s.text(182, YA + 66, "by source rank", 15, MUTED)
arrow(204, 224, YA + 22)
s.box(226, YA, 170, 44, "permute + pad", PRB_F, PRB, PRB, 17, "bold")
s.text(311, YA + 66, "extra read + write", 15, PRB)
arrow(396, 416, YA + 22)
chip(418, YA - 4)
arrow(460, 480, YA + 22)
s.box(482, YA, 150, 44, "grouped GEMM", NEU_F, NEU, INK, 17)

s.line(10, 150, 860, 150, "#999999", 1)

# ---------- (b) DeepEP V2 writes the layout
YB = 196
s.text(14, 176, "(b) DeepEP V2 dispatch writes the GEMM layout", 18, ACC, anchor="start", weight="bold")
s.box(14, YB, 176, 44, "dispatch", ACC_F, ACC, ACC, 17, "bold")
mono(s, 102, YB + 64, "expert_alignment=128", 15, ACC)
mono(s, 102, YB + 82, "do_zero_padding", 15, ACC)
arrow(190, 210, YB + 22, ACC)
chip(212, YB - 4)
mono(s, 232, YB + 64, "+ psum", 15, ACC)
arrow(254, 274, YB + 22)
s.box(276, YB, 124, 44, ["GEMM 1", "gate + up"], NEU_F, NEU, INK, 15)
arrow(400, 418, YB + 22)
s.box(420, YB, 120, 44, ["SwiGLU", "+ FP8 cast"], NEU_F, NEU, INK, 15)
arrow(540, 558, YB + 22)
s.box(560, YB, 124, 44, ["GEMM 2", "down"], NEU_F, NEU, INK, 15)
arrow(684, 702, YB + 22)
s.box(704, YB, 112, 44, "combine", NEU_F, NEU, INK, 17)
# the same layout feeds GEMM 2 and the backward pass
s.text(622, YB + 66, "same row layout for GEMM 2", 15, MUTED, style="italic")
s.text(622, YB + 82, "and backward (deck 32)", 15, MUTED, style="italic")
s.save(os.path.join(HERE, "fig_pipeline.svg"))
