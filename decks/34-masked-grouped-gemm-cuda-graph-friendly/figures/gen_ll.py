"""fig_ll: one decode MoE layer captured in a CUDA Graph. DeepEP low-latency dispatch writes recv_x in the masked
layout plus recv_count on the GPU; recv_count is the masked_m of both expert GEMMs; the BF16 output in the same
layout is what low-latency combine reads. No tensor is copied or reshaped and nothing returns to the CPU."""
import os
from tilekit import SVG, seg, outline, mono, EXP, INK, MUTED, ACC, ACC_F, NEU, NEU_F

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 292, "fig_ll: DeepEP low-latency dispatch -> masked GEMMs -> low-latency combine")


def chip3d(x, y, fills=(0.55, 0.15, 0.4), w=40, h=54):
    for k in (2, 1, 0):
        ox, oy = 5 * k, -5 * k
        s.rect(x + ox, y + oy, w, h, "#FFFFFF", "#7F7F7F", 1.1, rx=0)
        f, st = EXP[k]
        s.rect(x + ox, y + oy, w, h * fills[k], f, st, 1.1, rx=0)
    outline(s, x, y, w, h, INK, 1.6)


def arrow(x1, x2, y=128):
    s.line(x1 + 2, y, x2 - 3, y, INK, 2.2, "arr")


# graph container
s.rect(8, 46, 854, 186, "#FFFFFF", ACC, 2, rx=8, dash="7,4")
s.text(20, 36, "one CUDA Graph · captured once · replayed every decode step", 17, ACC, anchor="start", weight="bold")

Y = 106
s.box(20, Y, 112, 44, ["low-latency", "dispatch"], NEU_F, NEU, INK, 15)
arrow(132, 152)
chip3d(156, Y - 6)
arrow(206, 226)
s.box(228, Y, 118, 44, ["masked GEMM", "gate + up"], ACC_F, ACC, ACC, 15, "bold")
arrow(346, 364)
s.box(366, Y, 100, 44, ["SwiGLU", "+ FP8 cast"], NEU_F, NEU, INK, 15)
arrow(466, 484)
s.box(486, Y, 118, 44, ["masked GEMM", "down"], ACC_F, ACC, ACC, 15, "bold")
arrow(604, 624)
chip3d(628, Y - 6)
arrow(678, 698)
s.box(700, Y, 150, 44, ["low-latency", "combine"], NEU_F, NEU, INK, 15)

# tensor names under the chips
mono(s, 180, 176, "recv_x", 15, INK, weight="bold")
mono(s, 180, 194, "FP8", 15, MUTED)
mono(s, 652, 176, "D", 15, INK, weight="bold")
mono(s, 652, 194, "BF16", 15, MUTED)
# recv_count feeds both GEMMs as masked_m (device to device)
mono(s, 180, 222, "recv_count", 15, ACC, weight="bold")
s.path("M232,216 L287,216 L287,152", ACC, 1.8, "arrAcc", dash="4,3")
s.path("M287,216 L545,216 L545,152", ACC, 1.8, "arrAcc", dash="4,3")
mono(s, 416, 210, "masked_m", 15, ACC, weight="bold")

s.text(435, 258, "slot shape [local experts, T × R, 7168]", 16, INK, weight="bold")
s.text(435, 280, "T = max tokens per rank · R = EP ranks", 15, MUTED)
s.save(os.path.join(HERE, "fig_ll.svg"))
