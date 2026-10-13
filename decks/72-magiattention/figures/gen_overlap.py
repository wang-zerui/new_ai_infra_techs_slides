"""fig_overlap: multi-stage overlap with overlap_degree = 3 remote stages ([BLOG] Fig. 13, Multi-Stage Overlap).
Forward: GroupCast prefetches KV of stage i+1 while FFA computes stage i (s0 = host stage, local KV only).
Backward: prefetch KV i+1 and GroupReduce dKV of stage i-1 while FFA-bwd computes stage i; only the last dKV
reduce is exposed. Schematic, not to scale."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 282, "fig_overlap: forward and backward stage timelines")

X0, SW, G = 168, 150, 4
LH = 32


def st(i):
    return X0 + i * (SW + G)


def lane(y, label):
    s.text(X0 - 12, y + LH / 2 + 6, label, 16, INK, anchor="end", weight="bold")


def blk(i, y, text, kind, w=None, x=None):
    x = st(i) if x is None else x
    w = SW if w is None else w
    fill, stroke, col = {"comp": (NEU_F, NEU, INK), "comm": (ACC_F, ACC, ACC),
                         "exp": ("url(#hatchPrb)", PRB, PRB)}[kind]
    s.rect(x, y, w, LH, fill, stroke, 1.8, 3)
    s.text(x + w / 2, y + LH / 2 + 6, text, 16, col, weight="bold", halo=(kind == "exp"))


# forward
s.text(10, 22, "Forward", 18, INK, anchor="start", weight="bold")
YF1, YF2 = 32, 70
lane(YF1, "FFA")
lane(YF2, "GroupCast")
for i in range(4):
    blk(i, YF1, "host s0" if i == 0 else f"stage s{i}", "comp")
for i in range(3):
    blk(i, YF2, f"KV s{i + 1}", "comm")
# backward
s.text(10, 134, "Backward", 18, INK, anchor="start", weight="bold")
YB1, YB2, YB3 = 144, 182, 220
lane(YB1, "FFA bwd")
lane(YB2, "GroupCast")
lane(YB3, "GroupReduce")
for i in range(4):
    blk(i, YB1, "host s0" if i == 0 else f"stage s{i}", "comp")
for i in range(3):
    blk(i, YB2, f"KV s{i + 1}", "comm")
blk(2, YB3, "dKV s1", "comm")
blk(3, YB3, "dKV s2", "comm")
blk(4, YB3, "dKV s3", "exp", w=860 - st(4))
s.text(860, YB3 - 6, "exposed", 15, PRB, anchor="end", weight="bold")
# time axis
s.line(X0, 266, 862, 266, INK, 1.8, "arr")
s.text(X0 - 12, 272, "time", 16, MUTED, anchor="end", style="italic")
s.save(os.path.join(HERE, "fig_overlap.svg"))
