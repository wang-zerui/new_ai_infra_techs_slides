"""fig_baseline: the non-fused MoE layer on one GPU = five kernels in sequence (schematic, not to scale)."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

W, H = 870, 292
s = SVG(W, H, "five-kernel MoE layer timeline")

LX = 122                      # lane-label right edge
TC = (34, 78)                 # tensor-core lane y range
HB = (120, 156)               # HBM chips
NV = (198, 242)               # NVLink lane
AX = 266                      # time axis

# legend (top right)
s.line(470, 13, 500, 13, PRB, 2.2, dash="6,4")
s.text(506, 18, "kernel boundary", 16, INK, anchor="start")
s.rect(650, 5, 26, 16, PRB_F, PRB, 1.5, 3)
s.text(682, 18, "HBM round trip", 16, INK, anchor="start")

# lane labels
s.text(LX, (TC[0] + TC[1]) / 2 + 6, "Compute", 18, INK, anchor="end", weight="bold")
s.text(LX, (HB[0] + HB[1]) / 2 + 6, "HBM", 18, INK, anchor="end", weight="bold")
s.text(LX, (NV[0] + NV[1]) / 2 + 6, "NVLink", 18, INK, anchor="end", weight="bold")

k = {  # kernel spans (x0, x1)
    "dispatch": (132, 252), "Linear 1": (262, 432), "SwiGLU": (442, 522),
    "Linear 2": (532, 682), "combine": (692, 812)}
th = TC[1] - TC[0]
nh = NV[1] - NV[0]

# idle hatches
for x0, x1 in [k["dispatch"], k["combine"]]:
    s.rect(x0, TC[0], x1 - x0, th, "url(#hatchGray)", "#BFBFBF", 1, 3)
    s.text((x0 + x1) / 2, TC[0] + th / 2 + 6, "idle", 16, MUTED, style="italic", halo=True)
s.rect(k["Linear 1"][0], NV[0], k["Linear 2"][1] - k["Linear 1"][0], nh, "url(#hatchGray)", "#BFBFBF", 1, 3)
s.text((k["Linear 1"][0] + k["Linear 2"][1]) / 2, NV[0] + nh / 2 + 6, "idle", 16, MUTED, style="italic", halo=True)

# kernels
for name in ["Linear 1", "SwiGLU", "Linear 2"]:
    x0, x1 = k[name]
    s.box(x0, TC[0], x1 - x0, th, name, fill=NEU_F, stroke=NEU, size=18)
for name in ["dispatch", "combine"]:
    x0, x1 = k[name]
    s.box(x0, NV[0], x1 - x0, nh, name, fill=NEU_F, stroke=NEU, size=18)

# kernel boundaries
for xb in [257, 437, 527, 687]:
    s.line(xb, 26, xb, 252, PRB, 2.2, dash="6,4")

# HBM round trips (what the fused kernel removes)
c1 = (380, 494)
s.box(c1[0], HB[0], c1[1] - c1[0], HB[1] - HB[0], "BF16 [n, 2I]", fill=PRB_F, stroke=PRB, color=PRB, size=17)
s.line(408, TC[1], 408, HB[0] - 1, PRB, 2.2, "arrPrb")
s.line(470, HB[0], 470, TC[1] + 1, PRB, 2.2, "arrPrb")
c2 = (634, 742)
s.box(c2[0], HB[0], c2[1] - c2[0], HB[1] - HB[0], "BF16 [n, d]", fill=PRB_F, stroke=PRB, color=PRB, size=17)
s.line(660, TC[1], 660, HB[0] - 1, PRB, 2.2, "arrPrb")
s.line(716, HB[1], 716, NV[0] - 1, PRB, 2.2, "arrPrb")

# time axis
s.line(132, AX, 856, AX, INK, 2, "arr")
s.text(856, AX + 20, "time", 16, MUTED, anchor="end", style="italic")
s.text(132, AX + 20, "one MoE layer on one GPU (schematic)", 16, MUTED, anchor="start", style="italic")

s.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_baseline.svg"))
