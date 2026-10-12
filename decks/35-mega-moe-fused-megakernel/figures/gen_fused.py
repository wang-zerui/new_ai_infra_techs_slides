"""fig_fused: the same MoE layer as one persistent kernel (schematic, same canvas/lanes as fig_baseline)."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

W, H = 870, 292
s = SVG(W, H, "Mega MoE timeline: one kernel, tiles interleaved")

LX = 122
TC = (34, 78)
HB = (120, 156)
NV = (198, 242)
AX = 266
th, nh = TC[1] - TC[0], NV[1] - NV[0]

# lane labels
s.text(LX, (TC[0] + TC[1]) / 2 + 6, "Compute", 18, INK, anchor="end", weight="bold")
s.text(LX, (HB[0] + HB[1]) / 2 + 6, "HBM", 18, INK, anchor="end", weight="bold")
s.text(LX, (NV[0] + NV[1]) / 2 + 6, "NVLink", 18, INK, anchor="end", weight="bold")

# one-kernel container
K0, K1 = 130, 600
s.rect(K0, 22, K1 - K0, 230, "#FFFFFF", ACC, 2.2, 8, dash="7,4")
s.text(K0 + 8, 16, "one kernel, every SM", 17, ACC, anchor="start", weight="bold")

# compute lane: fill idle, L1 warmup, interleaved L2/L1, final reduce
s.rect(136, TC[0], 44, th, "url(#hatchGray)", "#BFBFBF", 1, 3)
tiles = [("L1", 184, 236), ("L1", 240, 292), ("L1", 296, 348), ("L2", 352, 398), ("L1", 402, 454),
         ("L2", 458, 504), ("L2", 508, 554)]
for lab, x0, x1 in tiles:
    s.box(x0, TC[0], x1 - x0, th, lab, fill=ACC_F, stroke=ACC, size=18)
s.box(558, TC[0], 36, th, "Σ", fill=ACC_F, stroke=ACC, size=18, weight="bold")

# NVLink lane: counts, pulls, stores
s.box(136, NV[0], 40, nh, "idx", fill=NEU_F, stroke=NEU, size=16)
pulls = [(180, 226), (230, 276), (280, 326), (330, 376), (380, 426)]
for x0, x1 in pulls:
    s.box(x0, NV[0], x1 - x0, nh, "pull", fill=ACC_F, stroke=ACC, size=16)
stores = [(430, 476), (480, 526), (530, 576)]
for x0, x1 in stores:
    s.box(x0, NV[0], x1 - x0, nh, "store", fill=ACC_F, stroke=ACC, size=15)

# HBM lane: ring buffers
s.box(184, HB[0], 196, HB[1] - HB[0], "L1 / L2 ring buffers", fill="#FFFFFF", stroke=ACC, color=ACC, size=17)

# one dependency chain: pull -> L1 -> L2 -> store
s.path(f"M 203 {NV[0]} L 203 {HB[1] + 4}", ACC, 1.8, dash="3,3")
s.path(f"M 210 {HB[0] - 4} L 210 {TC[1] + 1}", ACC, 1.8, "arrAcc", dash="3,3")
s.path(f"M 226 {TC[1] + 1} Q 300 118 372 {TC[1] + 1}", ACC, 1.8, "arrAcc", dash="3,3")
s.path(f"M 392 {TC[1]} L 450 {NV[0] - 1}", ACC, 1.8, "arrAcc", dash="3,3")

# badges
s.badge(253, 182, 1, fill=ACC)
s.badge(300, 99, 2, fill=ACC)
s.badge(505, 182, 3, fill=ACC)

# reference end of the five-kernel version
s.line(812, 26, 812, 252, MUTED, 2, dash="5,4")
s.text(806, 46, "five kernels", 16, MUTED, anchor="end", style="italic")
s.text(806, 64, "end here", 16, MUTED, anchor="end", style="italic")
s.line(K1 + 6, 232, 808, 232, MUTED, 2, "arrGray", start_marker="arrGray")
s.text((K1 + 812) / 2, 224, "time saved", 17, INK, weight="bold")

# time axis
s.line(132, AX, 856, AX, INK, 2, "arr")
s.text(856, AX + 20, "time", 16, MUTED, anchor="end", style="italic")
s.text(132, AX + 20, "same MoE layer, same GPU (schematic)", 16, MUTED, anchor="start", style="italic")

s.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_fused.svg"))
