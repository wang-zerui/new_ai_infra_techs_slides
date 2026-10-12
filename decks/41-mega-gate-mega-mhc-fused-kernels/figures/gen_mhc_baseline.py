"""fig_mhc_baseline: multi-kernel mHC layer boundary (normal mode, 4 kernels) and the HBM tensors between them.

Data flow from third-party/tilelang_ops/ref_mhc.py at DeepGEMM 66081d4c:
mhc_post -> tf32_hc_prenorm_gemm (split-K) -> mhc_pre_big_fuse -> norm. Schematic, not to scale.
"""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

W, H = 870, 292
s = SVG(W, H, "four-kernel mHC boundary")

KT, KB = 34, 86           # kernel lane
RA, RB = (122, 150), (178, 206)   # HBM rows
AX = 258

# legend
s.line(470, 13, 500, 13, PRB, 2.2, dash="6,4")
s.text(506, 18, "kernel boundary", 16, INK, anchor="start")
s.rect(650, 5, 26, 16, PRB_F, PRB, 1.5, 3)
s.text(682, 18, "HBM round trip", 16, INK, anchor="start")

# lane labels
s.text(104, (KT + KB) / 2 + 6, "Kernels", 18, INK, anchor="end", weight="bold")
s.text(104, (RA[0] + RB[1]) / 2 + 6, "HBM", 18, INK, anchor="end", weight="bold")

K = [(120, 248, ["post-mix", "H_post, H_res"]), (264, 398, ["24-wide GEMM", "split-K"]),
     (414, 588, ["mixes, Sinkhorn", "+ pre-mix"]), (604, 750, ["RMSNorm", "+ FP8 cast"])]
for x0, x1, lab in K:
    s.box(x0, KT, x1 - x0, KB - KT, lab, fill=NEU_F, stroke=NEU, sizes=[17, 15], weights=["bold", "normal"])
for xb in [256, 406, 596]:
    s.line(xb, 26, xb, 242, PRB, 2.2, dash="6,4")


def chip(x0, x1, row, label, bad):
    s.box(x0, row[0], x1 - x0, row[1] - row[0], label, fill=PRB_F if bad else "#FFFFFF",
          stroke=PRB if bad else NEU, color=PRB if bad else INK, size=16)


# inputs and outputs (neutral)
chip(120, 214, RA, "x, residual", False)
s.line(160, RA[0], 160, KB + 1, INK, 2, "arr")
chip(760, 860, RA, "y: d", False)
s.line(744, KB, 770, RA[0] - 1, INK, 2, "arr")

# new residual (row B): written by post-mix, re-read by the GEMM and by pre-mix (red reads)
chip(230, 540, RB, "new residual 4 × d", False)
s.line(240, KB, 240, RB[0] - 1, INK, 2, "arr")
s.line(300, RB[0], 300, KB + 1, PRB, 2.4, "arrPrb")
s.line(526, RB[0], 526, KB + 1, PRB, 2.4, "arrPrb")

# split-K partials (row A): GEMM -> big-fuse
chip(370, 480, RA, "partials", True)
s.line(388, KB, 388, RA[0] - 1, PRB, 2, "arrPrb")
s.line(462, RA[0], 462, KB + 1, PRB, 2, "arrPrb")

# norm input (row A): big-fuse -> norm
chip(556, 676, RA, "norm input d", True)
s.line(578, KB, 578, RA[0] - 1, PRB, 2, "arrPrb")
s.line(650, RA[0], 650, KB + 1, PRB, 2, "arrPrb")

# time axis
s.line(120, AX, 856, AX, INK, 2, "arr")
s.text(856, AX + 20, "time", 16, MUTED, anchor="end", style="italic")
s.text(120, AX + 20, "one layer boundary, one GPU (schematic)", 16, MUTED, anchor="start", style="italic")

s.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_mhc_baseline.svg"))
