"""fig_kernel: data path of one Mega MoE call and the warp role that executes each step.

From DeepGEMM @7f2a703 sm100_fp8_fp4_mega_moe.cuh: dispatch warps pull tokens (TMA over NVLink) into a local
pool; MMA warps run Linear-1 then Linear-2 tiles wave by wave; epilogue warps apply SwiGLU + FP8 cast and
write Linear-2 BF16 output into the token owner's combine buffer; after an NVLink barrier each rank reduces
its tokens' top-k contributions.
"""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 292
s = SVG(W, H, "fig_kernel: Mega MoE data path and warp roles")

# container: one persistent kernel on rank r
s.rect(10, 70, 850, 140, "#FFFFFF", ACC, 2, 6, dash="7,4")
s.text(22, 202, "rank r · one persistent kernel", 16, ACC, anchor="start", style="italic")

BY, BH = 100, 52
boxes = [  # x, w, label lines, role
    (30, 150, ["token pool"], "dispatch warps"),
    (236, 160, ["Linear-1 GEMM"], "MMA warps"),
    (452, 170, ["SwiGLU +", "FP8 cast"], "epilogue warps"),
    (678, 160, ["Linear-2 GEMM"], "MMA warps"),
]
for x, w, lab, role in boxes:
    fill = "#FFFFFF" if role == "dispatch warps" else ACC_F
    s.box(x, BY, w, BH, lab, fill, ACC, INK, 18, sw=2)
    s.text(x + w / 2, BY + BH + 22, role, 16, ACC, weight="bold")
for (x, w, _, _), (nx, _, _, _) in zip(boxes, boxes[1:]):
    s.line(x + w + 2, BY + BH / 2, nx - 2, BY + BH / 2, INK, 2.2, "arr")

# loop back: next wave
x2c = boxes[1][0] + 80
x4c = boxes[3][0] + 80
s.path(f"M{x4c},{BY - 2} C{x4c},{78} {x2c},{78} {x2c},{BY - 3}", ACC, 2, "arrAcc", dash="5,4")
s.text((x2c + x4c) / 2, 84, "next wave", 16, ACC, style="italic", halo=True)

# top-left: token owner's FP8 tokens; pull into the pool
s.box(20, 4, 260, 38, "token owner · FP8 tokens", NEU_F, NEU, INK, 17, sw=1.8)
s.line(105, 44, 105, BY - 2, ACC, 2.6, "arrAcc")
s.text(115, 60, "pull · TMA over NVLink", 16, ACC, anchor="start", weight="bold", halo=True)

# Linear-2 epilogue writes BF16 into the owner's combine buffer
s.path(f"M{boxes[3][0] + boxes[3][1] + 2},{BY + BH / 2} L842,{BY + BH / 2} L842,248", ACC, 2.6, "arrAcc")
s.text(834, 236, "write BF16", 16, ACC, anchor="end", weight="bold", halo=True)
s.box(640, 250, 220, 38, "owner · combine buffer", NEU_F, NEU, INK, 17, sw=1.8)
s.box(370, 250, 210, 38, "top-k reduce", "#FFFFFF", ACC, INK, 17, sw=1.8)
s.line(638, 269, 582, 269, INK, 2.2, "arr")
s.line(368, 269, 300, 269, INK, 2.2, "arr")
s.text(292, 275, "output y", 17, INK, anchor="end")
s.text(475, 242, "after NVLink barrier", 15, MUTED, style="italic")
s.save(os.path.join(HERE, "fig_kernel.svg"))
