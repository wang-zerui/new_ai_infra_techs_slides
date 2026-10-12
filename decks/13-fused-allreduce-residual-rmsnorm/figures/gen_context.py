"""fig_tp_layer (context) and fig_unfused (status quo) for deck 13.

fig_tp_layer: the TP layer boundary — row-parallel GEMM partial sums -> AllReduce -> +residual -> RMSNorm.
fig_unfused: baseline in HPC-Ops' benchmark = NCCL all_reduce + flashinfer fused_add_rmsnorm (two kernels);
the reduced tensor is written to HBM by the collective and read back by the norm kernel; every GPU
normalizes all N tokens. Widths are schematic, not to scale.
"""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- fig_tp_layer (480 x 336)
s = SVG(480, 336, "fig_tp_layer: TP layer boundary")
X, W = 24, 290
# stack effect for 8 GPUs
for k in (2, 1):
    s.rect(X + 5 * k, 10 - 5 * k + 5, W, 56, "#FFFFFF", NEU, 1.6)
s.box(X, 15, W, 56, ["row-parallel GEMM", "on GPU 0 … 7"], NEU_F, NEU)
s.line(X + W / 2, 73, X + W / 2, 104, INK, 2.2, "arr")
s.text(X + W / 2 + 10, 95, "partial sums", 17, INK, anchor="start")
# dashed group: the three ops
s.rect(X - 10, 110, W + 20, 178, "#FFFFFF", ACC, 2, 6, dash="7,4")
s.text(X - 2, 128, "TP layer boundary", 16, ACC, anchor="start", weight="bold")
s.box(X + 20, 136, W - 40, 40, "AllReduce over 8 GPUs", ACC_F, ACC)
s.line(X + W / 2, 177, X + W / 2, 186, INK, 2.2, "arr")
s.box(X + 20, 188, W - 40, 40, "+ residual", ACC_F, ACC)
s.line(X + W / 2, 229, X + W / 2, 238, INK, 2.2, "arr")
s.box(X + 20, 240, W - 40, 40, "RMSNorm", ACC_F, ACC)
s.line(X + W / 2, 289, X + W / 2, 300, INK, 2.2, "arr")
s.box(X, 302, W, 30, "next block", "#FFFFFF", MUTED, color=MUTED, dash="6,4", size=17)
# side annotation
s.text(336, 186, "twice per layer", 18, INK, anchor="start", weight="bold")
s.text(336, 210, "after attention", 17, INK, anchor="start")
s.text(336, 232, "after the MLP", 17, INK, anchor="start")
s.save(os.path.join(HERE, "fig_tp_layer.svg"))

# ---------------------------------------------------------------- fig_unfused (870 x 290)
s = SVG(870, 290, "fig_unfused: NCCL AllReduce then fused_add_rmsnorm, one GPU's view")
YS, YH, H = 46, 186, 46          # SM lane, HBM lane, box height
s.text(118, YS + H / 2 + 6.3, "GPU SMs", 18, INK, anchor="end", weight="bold")
s.text(118, YH + H / 2 + 6.3, "HBM", 18, INK, anchor="end", weight="bold")
# kernels
s.box(130, YS, 250, H, "NCCL AllReduce", NEU_F, NEU)
s.rect(380, YS, 56, H, "url(#hatchPrb)", PRB, 2, 3)
s.text(408, YS - 10, "kernel boundary", 16, PRB, weight="bold")
s.box(436, YS, 410, H, "fused_add_rmsnorm on all N tokens", NEU_F, NEU)
# HBM tensors
s.box(130, YH, 120, H, "x partial", "#FFFFFF", NEU)
s.box(300, YH, 170, H, "x reduced", PRB_F, PRB, color=PRB, weight="bold")
s.box(520, YH, 130, H, "residual", "#FFFFFF", NEU)
s.box(700, YH, 146, H, "output", "#FFFFFF", NEU)
# AllReduce reads partial, writes reduced
s.line(190, YH - 2, 190, YS + H + 3, INK, 2.2, "arr")
s.line(345, YS + H + 2, 345, YH - 2, PRB, 2.6, "arrPrb")
# norm reads reduced back, reads+writes residual, writes output
s.line(455, YH - 2, 455, YS + H + 3, PRB, 2.6, "arrPrb")
s.line(570, YH - 2, 570, YS + H + 3, INK, 2.2, "arr")
s.line(600, YS + H + 2, 600, YH - 2, INK, 2.2, "arr")
s.line(773, YS + H + 2, 773, YH - 2, INK, 2.2, "arr")
s.text(400, YH + H + 30, "HBM round trip", 17, PRB, weight="bold")
s.line(118, 268, 846, 268, INK, 1.8, "arr")
s.text(846, 286, "time", 16, INK, anchor="end")
s.text(130, 286, "same on all eight GPUs", 16, MUTED, anchor="start", style="italic")
s.save(os.path.join(HERE, "fig_unfused.svg"))
