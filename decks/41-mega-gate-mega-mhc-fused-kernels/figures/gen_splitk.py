"""fig_splitk: split-K partials of the 24-wide projection are reduced, mixed and normalised in the same kernel."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

W, H = 870, 292
s = SVG(W, H, "split-K finishing inside Mega mHC")

# flattened row split into 16 K-chunks, one task per (64-token block, split)
x0, segw, n = 10, 24, 16
s.text(x0, 26, "one 64-token block, K = 4d in 16 splits", 17, INK, anchor="start", weight="bold")
for i in range(n):
    s.rect(x0 + i * segw, 38, segw - 3, 24, ACC_F, ACC, 1.4, 2)
    s.line(x0 + i * segw + (segw - 3) / 2, 62, x0 + i * segw + (segw - 3) / 2, 107, ACC, 1.6, "arrAcc")
    s.rect(x0 + i * segw, 108, segw - 3, 24, "#FFFFFF", ACC, 1.4, 2)
s.text(x0, 154, "24 sums + Σx² per split, in workspace", 16, MUTED, anchor="start", style="italic")
s.text(x0, 176, "splits are tasks spread over all SMs", 16, MUTED, anchor="start", style="italic")

# split barrier then Mix
s.line(x0 + n * segw, 120, 419, 120, INK, 2.2, "arr")
s.box(420, 96, 62, 48, ["split", "barrier"], fill="#FFFFFF", stroke=INK, sizes=[15, 15])
s.line(482, 120, 501, 120, INK, 2.2, "arr")
s.box(502, 74, 186, 92, ["Mix warps", "fixed-order sum, ÷ RMS", "× α + b, σ, Sinkhorn"], fill=ACC_F, stroke=ACC,
      sizes=[18, 15, 15], weights=["bold", "normal", "normal"])
s.line(688, 120, 711, 120, INK, 2.2, "arr")
s.box(712, 92, 154, 56, ["next mixes", "H_post, H_res 4×4"], fill="#FFFFFF", stroke=NEU, sizes=[17, 15])

# Norm
s.line(595, 166, 595, 197, ACC, 2.4, "arrAcc")
s.text(603, 186, "H_pre", 15, ACC, anchor="start")
s.box(502, 198, 186, 72, ["Norm warps", "Σ H_pre·row → RMSNorm"], fill=ACC_F, stroke=ACC,
      sizes=[18, 15], weights=["bold", "normal"])
s.line(688, 234, 711, 234, INK, 2.2, "arr")
s.box(712, 206, 154, 56, ["y: BF16 and/or", "FP8 + UE8M0"], fill="#FFFFFF", stroke=NEU, sizes=[16, 16])

s.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_splitk.svg"))
