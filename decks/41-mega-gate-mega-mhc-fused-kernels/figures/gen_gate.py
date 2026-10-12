"""fig_gate: router before (cuBLASLt GEMM + top-k kernel) and as Mega Gate (one kernel)."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

W, H = 870, 292
s = SVG(W, H, "Mega Gate before/after")

# before
s.text(80, 56, "before", 18, MUTED, anchor="end", weight="bold")
s.box(92, 22, 170, 56, ["GEMM (cuBLASLt)", "+ split-K reduce"], fill=NEU_F, stroke=NEU, sizes=[17, 15],
      weights=["bold", "normal"])
s.line(262, 50, 293, 50, INK, 2.2, "arr")
s.box(294, 26, 172, 48, "logits T × E FP32", fill=PRB_F, stroke=PRB, color=PRB, size=16)
s.line(466, 50, 497, 50, INK, 2.2, "arr")
s.box(498, 22, 170, 56, ["top-k kernel", "(TileLang)"], fill=NEU_F, stroke=NEU, sizes=[17, 15],
      weights=["bold", "normal"])
s.line(668, 50, 715, 50, INK, 2.2, "arr")
s.box(716, 26, 150, 48, "idx, weights", fill="#FFFFFF", stroke=NEU, size=16)
for xb in [278, 482]:
    s.line(xb, 14, xb, 86, PRB, 2, dash="6,4")

# Mega Gate
s.text(80, 210, "Mega", 18, ACC, anchor="end", weight="bold")
s.text(80, 232, "Gate", 18, ACC, anchor="end", weight="bold")
s.rect(92, 112, 588, 172, "#FFFFFF", ACC, 2, 8, dash="7,4")
s.text(102, 132, "one kernel", 16, ACC, anchor="start", weight="bold")
s.box(104, 150, 166, 108, ["MMA warp", "BF16 x · Wᵀ", "experts on lanes", "tokens on columns"],
      fill=ACC_F, stroke=ACC, sizes=[17, 15, 15, 15], weights=["bold", "normal", "normal", "normal"])
steps = [(294, ["√softplus", "+ bias"]), (422, ["top-6", "per token"]), (550, ["normalise", "× scale, map"])]
for i, (x, lab) in enumerate(steps):
    s.box(x, 172, 104, 64, lab, fill=ACC_F, stroke=ACC, sizes=[16, 15])
    if i < 2:
        s.line(x + 104, 204, steps[i + 1][0] - 1, 204, ACC, 2.2, "arrAcc")
s.line(270, 204, 293, 204, ACC, 2.2, "arrAcc")
s.text(476, 160, "gate warps", 16, ACC, weight="bold")
s.line(654, 204, 715, 204, ACC, 2.4, "arrAcc")
s.box(716, 174, 150, 60, ["idx int64", "weights FP32"], fill="#FFFFFF", stroke=ACC, sizes=[16, 16])
s.text(791, 256, "→ Mega MoE", 16, ACC, weight="bold")

s.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_gate.svg"))
