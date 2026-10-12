"""fig_hc: one sub-layer wrapped in mHC with n = 4 residual streams (what is read, mixed and written)."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

W, H = 500, 336
s = SVG(W, H, "mHC around one layer")

ys = [160 + i * 22 for i in range(4)]
BH = 16
mid = ys[0] + (ys[-1] + BH - ys[0]) / 2      # vertical centre of the stack
# residual streams in and out
s.text(60, 148, "x_l: 4 × d", 17, INK, weight="bold")
s.text(445, 148, "x_l+1: 4 × d", 17, INK, weight="bold")
for y in ys:
    s.rect(8, y, 104, BH, NEU_F, NEU, 1.6, 2)
    s.rect(394, y, 102, BH, NEU_F, NEU, 1.6, 2)

# mapping boxes
s.box(126, 46, 70, 40, "H_pre", fill=ACC_F, stroke=ACC, size=17)
s.box(214, 38, 88, 56, ["F", "attn / MoE"], fill="#FFFFFF", stroke=NEU, sizes=[20, 15], weights=["bold", "normal"])
s.box(320, 46, 72, 40, "H_post", fill=ACC_F, stroke=ACC, size=17)
s.box(170, mid - 30, 146, 60, ["H_res 4×4", "doubly stochastic"], fill=ACC_F, stroke=ACC, sizes=[17, 15])

# residual path: x_l -> H_res -> (+) -> x_l+1
s.line(112, mid, 169, mid, INK, 2.4, "arr")
s.add(f'<circle cx="138" cy="{mid}" r="3.5" fill="{INK}"/>')
s.line(316, mid, 361, mid, INK, 2.4, "arr")
s.text(338, mid - 8, "4 × d", 15, MUTED, style="italic")
s.add(f'<circle cx="372" cy="{mid}" r="10" fill="#FFFFFF" stroke="{INK}" stroke-width="2"/>')
s.line(366, mid, 378, mid, INK, 1.8)
s.line(372, mid - 6, 372, mid + 6, INK, 1.8)
s.line(382, mid, 393, mid, INK, 2.4, "arr")

# layer path: x_l -> H_pre -> F -> H_post -> (+)
s.path(f"M 138 {mid} L 160 87", INK, 2.2, "arr")
s.text(142, 128, "4 × d", 15, MUTED, anchor="end", style="italic")
s.line(196, 66, 213, 66, INK, 2.2, "arr")
s.text(205, 58, "d", 15, MUTED, style="italic")
s.line(302, 66, 319, 66, INK, 2.2, "arr")
s.text(311, 58, "d", 15, MUTED, style="italic")
s.path(f"M 356 87 L 372 {mid - 11}", INK, 2.2, "arr")
s.text(372, 124, "4 × d", 15, MUTED, anchor="start", style="italic")

# where the 24 coefficients come from
s.text(250, 296, "24 coefficients per token and layer, from x_l:", 16, INK, style="italic")
s.text(250, 318, "RMSNorm · [4d → 24] GEMM · σ and Sinkhorn", 16, ACC, weight="bold")
s.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_hc.svg"))
