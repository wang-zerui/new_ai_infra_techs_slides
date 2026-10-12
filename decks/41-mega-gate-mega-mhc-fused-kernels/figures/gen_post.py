"""fig_post: Mega mHC Post workers compute each new-residual element once and hand it to three consumers."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

W, H = 870, 292
s = SVG(W, H, "one pass, three consumers")

# HBM chips (left: inputs, right: outputs)
s.text(73, 18, "HBM", 17, MUTED, weight="bold")
s.text(798, 18, "HBM", 17, MUTED, weight="bold")
s.box(6, 34, 134, 52, ["residual", "4 × d BF16"], fill="#FFFFFF", stroke=NEU, sizes=[17, 15])
s.box(6, 104, 134, 52, ["x (layer out)", "d BF16"], fill="#FFFFFF", stroke=NEU, sizes=[17, 15])
s.box(6, 226, 134, 52, ["fn weights", "24 × 4d FP32"], fill="#FFFFFF", stroke=NEU, sizes=[17, 15])
s.box(730, 34, 134, 52, ["new residual", "4 × d BF16"], fill="#FFFFFF", stroke=NEU, sizes=[17, 15])
s.box(730, 170, 134, 52, ["24 sums", "per K split"], fill="#FFFFFF", stroke=NEU, sizes=[17, 15])

# CTA container
s.rect(160, 8, 552, 280, "#FFFFFF", ACC, 2, 8, dash="7,4")
s.text(172, 28, "one CTA per SM", 16, ACC, anchor="start", weight="bold")

# Post workers
s.box(214, 84, 196, 82, ["Post workers", "H_post·x + H_resᵀ·res", "→ BF16, once"], fill=ACC_F, stroke=ACC,
      sizes=[18, 15, 15], weights=["bold", "normal", "normal"])
s.line(140, 60, 213, 100, INK, 2.2, "arr")
s.line(140, 130, 213, 140, INK, 2.2, "arr")
s.text(178, 110, "TMA", 15, MUTED, style="italic")

# consumer 1: TMA store of the new residual
s.box(470, 34, 180, 52, ["store warp", "TMA, kept in L2"], fill="#FFFFFF", stroke=ACC, sizes=[17, 15],
      weights=["bold", "normal"])
s.path("M 410 104 L 440 104 L 440 60 L 469 60", ACC, 2.4, "arrAcc")
s.line(650, 60, 729, 60, INK, 2.2, "arr")

# consumer 2: tensor memory as the MMA's A operand
s.box(470, 170, 180, 52, ["MMA warp", "TF32, 24 outputs"], fill="#FFFFFF", stroke=ACC, sizes=[17, 15],
      weights=["bold", "normal"])
s.box(250, 190, 124, 40, "TMEM: A", fill=ACC_F, stroke=ACC, size=16)
s.line(312, 166, 312, 189, ACC, 2.4, "arrAcc")
s.line(374, 210, 469, 210, ACC, 2.4, "arrAcc")
s.line(650, 196, 729, 196, INK, 2.2, "arr")
s.path("M 140 252 L 560 252 L 560 223", INK, 2.2, "arr")
s.text(330, 272, "TMA", 15, MUTED, style="italic")

# consumer 3: running sum of squares for the RMS
s.box(470, 104, 180, 48, ["Σ x² per row", "for the RMS"], fill="#FFFFFF", stroke=ACC, sizes=[17, 15],
      weights=["bold", "normal"])
s.line(410, 128, 469, 128, ACC, 2.4, "arrAcc")

s.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_post.svg"))
