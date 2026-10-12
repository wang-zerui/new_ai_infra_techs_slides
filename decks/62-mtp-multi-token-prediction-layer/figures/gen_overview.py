"""fig_overview: main model + MTP depth 1 (DeepSeek-V3 §2.2 Eq. 21-25), redrawn.
MTP: Combine = M[RMSNorm(h_i); RMSNorm(Emb(t_{i+1}))] -> Transformer block -> shared output head -> t_{i+2}."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 280
s = SVG(W, H, "fig_overview: main model and one MTP depth with shared embedding/head")

Y1, Y2, BH = 42, 196, 50
# main model row
s.text(20, 28, "main model", 16, MUTED, anchor="start", style="italic")
s.box(20, Y1, 136, BH, "Emb(tᵢ)", NEU_F, NEU)
s.box(190, Y1, 316, BH, "Transformer blocks × L", NEU_F, NEU)
s.box(540, Y1, 136, BH, "Output head", NEU_F, NEU)
s.line(158, Y1 + BH / 2, 188, Y1 + BH / 2, INK, 2.2, "arr")
s.line(508, Y1 + BH / 2, 538, Y1 + BH / 2, INK, 2.2, "arr")
s.line(678, Y1 + BH / 2, 706, Y1 + BH / 2, INK, 2.2, "arr")
s.text(712, Y1 + BH / 2 + 6.5, "tᵢ₊₁", 19, INK, anchor="start", weight="bold")
s.box(760, Y1 + 5, 106, 40, "L_main", "#FFFFFF", NEU, size=18)
# MTP depth-1 row
s.text(20, Y2 + BH + 24, "MTP depth 1", 16, ACC, anchor="start", style="italic", weight="bold")
s.box(20, Y2, 136, BH, "Emb(tᵢ₊₁)", "#FFFFFF", NEU, dash="6,4")
s.box(190, Y2, 150, BH, ["Norm + concat", "+ proj M"], ACC_F, ACC, color=ACC, weight="bold", size=17)
s.box(374, Y2, 132, BH, ["Transformer", "block"], ACC_F, ACC, color=ACC, weight="bold", size=17)
s.box(540, Y2, 136, BH, "Output head", "#FFFFFF", NEU, dash="6,4")
s.line(158, Y2 + BH / 2, 188, Y2 + BH / 2, INK, 2.2, "arr")
s.line(342, Y2 + BH / 2, 372, Y2 + BH / 2, INK, 2.2, "arr")
s.line(508, Y2 + BH / 2, 538, Y2 + BH / 2, INK, 2.2, "arr")
s.line(678, Y2 + BH / 2, 706, Y2 + BH / 2, ACC, 2.2, "arrAcc")
s.text(712, Y2 + BH / 2 + 6.5, "tᵢ₊₂", 19, ACC, anchor="start", weight="bold")
s.box(760, Y2 + 5, 106, 40, "λ · L_MTP", ACC_F, ACC, color=ACC, size=18, weight="bold")
# hidden state h_i from the last main block into the MTP combine
s.line(265, Y1 + BH + 2, 265, Y2 - 2, ACC, 2.5, "arrAcc")
s.text(275, 136, "hᵢ", 20, ACC, anchor="start", weight="bold", style="italic")
# shared embedding and output head
for x in (88, 608):
    s.line(x, Y1 + BH + 2, x, Y2 - 2, MUTED, 1.8, dash="4,3")
    s.text(x + 8, 140, "shared", 16, MUTED, anchor="start", style="italic")
s.save(os.path.join(HERE, "fig_overview.svg"))
