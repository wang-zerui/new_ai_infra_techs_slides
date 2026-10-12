"""fig_sfa: the Sparse Flash Attention super-operator.

One kernel selects the top-2048 KV rows and runs the attention over them, so the gather of the
next tile overlaps the attention math of the current one (GLM-5 report: selection and sparse
attention "in parallel"). Tile count is illustrative.
"""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 300, "fig_sfa: Sparse Flash Attention gathers the selected KV rows and attends")

# ---- KV cache column with the selected rows
GX, GY, GW, RH = 30, 50, 104, 22
sel = {1, 4, 7}
for r in range(9):
    fill = ACC_F if r in sel else "#FFFFFF"
    stroke = ACC if r in sel else "#B0B0B0"
    s.rect(GX, GY + r * RH, GW, RH - 3, fill, stroke, 1.4, 2)
s.text(GX + GW / 2, GY - 32, "KV cache", 17, INK, weight="bold")
s.text(GX + GW / 2, GY - 12, "S rows", 16, MUTED)
s.text(GX + GW / 2, GY + 9 * RH + 18, "2048 rows selected", 16, ACC, weight="bold")
s.text(GX + GW / 2, GY + 9 * RH + 38, "by the indexer", 16, ACC)

# ---- the fused kernel
CX0, CX1, CY0, CY1 = 176, 856, 36, 212
s.rect(CX0, CY0, CX1 - CX0, CY1 - CY0, ACC_F, ACC, 3, 6)
s.text(CX0 + 14, CY0 + 20, "❸ Sparse Flash Attention  ·  one kernel", 19, ACC, anchor="start",
       weight="bold")
YG, YA, BH, TW = 62, 142, 46, 160
for i in range(3):
    xg = CX0 + 24 + i * (TW + 24)
    s.box(xg, YG, TW, BH, [f"gather tile {i + 1}"], "#FFFFFF", ACC, color=INK, size=16)
    xa = xg + 80
    s.box(xa, YA, TW, BH, [f"attend tile {i + 1}"], "#FFFFFF", ACC, color=INK, size=16)
    s.line(xg + TW - 14, YG + BH + 3, xa + 22, YA - 3, ACC, 2.2, "arrAcc")
s.text(CX1 - 14, CY0 + 20, "gather overlaps attention", 16, ACC, anchor="end", style="italic")
# selected rows feed the kernel
s.line(GX + GW + 4, 121, CX0 - 4, 121, ACC, 2.6, "arrAcc")
# output
s.line(738, CY1, 738, CY1 + 28, INK, 2.2, "arr")
s.text(750, CY1 + 26, "attention out", 17, INK, anchor="start")
s.text(CX0 + 24, CY1 + 26, "scores: q_nope · k_nopeᵀ + q_rope · k_ropeᵀ", 16, MUTED,
       anchor="start", style="italic")
s.save(os.path.join(HERE, "fig_sfa.svg"))
