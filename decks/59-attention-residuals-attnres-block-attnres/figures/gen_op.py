"""fig_op: the AttnRes op for layer l (AR Eq. 2-4): keys = RMSNorm(v_i), one learned pseudo-query w_l,
softmax over sources, weighted sum of the raw v_i. Toy weights (row 'layer 4' of fig_matrix_2)."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
I = "italic"
s = SVG(490, 300, "fig_op: AttnRes op for one layer")

rows = [("v", "0", "embedding", .40), ("v", "1", "layer 1 out", .05), ("v", "2", "layer 2 out", .05),
        ("v", "3", "layer 3 out", .50)]
Y0, DY, H = 70, 52, 38
# pseudo-query
s.rect(186, 8, 130, 36, ACC_F, ACC, 2, 4)
s.rich(251, 32, [("query w", ACC, "bold"), ("l", ACC, "bold", "sub")], 18)
s.line(251, 46, 251, Y0 - 6, ACC, 2.2, "arrAcc")
# score box spanning rows
SBX, SBW = 166, 170
s.rect(SBX, Y0 - 4, SBW, len(rows) * DY - DY + H + 8, ACC_F, ACC, 2, 6)
for k, (v, sub, lab, a) in enumerate(rows):
    y = Y0 + k * DY
    s.rect(6, y, 128, H, NEU_F, NEU, 1.6, 4)
    s.rich(30, y + H / 2 + 7, [(v, INK, "bold", I), (sub, INK, "bold", "sub")], 19)
    s.text(46, y + H / 2 + 6, lab, 16, MUTED, anchor="start")
    s.line(136, y + H / 2, SBX - 2, y + H / 2, INK, 1.8, "arr")
    # alpha bar
    bw = 90 * a
    s.rect(SBX + SBW + 14, y + 8, max(bw, 3), H - 16, "#2E75B6" if a >= .3 else "#9DC3E6", ACC, 1.2, 2)
    s.text(SBX + SBW + 14 + max(bw, 3) + 6, y + H / 2 + 6, f"{a:.2f}".lstrip("0"), 16, INK, anchor="start")
s.rich(SBX + SBW / 2, Y0 + 54, [("w", INK, "bold", I), ("l", INK, "normal", "sub"), ("ᵀ RMSNorm(", INK, "normal"),
                                ("v", INK, "bold", I), ("i", INK, "normal", "sub"), (")", INK, "normal")], 17)
s.text(SBX + SBW / 2, Y0 + 92, "→ softmax over i", 17, INK)
s.rich(SBX + SBW + 14, Y0 - 16, [("α", ACC, "bold", I), ("i→l", ACC, "bold", "sub")], 19, anchor="start")
# weighted sum
SX, SY = 468, Y0 + 1.5 * DY + H / 2
for k in range(len(rows)):
    y = Y0 + k * DY + H / 2
    s.line(432, y, SX - 17, SY + (y - SY) * 0.25, INK, 1.4, "arr")
s.circle(SX, SY, 18, "#FFFFFF", INK, 2)
s.text(SX, SY + 7, "Σ", 19, INK, weight="bold")
s.text(SX, SY + 40, "Σ α·v", 16, MUTED)
s.line(SX, SY + 50, SX, 270, INK, 2.2, "arr")
s.rich(SX - 6, 292, [("h", INK, "bold", I), ("l", INK, "normal", "sub"), (" → layer l", INK, "normal")], 18,
       anchor="end")
s.save(os.path.join(HERE, "fig_op.svg"))
