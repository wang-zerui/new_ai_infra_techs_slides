"""fig_moe_layer: one MoE layer on 4 EP ranks — attention, dispatch all-to-all, experts, combine all-to-all."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(480, 300, "fig_moe_layer: MoE layer = attention -> dispatch a2a -> experts -> combine a2a")

ROW_Y = [48, 105, 162, 219]
RH = 44
XL, XA, WA = 64, 74, 80          # label right edge, attention box
XD0, XD1 = XA + WA + 4, 248       # dispatch crossbar
XE, WE = 252, 62                  # experts
XC0, XC1 = XE + WE + 4, 396       # combine crossbar
XN, WN = 400, 72                  # next layer

s.text(XA + WA / 2, 32, "attention", 16, MUTED, style="italic")
s.text((XD0 + XD1) / 2, 32, "dispatch", 18, PRB, weight="bold")
s.text((XC0 + XC1) / 2, 32, "combine", 18, PRB, weight="bold")
s.text(XE + WE / 2, 32, "experts", 16, MUTED, style="italic")

offs = [-13.5, -4.5, 4.5, 13.5]
for i, y in enumerate(ROW_Y):
    cy = y + RH / 2
    s.text(XL, cy + 6.3, f"GPU {i}", 18, INK, anchor="end")
    s.box(XA, y, WA, RH, "Attn", NEU_F, NEU)
    s.box(XE, y, WE, RH, f"E{i}", NEU_F, NEU)
    s.box(XN, y, WN, RH, "Attn", "#FFFFFF", MUTED, color=MUTED, dash="6,4")
# all-to-all crossbars: every rank sends to every rank (all 16 arrows drawn)
for (x0, x1) in ((XD0, XD1), (XC0, XC1)):
    for i, ys in enumerate(ROW_Y):
        for j, yt in enumerate(ROW_Y):
            y1 = ys + RH / 2 + offs[j]
            y2 = yt + RH / 2 + offs[i]
            s.line(x0, y1, x1 - 0.5, y2, PRB, 1.5, "arrPrb")
s.text(XN + WN / 2, 32, "layer ℓ+1", 16, MUTED, style="italic")
s.text(240, 290, "repeats in all 58 MoE layers and again in backward", 17, INK, style="italic")
s.save(os.path.join(HERE, "fig_moe_layer.svg"))
