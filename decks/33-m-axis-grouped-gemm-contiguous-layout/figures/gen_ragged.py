"""fig_ragged: one GPU's experts in one MoE layer = several GEMMs with the same weight shape, ragged M."""
import os
from tilekit import SVG, seg, INK, MUTED, ACC

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(480, 296, "fig_ragged: per-expert GEMMs, same N x K, different M")

XX, WX = 58, 118          # tokens x_e  (M_e x K)
XW, WW = 222, 66          # weight W_e  (N x K), identical for every expert
XY, WY = 334, 96          # output y_e  (M_e x N)
rows = [(0, 40, 78), (1, 140, 26), (2, 194, 54)]   # (expert, y, height ∝ M_e)

s.text(XX + WX / 2, 22, "tokens x", 17, INK, weight="bold")
s.text(XW + WW / 2, 22, "weights W", 17, INK, weight="bold")
s.text(XY + WY / 2, 22, "outputs y", 17, INK, weight="bold")

for e, y, h in rows:
    s.text(40, y + h / 2 + 6, f"E{e}", 18, INK, anchor="end", weight="bold")
    seg(s, XX, y, WX, h, e, f"M{e} × K" if h >= 40 else None, size=16, weight="normal")
    if h < 40:
        s.text(XX + WX / 2, y + h / 2 + 5.6, f"M{e} × K", 16, INK)
    s.text((XX + WX + XW) / 2, y + h / 2 + 7, "×", 22, MUTED)
    seg(s, XW, y + h / 2 - 21, WW, 42, e, "N × K", size=15, weight="normal")
    s.text((XW + WW + XY) / 2, y + h / 2 + 7, "=", 22, MUTED)
    seg(s, XY, y, WY, h, e, f"M{e} × N" if h >= 40 else None, size=16, weight="normal")
    if h < 40:
        s.text(XY + WY / 2, y + h / 2 + 5.6, f"M{e} × N", 16, INK)

# shared-shape bracket under the weights, ragged-M bracket under the tokens
s.line(XW, 266, XW + WW, 266, ACC, 2)
s.text(XW + WW / 2, 286, "same shape", 16, ACC, weight="bold")
s.line(XX - 10, 266, XX + WX + 10, 266, INK, 2)
s.text(XX + WX / 2, 286, "M differs per expert", 16, INK, weight="bold")
s.save(os.path.join(HERE, "fig_ragged.svg"))
