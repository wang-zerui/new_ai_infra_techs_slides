"""fig_kgroup: K-grouped Wgrad. A = dY and B = X are the same token-sorted rows (K = tokens), each expert's segment
padded to the 128-token K block; grouped_layout holds each expert's padded K; each expert writes its own M x N
output D[e] = C[e] + A_e^T B_e."""
import os
from tilekit import SVG, seg, pad, hlines, outline, mono, EXP, INK, MUTED, ACC, ACC_F

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 292, "fig_kgroup: experts stacked along K, one output per expert")

BK, Y0 = 128, 50
experts = [(0, 300), (1, 60), (2, 200)]
px = 194 / 768
XG, WG = 24, 46
XA, WA = 112, 120
XB, WB = 262, 92
XD, WD, HD = 560, 104, 58

s.text(XA + WA / 2, 38, "A = dY (ΣK × M)", 17, INK, weight="bold")
s.text(XB + WB / 2, 38, "B = X (ΣK × N)", 17, INK, weight="bold")
s.text(XD + WD / 2 + 40, 38, "D = dW (G × M × N)", 17, INK, weight="bold")

y = Y0
segs = []
for e, k in experts:
    al = -(-k // BK) * BK
    hv, hp = k * px, (al - k) * px
    for x, w in ((XA, WA), (XB, WB)):
        seg(s, x, y, w, hv, e, f"E{e}" if hv > 20 else None, size=16, halo=True)
        pad(s, x, y + hv, w, hp)
    s.box(XG, y + 1, WG, al * px - 2, str(al), ACC_F, ACC, ACC, 15, "bold")
    segs.append((e, y, al * px))
    y += al * px
YEND = y
# K blocks of 128 tokens across both operands
blocks = [Y0 + i * BK * px for i in range(1, 768 // BK)]
hlines(s, XA, 0, WA, blocks, sw=1.4)
hlines(s, XB, 0, WB, blocks, sw=1.4)
outline(s, XA, Y0, WA, YEND - Y0)
outline(s, XB, Y0, WB, YEND - Y0)
s.text((XA + XB + WB) / 2, YEND + 22, "same token rows", 15, ACC, weight="bold")
mono(s, 6, YEND + 22, "grouped_layout", 15, ACC, weight="bold", anchor="start")
s.text(XA - 6, Y0 + 5, "K", 16, MUTED, anchor="end", weight="bold")

# one output per expert, arrows from the shared token segment
for e, ys, h in segs:
    yd = Y0 + 2 + e * (HD + 8)
    f, st = EXP[e]
    seg(s, XD, yd, WD, HD, e, f"D[{e}]", size=16)
    s.text(XD + WD + 8, yd + HD / 2 + 6, "M × N", 15, MUTED, anchor="start")
    s.line(XB + WB + 4, ys + h / 2, XD - 4, yd + HD / 2, ACC, 2, "arrAcc")
    s.add(f'<circle cx="{XB + WB + 4:.1f}" cy="{ys + h / 2:.1f}" r="3" fill="{ACC}"/>')
mono(s, XD + 60, YEND + 22, "D[e] = C[e] + A_eᵀ · B_e  (FP32)", 15, ACC, weight="bold")
s.save(os.path.join(HERE, "fig_kgroup.svg"))
