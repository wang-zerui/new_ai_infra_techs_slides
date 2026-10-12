"""fig_zero: (a) ZeRO-1 + AdamW: flattened params cut into equal DP slices, matrices A/B/C cut across ranks, element-wise
update works on any slice. (b) ZeRO-1 + Muon (Moonlight Distributed Muon, Alg. 1): every rank gathers the full matrix B,
runs Newton-Schulz on all of it, keeps only its slice."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED, GOOD

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 292
s = SVG(W, H, "fig_zero")
LY0, LH, LG = 66, 36, 14
lanes = [LY0 + i * (LH + LG) for i in range(4)]
BF = "#D9D9D9"   # matrix B tint

# ---------------- (a)
s.text(200, 22, "(a) ZeRO-1 + AdamW", 20, INK, weight="bold")
s.text(130, 52, "own slice of flat params", 17, MUTED)
U = 10   # pt per element unit; slice = 16 units
segs = [[("A", 16, "#FFFFFF")], [("A", 6, "#FFFFFF"), ("B", 10, BF)], [("B", 16, BF)], [("B", 4, BF), ("C", 12, "#FFFFFF")]]
for i, y in enumerate(lanes):
    s.text(4, y + LH / 2 + 6.5, f"R{i}", 19, INK, anchor="start", weight="bold")
    x = 50
    for name, n, f in segs[i]:
        s.rect(x, y, n * U, LH, f, NEU, 1.6, 0)
        s.text(x + n * U / 2, y + LH / 2 + 6.5, name, 18, INK)
        x += n * U
    s.line(214, y + LH / 2, 236, y + LH / 2, INK, 2, "arr")
    s.box(238, y, 92, LH, "AdamW", NEU_F, NEU, size=18)
    s.text(352, y + LH / 2 + 8, "✓", 24, GOOD, weight="bold")
s.text(200, 280, "element-wise: any slice works", 19, GOOD, weight="bold")

s.line(410, 10, 410, H - 10, "#BFBFBF", 1.2)

# ---------------- (b)
s.text(645, 22, "(b) ZeRO-1 + Muon (gather)", 20, INK, weight="bold")
heads = [(486, "own"), (600, "gathered B"), (727, "NS"), (816, "keep")]
for x, t in heads:
    s.text(x, 52, t, 17, MUTED)
for i, y in enumerate(lanes):
    s.text(424, y + LH / 2 + 6.5, f"R{i}", 19, INK, anchor="start", weight="bold")
    s.box(464, y, 44, LH, f"b{i}", BF, NEU, size=18)
    s.line(510, y + LH / 2, 536, y + LH / 2, INK, 2, "arr")
    for k in range(4):
        s.rect(540 + k * 30, y, 30, LH, BF if k == i else PRB_F, "#B0B0B0", 1.2, 0)
        s.text(540 + k * 30 + 15, y + LH / 2 + 6, f"b{k}", 15, INK)
    s.rect(540, y, 120, LH, "none", PRB, 1.6, 0)
    s.line(662, y + LH / 2, 690, y + LH / 2, INK, 2, "arr")
    s.box(692, y, 70, LH, "NS", PRB_F, PRB, color=PRB, weight="bold", size=18)
    s.line(764, y + LH / 2, 790, y + LH / 2, INK, 2, "arr")
    s.box(794, y, 44, LH, f"b{i}", BF, NEU, size=18)
s.text(645, 280, "every rank orthogonalizes all of B", 19, PRB, weight="bold")
s.save(os.path.join(HERE, "fig_zero.svg"))
