"""fig_chain_1 / fig_chain_2: the MLA pre-processing chain on an Ascend AI core.

fig_chain_1 — op-by-op execution: each operator uses either the Cube unit (matmul) or the
Vector unit (element-wise), and every intermediate round-trips through HBM.
fig_chain_2 — the same work as one MLAPO super operator: intermediates stay in the on-chip
buffer and the two units pipeline over tiles.

The op chain is vLLM-Ascend's MLA pre-processing path (mla_v1.py unfused path + the argument
list of the fused `mla_prolog` call). The GLM-5 report says MLAPO fuses 13 small operators but
does not list them. Box widths are illustrative, not measurements.
"""
import os
from svgkit import SVG, ACC, ACC_F, PRB, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))

W, H = 870, 296
X0, X1 = 150, 845
YC, YV, BH = 58, 128, 48          # Cube lane top, Vector lane top, box height
YH, HH = 226, 38                  # HBM band
GAP = 6
FS = 16

OPS = [  # (lane, label lines)
    ("v", ["quantize", "x"]),
    ("c", ["q down", "proj"]),
    ("v", ["RMSNorm"]),
    ("c", ["q up", "proj"]),
    ("v", ["RoPE", "(q)"]),
    ("c", ["absorb", "W_UKᵀ"]),
    ("c", ["kv down", "proj"]),
    ("v", ["RMSNorm", "RoPE (k)"]),
    ("v", ["KV-cache", "write"]),
]


def widths():
    raw = [8.2 * max(len(t) for t in lab) + 18 for _, lab in OPS]
    avail = (X1 - X0) - GAP * (len(OPS) - 1)
    k = avail / sum(raw)
    return [w * k for w in raw]


def lanes(s, hbm_label, note=None):
    s.text(142, YC + BH / 2 + 6, "Cube unit", 18, INK, anchor="end", weight="bold")
    s.text(142, YC + BH / 2 + 26, "(matmul)", 15, MUTED, anchor="end", style="italic")
    s.text(142, YV + BH / 2 + 6, "Vector unit", 18, INK, anchor="end", weight="bold")
    s.text(142, YV + BH / 2 + 26, "(element-wise)", 15, MUTED, anchor="end", style="italic")
    s.rect(X0, YH, X1 - X0, HH, NEU_F, NEU, 2, 4)
    s.text(142, YH + HH / 2 + 6, "HBM", 18, INK, anchor="end", weight="bold")
    if hbm_label:
        s.text((X0 + X1) / 2, YH + HH / 2 + 6, hbm_label[0], 17, hbm_label[1])
    if note:
        s.text(X0, 28, note, 17, MUTED, anchor="start", style="italic")
    s.line(X0, 286, X1, 286, INK, 2, "arr")
    s.text(X1, 280, "time", 17, INK, anchor="end")


# ------------------------------------------------------------------ fig_chain_1
s = SVG(W, H, "fig_chain_1: MLA pre-processing as a chain of small operators")
lanes(s, ("every intermediate written, then read back", PRB),
      note="13 small operators in all; 9 shown")
x = X0
for (lane, label), w in zip(OPS, widths()):
    y = YC if lane == "c" else YV
    s.box(x, y, w, BH, label, NEU_F if lane == "c" else "#FFFFFF", NEU, size=FS)
    s.line(x + w / 2, y + BH + 3, x + w / 2, YH - 3, PRB, 2.2, "arrPrb", start_marker="arrPrb")
    if lane == "v":                       # the Cube unit is idle above this op
        s.rect(x, YC, w, BH, "#FFFFFF", "#D9D9D9", 1.6, 4, dash="5,4")
    else:                                 # the Vector unit is idle below this op
        s.rect(x, YV, w, BH, "#FFFFFF", "#D9D9D9", 1.6, 4, dash="5,4")
    x += w + GAP
# point the annotation at one empty (dashed) slot on the Cube lane
ws = widths()
gapx = X0 + sum(ws[:4]) + GAP * 4 + ws[4] / 2
s.path(f"M {gapx + 128} 26 L {gapx + 2} {YC + BH / 2 - 2}", PRB, 2.4, "arrPrb")
s.text(gapx + 136, 24, "the other unit idles", 17, PRB, anchor="start", weight="bold")
s.save(os.path.join(HERE, "fig_chain_1.svg"))

# ------------------------------------------------------------------ fig_chain_2
s = SVG(W, H, "fig_chain_2: the same chain as one MLAPO super operator")
lanes(s, None)
CT, CB = YC - 24, YV + BH + 16
s.rect(X0, CT, X1 - X0, CB - CT, ACC_F, ACC, 3, 6)
s.text(X0 + 14, CT + 18, "MLAPO super operator  ·  one kernel", 19, ACC, anchor="start", weight="bold")
s.text(X1 - 14, CT + 18, "intermediates stay on chip", 17, ACC, anchor="end", style="italic")
TW = (X1 - X0 - 170) / 3
for i in range(3):
    xc = X0 + 28 + i * (TW + 12)
    s.box(xc, YC, TW, BH, [f"GEMM tile {i + 1}"], "#FFFFFF", ACC, color=INK, size=FS)
    xv = xc + TW / 2 + 6
    s.box(xv, YV, TW, BH, ["norm · quant · RoPE", f"tile {i + 1}"], "#FFFFFF", ACC, color=INK, size=15)
    s.line(xc + TW - 12, YC + BH + 3, xv + 20, YV - 3, ACC, 2.2, "arrAcc")
# the two HBM transfers that remain
s.line(X0 + 70, YH - 3, X0 + 70, CB + 6, ACC, 2.6, "arrAcc")
s.text(X0 + 82, YH - 8, "read x + weights", 17, INK, anchor="start")
s.line(X1 - 70, CB + 6, X1 - 70, YH - 3, ACC, 2.6, "arrAcc")
s.text(X1 - 82, YH - 8, "write q + KV-cache row", 17, INK, anchor="end")
s.save(os.path.join(HERE, "fig_chain_2.svg"))
