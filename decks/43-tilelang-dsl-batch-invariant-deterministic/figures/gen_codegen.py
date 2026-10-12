"""fig_codegen: TileLang Host Codegen (DeepSeek-V4 §3.2): the per-call contract checks and argument marshaling move
from Python into a host launcher co-generated with the device kernel from the same IR (TVM-FFI)."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 262
s = SVG(W, H, "fig_codegen")

X_CALL, W_CALL = 186, 140
X_MID, W_MID = 366, 270
X_K, W_K = 676, 180
BH = 60


def row(y, name, mid_lines, mid_kind):
    s.text(4, y + 26, name[0], 18, ACC if mid_kind == "acc" else INK, anchor="start", weight="bold")
    s.text(4, y + 48, name[1], 18, ACC if mid_kind == "acc" else INK, anchor="start", weight="bold")
    s.box(X_CALL, y, W_CALL, BH, ["Python", "kernel call"], NEU_F, NEU, INK, 17, sw=1.8)
    if mid_kind == "prb":
        s.box(X_MID, y, W_MID, BH, mid_lines, PRB_F, PRB, INK, 17, sw=2)
    else:
        s.box(X_MID, y, W_MID, BH, mid_lines, ACC_F, ACC, INK, 17, sw=2)
    s.box(X_K, y, W_K, BH, "device kernel", "#FFFFFF", NEU, INK, 17, sw=1.8)
    s.line(X_CALL + W_CALL + 2, y + BH / 2, X_MID - 2, y + BH / 2, INK, 2.2, "arr")
    s.line(X_MID + W_MID + 2, y + BH / 2, X_K - 2, y + BH / 2, INK, 2.2, "arr")


row(10, ("(a) Python", "wrapper"), ["checks + marshaling", "in Python per call"], "prb")
s.line(4, 92, 866, 92, "#BFBFBF", 1.2)
row(110, ("(b) Host", "codegen"), ["checks + marshaling in", "generated C++ on TVM-FFI"], "acc")

# shared IR chip, co-generating launcher and kernel
s.box(X_MID, 216, X_K + W_K - X_MID, 40, "TileLang IR with dtype · shape · stride metadata", "#FFFFFF", ACC, ACC, 17,
      weight="bold", sw=1.8, dash="6,4")
for x in (X_MID + W_MID / 2, X_K + W_K / 2):
    s.line(x, 214, x, 172, ACC, 2.2, "arrAcc", dash="6,4")
s.text(X_MID + W_MID / 2 + 10, 199, "co-generated", 16, ACC, anchor="start", style="italic", halo=True)
s.save(os.path.join(HERE, "fig_codegen.svg"))
