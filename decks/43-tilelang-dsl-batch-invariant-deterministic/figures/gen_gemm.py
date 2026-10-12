"""fig_gemm: the mHC matmul (N = 24, K = n_hc*d = 28,672 for V4-Pro) still needs split-K at small batch.
Default DeepGEMM picks the split count per call; deterministic mode fixes 16 splits, writes each split's partial
output separately and reduces them in index order in a later kernel (DeepSeek-V4 §3.3; DeepGEMM mega_mhc.hpp)."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 276
s = SVG(W, H, "fig_gemm")

BX, BW = 190, 480
CW = BW / 16

# ---- row (a): split count chosen per call
s.text(4, 30, "(a) Default", 18, INK, anchor="start", weight="bold")
s.text(4, 52, "per-call splits", 16, MUTED, anchor="start")
s.rect(BX, 14, BW, 32, NEU_F, NEU, 1.8, 3)
for x in (BX + 110, BX + 250, BX + 340):
    s.line(x, 10, x, 50, PRB, 2.2, dash="5,4")
s.text(BX + 55, 36, "?", 18, PRB, weight="bold")
s.text(BX + 180, 36, "?", 18, PRB, weight="bold")
s.text(BX + 295, 36, "?", 18, PRB, weight="bold")
s.text(BX + 410, 36, "?", 18, PRB, weight="bold")
s.text(BX + BW + 14, 30, "split count depends", 16, PRB, anchor="start", weight="bold")
s.text(BX + BW + 14, 50, "on batch T", 16, PRB, anchor="start", weight="bold")

s.line(4, 70, 866, 70, "#BFBFBF", 1.2)

# ---- row (b): fixed 16 splits + ordered reduce
s.text(4, 112, "(b) Deterministic", 18, ACC, anchor="start", weight="bold")
s.text(4, 134, "fixed 16 splits", 16, MUTED, anchor="start")
s.text(BX + BW / 2, 90, "reduction length K = 28,672 (V4-Pro mHC)", 16, MUTED, style="italic")
for i in range(16):
    x = BX + i * CW
    s.box(x, 98, CW, 32, str(i), ACC_F, ACC, ACC, 15, sw=1.2, rx=0)
    s.line(x + CW / 2, 132, x + CW / 2, 156, ACC, 1.6, "arrAcc")
    lab = f"P{i}" if i in (0, 1, 15) else ("…" if i == 8 else "")
    s.box(x + 1, 158, CW - 2, 26, lab, "#FFFFFF", ACC, INK, 15, sw=1.2, rx=2)
s.text(BX + BW + 14, 176, "one buffer per split", 16, ACC, anchor="start", style="italic")
# bracket + arrow to the reduce kernel
s.path(f"M{BX + 2},{190} L{BX + 2},{196} L{BX + BW - 2},{196} L{BX + BW - 2},{190}", ACC, 1.8)
s.line(BX + BW / 2, 197, BX + BW / 2, 224, ACC, 2.2, "arrAcc")
s.box(BX, 226, BW, 40, "reduce kernel: P0 + P1 + … + P15 in index order", ACC_F, ACC, INK, 17, sw=2)
s.line(BX + BW + 2, 246, BX + BW + 62, 246, INK, 2.2, "arr")
s.box(BX + BW + 64, 226, 112, 40, "y  [T × 24]", "#FFFFFF", NEU, INK, 17, weight="bold", sw=1.8)
s.save(os.path.join(HERE, "fig_gemm.svg"))
