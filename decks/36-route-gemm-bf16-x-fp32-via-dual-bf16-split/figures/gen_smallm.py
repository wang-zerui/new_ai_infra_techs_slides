"""fig_smallm: (a) swap-AB: the 64-row side of a 64xNx16 WGMMA takes 64 weight rows instead of 16 tokens;
(b) split-K: M=16, N=192, K=4096 -> 3 output tiles x 8 K chunks = 24 CTAs, FP32 partials, per-tile counter,
in-kernel reduction (config from src/gemm/entry.cc select_config). Schematic."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 300
s = SVG(W, H, "fig_smallm")

# ---------------- (a) swap-AB
s.text(8, 22, "(a) swap-AB", 20, INK, anchor="start", weight="bold")
TOP, HH = 56, 160            # 64 rows -> 160 pt (2.5 pt per row)
# left: tokens as operand A
xa = 52
s.rect(xa, TOP, 96, 16 * 2.5, ACC_F, ACC, 2, 0)
s.rect(xa, TOP + 40, 96, 48 * 2.5, "url(#hatchPrb)", PRB, 2, 0)
s.text(xa + 48, TOP + 26, "16 tokens", 17, ACC, weight="bold")
s.text(xa + 48, TOP + 105, "48 idle", 17, PRB, weight="bold", halo=True)
s.text(xa + 48, TOP + 125, "rows", 17, PRB, weight="bold", halo=True)
s.text(xa + 48, TOP + HH + 26, "x as operand A", 17, INK)
# brace for 64 rows
s.path(f"M{xa - 8},{TOP} L{xa - 14},{TOP} L{xa - 14},{TOP + HH} L{xa - 8},{TOP + HH}", MUTED, 1.5)
s.text(xa - 18, TOP + HH / 2 + 6, "64", 17, MUTED, anchor="end")
# arrow
s.line(166, TOP + HH / 2, 208, TOP + HH / 2, ACC, 2.5, "arrAcc")
# right: weights as operand A
xb = 228
s.rect(xb, TOP, 40, HH, ACC_F, ACC, 2, 0)
s.text(xb + 52, TOP + 28, "64 weight", 17, ACC, anchor="start", weight="bold")
s.text(xb + 52, TOP + 50, "rows", 17, ACC, anchor="start", weight="bold")
s.text(xb + 52, TOP + HH - 30, "16 tokens", 17, INK, anchor="start")
s.text(xb + 52, TOP + HH - 8, "as N side", 17, INK, anchor="start")
s.text(xb + 70, TOP + HH + 26, "Yᵀ = W · xᵀ", 17, INK)

# divider
s.line(398, 12, 398, H - 8, "#BFBFBF", 1)

# ---------------- (b) split-K
X0 = 420
s.text(X0, 22, "(b) split-K, in-kernel reduce", 20, INK, anchor="start", weight="bold")
s.text(X0, 50, "M = 16 · N = 192 · K = 4096", 17, MUTED, anchor="start")
GX, GY, CWd, CHt = X0 + 96, 62, 32, 30
for r in range(3):
    s.text(GX - 10, GY + r * CHt + 21, f"n{r}", 17, MUTED, anchor="end")
    for c in range(8):
        s.rect(GX + c * CWd, GY + r * CHt, CWd - 3, CHt - 3, ACC_F, ACC, 1.5, 2)
s.text(GX + 4 * CWd, GY + 3 * CHt + 20, "8 K chunks", 17, INK)
s.text(X0, GY + 40, "3 tiles", 17, INK, anchor="start")
s.text(X0, GY + 62, "× 8", 17, INK, anchor="start")
s.text(GX + 8 * CWd + 10, GY + 40, "24", 19, ACC, anchor="start", weight="bold")
s.text(GX + 8 * CWd + 10, GY + 62, "CTAs", 17, ACC, anchor="start", weight="bold")
# reduction path
RY = 178
s.line(GX + 4 * CWd, RY, GX + 4 * CWd, RY + 14, INK, 2.2, "arr")
s.box(X0 + 10, RY + 16, 430, 38, "FP32 partials + per-tile counter", "#FFFFFF", NEU, size=17)
s.line(GX + 4 * CWd, RY + 56, GX + 4 * CWd, RY + 70, INK, 2.2, "arr")
s.box(X0 + 10, RY + 72, 430, 38, "counter = 8 → reduce in the same kernel", ACC_F, ACC, color=ACC, weight="bold", size=17)
s.save(os.path.join(HERE, "fig_smallm.svg"))
