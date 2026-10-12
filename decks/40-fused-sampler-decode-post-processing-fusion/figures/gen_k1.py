"""fig_k1: K1 fused_scan_topk_kernel. A logits row is split over N blocks (N = SMs/B clamped to [8, Nmax]); inside a
1024-thread block, 1016 loader threads load 4 logits each per chunk, apply the penalty bit and 1/T in registers,
update online max/sum (softmax-before-top-k policy only), then a block radix sort puts the rolling top-32 back into
8 keeper threads. Example K = 32. Schematic."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 300
s = SVG(W, H, "fig_k1")

# vocabulary row split over N blocks
X0, X1, VY, VH = 170, 862, 10, 36
s.text(X0 - 10, VY + 16, "one logits row", 17, INK, anchor="end", weight="bold")
s.text(X0 - 10, VY + 36, "V = 120,832", 16, MUTED, anchor="end")
n = 8
sw = (X1 - X0) / n
for i in range(n):
    hl = i == 2
    s.box(X0 + i * sw, VY, sw, VH, f"blk {i}" if i < n - 1 else "blk N−1", ACC_F if hl else "#FFFFFF",
          ACC if hl else NEU, size=16, sw=1.5, rx=0)
s.text(X1, VY + VH + 20, "blocks per row N = SMs ÷ B within [8, 32] for K = 32", 16, MUTED, anchor="end", style="italic")
# zoom lines
BX0, BY0, BX1, BY1 = 170, 88, 862, 294
s.line(X0 + 2 * sw, VY + VH, BX0 + 120, BY0, MUTED, 1.2, dash="4,3")
s.line(X0 + 3 * sw, VY + VH, BX0 + 330, BY0, MUTED, 1.2, dash="4,3")
s.rect(BX0, BY0, BX1 - BX0, BY1 - BY0, "#FFFFFF", ACC, 1.6, 6, dash="6,4")
s.text(10, BY0 + 24, "one block", 18, ACC, anchor="start", weight="bold")
s.text(10, BY0 + 46, "1024 threads", 16, MUTED, anchor="start")

# row 1: per-chunk steps
RY, RH = 134, 52
steps = [(186, 168, ["1016 loaders", "load 4 logits each"]), (380, 150, ["penalty bit", "× 1/T"]),
         (556, 136, ["online", "max + sum"]), (718, 132, ["block radix", "sort"])]
for x, w, lab in steps:
    s.box(x, RY, w, RH, lab, ACC_F, ACC, size=17)
for (x, w, _), (x2, _, _) in zip(steps[:-1], steps[1:]):
    s.line(x + w + 2, RY + RH / 2, x2 - 2, RY + RH / 2, INK, 2.2, "arr")
s.text(380 + 75, RY + RH + 18, "in registers", 15, MUTED, style="italic")
s.text(556 + 68, RY + RH + 18, "softmax-first only", 15, MUTED, style="italic")
# loop: next chunk
xs4 = 718 + 66
xs1 = 186 + 84
s.path(f"M{xs4},{RY - 2} L{xs4},{RY - 16} L{xs1},{RY - 16} L{xs1},{RY - 2}", INK, 2, "arr")
s.text((xs1 + xs4) / 2, RY - 22, "next chunk of the slice", 16, INK)
# row 2: keepers
KY = 228
s.box(650, KY, 200, 52, ["8 keepers", "rolling top-32"], ACC, ACC, color="#FFFFFF", weight="bold", size=17)
s.line(xs4 - 18, RY + RH + 2, xs4 - 18, KY - 2, ACC, 2.2, "arrAcc")
s.line(xs4 + 18, KY - 2, xs4 + 18, RY + RH + 2, ACC, 2.2, "arrAcc")
# output
s.line(648, KY + 26, 470, KY + 26, INK, 2.2, "arr")
s.text(462, KY + 22, "after the last chunk:", 16, MUTED, anchor="end", style="italic")
s.text(462, KY + 44, "32 candidates + max/sum → K2", 17, INK, anchor="end")
s.save(os.path.join(HERE, "fig_k1.svg"))
