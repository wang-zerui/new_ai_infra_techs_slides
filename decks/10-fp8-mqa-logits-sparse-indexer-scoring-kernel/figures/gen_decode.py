"""Slide 9 (mechanism 3): SM90 paged decode. The metadata kernel prefix-sums ceil(len/256) splits per request and
gives each SM an equal share (sched/sm90_paged_mqa_logits.cuh); inside a CTA, 4 KV groups (TMA warp + math
warpgroup) each take one 64-key page of a 256-key split (impls/sm90_fp8_paged_mqa_logits.cuh). Sizes illustrative."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 266
s = SVG(W, H, "paged decode split-KV schedule")
reqs = [("R0", 5), ("R1", 2), ("R2", 7)]
segs = [r for r, n in reqs for _ in range(n)]
N, NSM = len(segs), 4
q, r = divmod(N, NSM)
pivot = NSM - r
starts = [i * q + max(0, i - pivot) for i in range(NSM)] + [N]
tints = ["#DEEBF7", "#BDD7EE", "#DEEBF7", "#BDD7EE"]
x0, sw, gap, y0, sh = 30, 54, 4, 28, 44
s.text(0, 18, "requests cut into 256-key splits (4 pages of 64)", 17, INK, anchor="start", weight="bold")
for i, name in enumerate(segs):
    sm = max(k for k in range(NSM) if starts[k] <= i)
    x = x0 + i * (sw + gap)
    s.rect(x, y0, sw, sh, tints[sm], ACC, 1.8, 3)
    for p in range(1, 4):
        s.line(x + p * sw / 4, y0 + 4, x + p * sw / 4, y0 + sh - 4, "#9DB7D5", 1)
    s.text(x + sw / 2, y0 + sh / 2 + 6, name, 16, INK, weight="bold")
# request boundaries
acc = 0
for name, n in reqs[:-1]:
    acc += n
    xb = x0 + acc * (sw + gap) - gap / 2
    s.line(xb, y0 - 6, xb, y0 + sh + 6, INK, 2.4)
# SM brackets
for k in range(NSM):
    a, b = starts[k], starts[k + 1]
    xa, xb = x0 + a * (sw + gap) + 2, x0 + b * (sw + gap) - gap - 2
    yb = y0 + sh + 12
    s.path(f"M{xa},{yb} V{yb + 8} H{xb} V{yb}", ACC, 2)
    s.text((xa + xb) / 2, yb + 28, f"SM{k}", 17, ACC, weight="bold")
s.text(866, 18, "equal shares per SM", 16, MUTED, anchor="end", style="italic")

# zoom: one split inside a CTA
yz = 134
s.line(0, yz - 14, 870, yz - 14, "#BFBFBF", 1.2)
s.text(0, yz + 8, "one split inside a CTA", 17, INK, anchor="start", weight="bold")
for g in range(4):
    x = 0 + g * 112
    s.rect(x, yz + 20, 100, 34, NEU_F, NEU, 2, 3)
    s.text(x + 50, yz + 43, f"page {g}", 16, INK, weight="bold")
    s.rect(x, yz + 72, 100, 46, ACC_F, ACC, 2.2, 5)
    s.text(x + 50, yz + 92, f"KV group {g}", 15, INK, weight="bold")
    s.text(x + 50, yz + 110, "TMA + WG", 15, INK)
    s.arrow(x + 50, yz + 56, x + 50, yz + 69)
s.text(456, yz + 43, "pages located via the block table", 16, MUTED, anchor="start", style="italic")
s.rect(486, yz + 72, 150, 46, "#FFFFFF", ACC, 2.4, 5)
s.text(561, yz + 92, "own logit", 16, INK, weight="bold")
s.text(561, yz + 110, "columns", 16, INK, weight="bold")
s.arrow(450, yz + 95, 483, yz + 95, ACC, 2.4)
s.text(660, yz + 84, "no combine step:", 17, ACC, anchor="start", weight="bold")
s.text(660, yz + 106, "each key is scored", 16, INK, anchor="start")
s.text(660, yz + 126, "independently", 16, INK, anchor="start")
s.save(os.path.join(HERE, "fig_decode.svg"))
