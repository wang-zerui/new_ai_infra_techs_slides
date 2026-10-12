"""fig_split: the 24-bit significand of an FP32 weight split into w_hi (bits 1-8), w_lo (bits 9-16), dropped tail
(bits 17-24), with a worked example (values computed by notes/split_sim.py, round-to-nearest BF16)."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED, rich

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 290
s = SVG(W, H, "fig_split: significand bits 1-8 -> w_hi, 9-16 -> w_lo (after x256), 17-24 dropped")

X0, CW, Y0, CH = 186, 24, 38, 40
seg = [(0, 8, ACC, ACC, "w_hi"), (8, 16, ACC_F, ACC, "w_lo"), (16, 24, "url(#hatchGray)", NEU, "dropped")]

s.text(X0 - 16, Y0 + 18, "FP32 w", 20, INK, anchor="end", weight="bold")
s.text(X0 - 16, Y0 + 38, "significand", 16, MUTED, anchor="end")
for c0, c1, fill, stroke, _ in seg:
    for c in range(c0, c1):
        s.rect(X0 + c * CW, Y0, CW, CH, fill, stroke, 1.2, 0)
for c0, c1, _, _, lab in seg:
    xa, xb = X0 + c0 * CW + 2, X0 + c1 * CW - 2
    s.text((xa + xb) / 2, Y0 - 12, f"bits {c0 + 1}–{c1}", 17, MUTED)

# arrows down to the two BF16 halves
BY, BH = 112, 64
c1, c2, c3 = X0 + 4 * CW, X0 + 12 * CW, X0 + 20 * CW
s.line(c1, Y0 + CH + 2, c1, BY - 2, INK, 2.2, "arr")
s.line(c2, Y0 + CH + 2, c2, BY - 2, INK, 2.2, "arr")
s.box(c1 - 78, BY, 156, BH, ["w_hi", "= bf16(w)"], ACC_F, ACC, color=ACC, weight="bold", size=19)
s.box(c2 - 98, BY, 196, BH, ["w_lo", "= bf16(256 · r)"], ACC_F, ACC, color=ACC, weight="bold", size=19)
s.text(c2, BY + BH + 22, "r = w − w_hi;  ×256 is exact", 17, MUTED, anchor="middle", style="italic")
# dropped tail
s.text(c3, BY + 24, "tail lost", 18, INK)
rich(s, c3, BY + 54, [("≤ 2", {}), ("−16", {"sup": True}), (" · |w|", {})], size=19, anchor="middle")

# worked example panel (values: notes/sources.md, split_sim.py)
PY = 206
s.rect(8, PY, 854, 78, "#FFFFFF", NEU, 1.5, 6, dash="5,3")
s.text(22, PY + 46, "example", 18, MUTED, anchor="start", weight="bold")
mono = "Menlo, 'Courier New', monospace"
cols = [(116, "w = 0.1234567", "w_lo = −0.02014", False),
        (350, "w_hi = 0.1235352", "w_hi + w_lo/256 = 0.1234565", True),
        (660, "r = −7.85e−5", "error = 2.2e−7", False)]
for x, l1, l2, hl in cols:
    s.text(x, PY + 30, l1, 17, INK, anchor="start", family=mono)
    s.text(x, PY + 60, l2, 17, ACC if hl else INK, anchor="start", family=mono, weight="bold" if hl else "normal")
s.save(os.path.join(HERE, "fig_split.svg"))
