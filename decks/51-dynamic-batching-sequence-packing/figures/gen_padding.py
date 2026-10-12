"""fig_padding: (a) padding the 4K/8K/32K batch to 32K; (b) naive concatenation (S1..S3 = 4, 2, 3 tokens):
causal attention leaks across sample boundaries and positions keep counting."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 284, "fig_padding")

# ---- (a) padding
s.text(10, 26, "(a) Pad to the longest", 20, INK, anchor="start", weight="bold")
X0, W = 82, 330
def x(k):
    return X0 + W * k / 32
for i, (name, k) in enumerate([("4K", 4), ("8K", 8), ("32K", 32)]):
    y = 52 + i * 58
    s.text(X0 - 10, y + 26, name, 18, INK, anchor="end", weight="bold")
    s.rect(X0, y, x(k) - X0, 38, NEU_F, NEU, 2, 2)
    if k < 32:
        s.rect(x(k), y, X0 + W - x(k), 38, "url(#hatchPrb)", PRB, 2, 2)
        s.text((x(k) + X0 + W) / 2, y + 26, "pad", 18, PRB, weight="bold", halo=True)
s.text(X0 + W / 2, 252, "rectangle [3, 32K]", 17, MUTED, style="italic")

s.line(445, 10, 445, 274, "#BFBFBF", 1.5)

# ---- (b) naive concatenation: 9 x 9 causal mask, sizes 4, 2, 3
s.text(466, 26, "(b) Concatenate naively", 20, INK, anchor="start", weight="bold")
sizes = [4, 2, 3]
seg = []
for i, n in enumerate(sizes):
    seg += [i] * n
GX, GY, C = 560, 52, 21
toks = list("ABCDEFGHI")
for r in range(9):
    s.text(GX - 8, GY + r * C + 15.5, toks[r], 15, INK, anchor="end")
    s.text(GX + r * C + C / 2, GY - 8, toks[r], 15, INK)
    for c in range(9):
        if c <= r:
            same = seg[r] == seg[c]
            fill, stroke = (NEU_F, NEU) if same else (PRB_F, PRB)
        else:
            fill, stroke = "#FFFFFF", "#D9D9D9"
        s.rect(GX + c * C, GY + r * C, C - 2, C - 2, fill, stroke, 1.2, 1)
# positions
s.text(GX - 8, GY + 9 * C + 24, "pos", 15, MUTED, anchor="end")
for c in range(9):
    s.text(GX + c * C + C / 2 - 1, GY + 9 * C + 24, str(c), 15, PRB if seg[c] > 0 else INK)
# legend
LX = 770
s.rect(LX, 70, 20, 16, NEU_F, NEU, 1.4, 2)
s.text(LX + 26, 84, "within", 16, INK, anchor="start")
s.text(LX + 26, 102, "sample", 16, INK, anchor="start")
s.rect(LX, 130, 20, 16, PRB_F, PRB, 1.4, 2)
s.text(LX + 26, 144, "across", 16, PRB, anchor="start", weight="bold")
s.text(LX + 26, 162, "samples", 16, PRB, anchor="start", weight="bold")
s.save(os.path.join(HERE, "fig_padding.svg"))
