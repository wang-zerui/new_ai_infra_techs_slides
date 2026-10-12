"""fig_gqa: (a) width-matched MHA at 5120 hidden = 40 heads of 128 (5120/128), one KV head per query head;
(b) GLM-4.5: 96 query heads in 8 groups of 12, each group sharing one KV head (GQA-8). Every Q->KV edge drawn.

Config: GLM-4.5 Table 1 (96 heads, 8 KV heads, head dim 128); HF config num_key_value_heads 8.
"""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 288
s = SVG(W, H, "fig_gqa: width-matched MHA vs GLM-4.5 GQA-8 with 96 query heads")

X0, P, RW, GAP = 46, 7.6, 5.6, 10
RH = 22


def head(x, y, fill, stroke):
    s.rect(x, y, RW, RH, fill, stroke, 1.2, rx=1)


# ---------------- (a) width-matched MHA
s.text(0, 18, "(a) Width-matched layout: 5120 / 128 = 40 heads", 18, INK, anchor="start", weight="bold")
QA, KA = 28, 72
s.text(0, QA + RH / 2 + 6, "Q", 18, INK, anchor="start", weight="bold")
s.text(0, KA + RH / 2 + 6, "KV", 18, INK, anchor="start", weight="bold")
for i in range(40):
    x = X0 + i * P
    head(x, QA, NEU_F, NEU)
    head(x, KA, NEU, NEU)
    s.line(x + RW / 2, QA + RH, x + RW / 2, KA, MUTED, 1.0)
xr = X0 + 40 * P + 30
s.text(xr, QA + RH / 2 + 6, "query width 5,120", 18, INK, anchor="start")
s.text(xr, KA + RH / 2 + 6, "cached K and V width 5,120 each", 18, INK, anchor="start")

# ---------------- (b) GLM-4.5 GQA-8
s.text(0, 128, "(b) GLM-4.5: 96 query heads, 8 KV heads", 18, ACC, anchor="start", weight="bold")
QB, KB = 138, 202
s.text(0, QB + RH / 2 + 6, "Q", 18, INK, anchor="start", weight="bold")
s.text(0, KB + RH / 2 + 6, "KV", 18, INK, anchor="start", weight="bold")
gw = 12 * P - (P - RW)
for g in range(8):
    gx = X0 + g * (12 * P + GAP)
    kx = gx + gw / 2 - RW / 2
    for j in range(12):
        x = gx + j * P
        head(x, QB, ACC_F, ACC)
        s.line(x + RW / 2, QB + RH, kx + RW / 2, KB, ACC, 0.9)
    head(kx, KB, ACC, ACC)
s.text(X0, 262, "query width 96 × 128 = 12,288", 18, INK, anchor="start")
s.text(X0, 284, "cached K and V width 8 × 128 = 1,024 each", 18, INK, anchor="start")
s.text(W - 2, 262, "12 query heads per KV head", 18, ACC, anchor="end", style="italic")

s.save(os.path.join(HERE, "fig_gqa.svg"))
