"""fig_ksched: K-grouped scheduler. Every expert has the same M x N tile grid; the CTA that takes tile (g, m, n)
loops only over expert g's 128-token K blocks and accumulates into FP32."""
import os
from tilekit import SVG, outline, mono, EXP, INK, MUTED, ACC, ACC_F

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 292, "fig_ksched: same tile grid per expert, own K range")

kblocks = [3, 1, 2]                 # 128-token K blocks of E0, E1, E2 (running example)
TW, TH, X0, Y0, GAP = 40, 34, 66, 44, 16
s.text(X0 + TW, 26, "M × N tiles", 17, INK, weight="bold")
s.text(250, 26, "K loop of each tile", 17, INK, weight="bold")
order = 0
for g, nk in enumerate(kblocks):
    y = Y0 + g * (2 * TH + GAP)
    f, st = EXP[g]
    s.text(X0 - 10, y + TH + 6, f"E{g}", 17, INK, anchor="end", weight="bold")
    for n in range(2):
        for m in range(2):
            s.rect(X0 + n * TW, y + m * TH, TW, TH, f, st, 1.2, rx=0)
            mono(s, X0 + n * TW + TW / 2, y + m * TH + TH / 2 + 5.5, str(order), 15, INK)
            order += 1
    outline(s, X0, y, 2 * TW, 2 * TH, INK, 1.8)
    s.line(X0 + 2 * TW + 6, y + TH, 186, y + TH, ACC, 2, "arrAcc")
    for b in range(nk):
        s.rect(192 + b * 44, y + TH - 16, 40, 32, f, ACC, 2, rx=3)
        mono(s, 192 + b * 44 + 20, y + TH + 5, f"k{b}", 15, ACC, weight="bold")
    s.text(192 + nk * 44 + 8, y + TH + 6, f"{nk} × 128 tokens", 15, MUTED, anchor="start")

# pseudo-code of one persistent CTA
XC, YC, WC, HC = 440, 44, 420, 214
s.rect(XC, YC, WC, HC, "#F7F7F7", "#BFBFBF", 1.2, rx=6)
s.text(XC + 10, YC - 10, "one CTA per SM (persistent)", 16, INK, anchor="start", weight="bold")
lines = [
    ("for t in range(sm, G*tiles, SMs):", INK, "normal", "all experts"),
    ("  g, m, n = split(t)", INK, "normal", "same grid"),
    ("  acc = C[g][m, n]", INK, "normal", "fused accum"),
    ("  for kb in k_blocks(g):", ACC, "bold", "own tokens"),
    ("    acc += A[kb,m].T @ B[kb,n]", INK, "normal", "FP8 MMA"),
    ("  D[g][m, n] = acc", INK, "normal", "FP32 store"),
]
for i, (code, col, w, note) in enumerate(lines):
    yy = YC + 30 + i * 33
    if col == ACC:
        s.rect(XC + 6, yy - 20, WC - 12, 28, ACC_F, ACC, 1.2, rx=3)
    mono(s, XC + 12, yy, code.replace(" ", " "), 15, col, anchor="start", weight=w)
    s.text(XC + WC - 8, yy, note, 15, MUTED, anchor="end", style="italic")
s.text(XC + 4, YC + HC + 24, "k_blocks(g) from grouped_layout · start aligned to 128", 15, INK, anchor="start")
s.save(os.path.join(HERE, "fig_ksched.svg"))
