"""fig_sched: the M-grouped contiguous tile scheduler. One tile grid over sum(M) x N, persistent CTAs walk it,
each M tile reads its expert from the first row's m_indices entry; B multicast only inside one expert."""
import os
from tilekit import SVG, seg, hlines, vlines, outline, mono, EXP, INK, MUTED, ACC, ACC_F, PRB, GOOD, NEU_F

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 292, "fig_sched: per-tile expert lookup in the persistent scheduler")

TW, TH, X0, Y0 = 46, 36, 150, 48
tile_e = [0, 0, 0, 1, 2, 2]              # expert of each M tile (running example)
NM, NN = len(tile_e), 4

s.text(X0 + NN * TW / 2, 24, "D tiles (ΣM × N)", 18, INK, weight="bold")
mono(s, 96, 38, "e", 16, ACC, weight="bold")
mono(s, 40, 38, "m", 16, MUTED, weight="bold")
order = 0
for m, e in enumerate(tile_e):
    y = Y0 + m * TH
    f, st = EXP[e]
    s.rect(X0, y, NN * TW, TH, f, st, 1.2, rx=0)
    mono(s, 40, y + TH / 2 + 5.5, str(m), 16, MUTED)
    mono(s, 96, y + TH / 2 + 5.5, str(e), 17, ACC, weight="bold")
    s.line(110, y + TH / 2, X0 - 6, y + TH / 2, ACC, 1.6, "arrAcc")
# visit order: down M first inside an N column (swizzle), so the B tile of a column is reused from L2
for n in range(NN):
    for m in range(NM):
        mono(s, X0 + n * TW + TW / 2, Y0 + m * TH + TH / 2 + 5.5, str(n * NM + m), 15, INK)
hlines(s, X0, 0, NN * TW, [Y0 + TH * k for k in range(1, NM)], color="#7F7F7F", dash="none", sw=1)
vlines(s, [X0 + TW * k for k in range(1, NN)], Y0, NM * TH, color="#7F7F7F", dash="none", sw=1)
outline(s, X0, Y0, NN * TW, NM * TH)
s.text(X0 + NN * TW / 2, Y0 + NM * TH + 20, "number = visit order", 15, MUTED, style="italic")

# multicast pairs (m, m^1): share one B tile only if both tiles use the same expert
XP = X0 + NN * TW + 14
s.text(XP + 4, 38, "B multicast", 15, INK, anchor="start", weight="bold")
for k in range(0, NM, 2):
    y1, y2 = Y0 + k * TH + 6, Y0 + (k + 2) * TH - 6
    ok = tile_e[k] == tile_e[k + 1]
    col = ACC if ok else PRB
    s.path(f"M{XP},{y1} L{XP + 8},{y1} L{XP + 8},{y2} L{XP},{y2}", col, 2.2)
    s.text(XP + 18, (y1 + y2) / 2 + 6, "✓ shared" if ok else "✗ E0 ≠ E1", 16, col, anchor="start", weight="bold")

# pseudo-code of one persistent CTA
XC, YC, WC, HC = 456, 44, 404, 214
s.rect(XC, YC, WC, HC, "#F7F7F7", "#BFBFBF", 1.2, rx=6)
s.text(XC + 12, YC - 10, "one CTA per SM (persistent)", 16, INK, anchor="start", weight="bold")
lines = [
    ("for t in range(sm, tiles, SMs):", INK, "normal", "all experts"),
    ("  m, n = swizzle(t)", INK, "normal", "down M first"),
    ("  e = m_indices[m * 128]", ACC, "bold", "expert lookup"),
    ("  if e < 0: continue", INK, "normal", "pure padding"),
    ("  acc = A[m] @ B[e][n].T", INK, "normal", "FP8 MMA"),
    ("  D[m, n] = bf16(acc)", INK, "normal", "TMA store"),
]
for i, (code, col, w, note) in enumerate(lines):
    yy = YC + 32 + i * 33
    if col == ACC:
        s.rect(XC + 6, yy - 20, WC - 12, 28, ACC_F, ACC, 1.2, rx=3)
    mono(s, XC + 14, yy, code.replace(" ", "\u00a0"), 15, col, anchor="start", weight=w)
    s.text(XC + WC - 10, yy, note, 15, MUTED, anchor="end", style="italic")
s.save(os.path.join(HERE, "fig_sched.svg"))
