"""fig_sched_masked: the persistent masked scheduler. Each group owns ceil(masked_m[g] / BLOCK_M) x N-blocks tiles,
read from device memory at run time; tiles past masked_m are never scheduled; the launch grid is always #SMs."""
import os
from tilekit import SVG, outline, mono, EXP, INK, MUTED, ACC, ACC_F

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 292, "fig_sched_masked: device-side masked_m drives the tile walk")

BM, NT_M, NN = 128, 4, 2
mm = [300, 60, 200]
TW, TH, Y0 = 42, 38, 70
XS = [78, 198, 318]

s.text(198 + NN * TW / 2, 22, "D tiles per expert slot", 18, INK, weight="bold")
order = 0
for g, (x, m) in enumerate(zip(XS, mm)):
    f, st = EXP[g]
    nm = -(-m // BM)
    s.box(x + 8, Y0 - 32, NN * TW - 16, 22, str(m), ACC_F, ACC, ACC, 15, "bold")
    for n in range(NN):
        for mi in range(NT_M):
            tx, ty = x + n * TW, Y0 + mi * TH
            if mi < nm:
                s.rect(tx, ty, TW, TH, f, st, 1.2, rx=0)
            else:
                s.rect(tx, ty, TW, TH, "#F7F7F7", "#BFBFBF", 1, rx=0, dash="3,3")
    for n in range(NN):
        for mi in range(nm):
            mono(s, x + n * TW + TW / 2, Y0 + mi * TH + TH / 2 + 5.5, str(order), 15, INK)
            order += 1
    outline(s, x, Y0, NN * TW, NT_M * TH, INK, 1.8)
    s.text(x + NN * TW / 2, Y0 + NT_M * TH + 22, f"E{g}: {nm} M tile" + ("s" if nm > 1 else ""), 15, INK, weight="bold")
mono(s, 40, Y0 - 16, "masked_m", 15, ACC, weight="bold")
s.text(198 + NN * TW / 2, Y0 + NT_M * TH + 46, "number = visit order · dashed = skipped", 15, MUTED, style="italic")

# pseudo-code of one persistent CTA
XC, YC, WC, HC = 470, 46, 392, 182
s.rect(XC, YC, WC, HC, "#F7F7F7", "#BFBFBF", 1.2, rx=6)
s.text(XC + 10, YC - 10, "grid = #SMs at every step", 16, INK, anchor="start", weight="bold")
lines = [
    ("for t in range(sm, total, SMs):", INK, "normal", "persistent"),
    ("  g, m, n = locate(t, masked_m)", ACC, "bold", "GPU read"),
    ("  if g == G: break", INK, "normal", "done"),
    ("  acc = A[g][m] @ B[g][n].T", INK, "normal", "FP8 MMA"),
    ("  D[g][m, n] = bf16(acc)", INK, "normal", "TMA store"),
]
for i, (code, col, w, note) in enumerate(lines):
    yy = YC + 32 + i * 33
    if col == ACC:
        s.rect(XC + 6, yy - 20, WC - 12, 28, ACC_F, ACC, 1.2, rx=3)
    mono(s, XC + 12, yy, code.replace(" ", " "), 15, col, anchor="start", weight=w)
    s.text(XC + WC - 8, yy, note, 15, MUTED, anchor="end", style="italic")
s.text(XC + 4, YC + HC + 26, "locate: group g owns ceil(masked_m[g] / 128) M tiles", 15, INK, anchor="start")
s.save(os.path.join(HERE, "fig_sched_masked.svg"))
