"""fig_kscale: what follows the token axis. (a) FP8 scales: one per channel per 128 tokens, rows grouped per expert
and packed per expert (UE8M0 on SM100); (b) psum layout: one end offset per expert on the GPU, next start aligned
to 128; (c) SM90: per-expert K-major blocks, the CTA rewrites its TMA descriptor when it moves to the next expert."""
import os
from tilekit import SVG, seg, pad, outline, mono, EXP, INK, MUTED, ACC, ACC_F, NEU, NEU_F

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 292, "fig_kscale: token-axis scales, psum offsets, SM90 descriptors")
ks = [300, 60, 200]
nb = [3, 1, 2]

# ---------- (a) scales
s.text(12, 24, "(a) FP8 scales", 18, INK, anchor="start", weight="bold")
RH = 30
y = 50
for e, n in enumerate(nb):
    for b in range(n):
        f, st = EXP[e]
        s.rect(20, y, 96, RH, f, st, 1.2, rx=0)
        s.rect(150, y + 7, 96, 16, f, st, 1.2, rx=0)
        y += RH
outline(s, 20, 50, 96, y - 50)
outline(s, 150, 57, 96, y - 50 - 14, INK, 0)
mono(s, 68, y + 20, "A [ΣK, M]", 15, INK)
mono(s, 198, y + 20, "SF", 15, INK)
s.text(198, 44, "1 per 128 tokens", 15, MUTED)
for k in range(1, 6):
    s.line(20, 50 + k * RH, 116, 50 + k * RH, ACC, 1.4, dash="5,3")
for k in range(6):
    s.line(118, 50 + (k + 0.5) * RH, 148, 50 + (k + 0.5) * RH, ACC, 1.4, "arrAcc")
s.text(134, y + 44, "packed per expert (UE8M0 on SM100)", 15, ACC, weight="bold")

s.line(290, 14, 290, 286, "#999999", 1)

# ---------- (b) psum offsets
s.text(304, 24, "(b) psum offsets on GPU", 18, ACC, anchor="start", weight="bold")
PX = 186 / 768
y0 = 50
ends, starts, st = [], [], 0
for k in ks:
    starts.append(st)
    ends.append(st + k)
    st = -(-(st + k) // 128) * 128
for e, (a, b) in enumerate(zip(starts, ends)):
    seg(s, 318, y0 + a * PX, 70, (b - a) * PX, e, f"E{e}" if (b - a) * PX > 18 else None, size=15)
    nxt = starts[e + 1] if e + 1 < len(starts) else 768
    pad(s, 318, y0 + b * PX, 70, (nxt - b) * PX)
    s.box(436, y0 + b * PX - 11, 54, 22, str(b), ACC_F, ACC, ACC, 15, "bold")
    s.line(434, y0 + b * PX, 391, y0 + b * PX, ACC, 1.8, "arrAcc")
outline(s, 318, y0, 70, 768 * PX)
mono(s, 463, 44, "ends", 15, ACC, weight="bold")
s.text(430, y0 + 768 * PX + 20, "next start = align(end, 128)", 15, INK)
s.text(430, y0 + 768 * PX + 38, "no host K list (SM100)", 15, ACC, weight="bold")

s.line(578, 14, 578, 286, "#999999", 1)

# ---------- (c) SM90 descriptors
s.text(592, 24, "(c) SM90: K-major copies", 18, INK, anchor="start", weight="bold")
bx = 600
for e, k in enumerate(ks):
    al = -(-k // 128) * 128
    w = al * 0.42
    f, stc = EXP[e]
    s.rect(bx, 60 + e * 44, w, 32, f, stc, 1.4, rx=0)
    mono(s, bx + w + 8, 60 + e * 44 + 22, f"[M, {al}]", 15, INK, anchor="start")
s.box(612, 206, 230, 44, ["TMA descriptor in SMEM", "addr + stride per expert"], ACC_F, ACC, ACC, 15, "bold")
s.line(640, 202, 640, 194, ACC, 1.8, "arrAcc")
s.save(os.path.join(HERE, "fig_kscale.svg"))
