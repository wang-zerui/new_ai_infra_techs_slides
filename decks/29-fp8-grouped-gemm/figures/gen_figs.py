"""Figures for deck 29 (HPC-Ops FP8 grouped GEMM). Facts: see notes/sources.md.

Running example: 4 experts with seqlens = [3, 1, 5, 2] -> cu_seqlens = [0, 3, 4, 9, 11]
(the API's seqlens / cu_seqlens; the README calls them num_tokens_per_group / cu_num_tokens_per_group).
Padding example: tokens per expert 8, 3, 13, 1 under 64-row tiles (DeepGEMM SM90 masked BLOCK_M >= 64)
and 128-row alignment (DeepGEMM contiguous layout default).
"""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
MONO = "Menlo, 'Courier New', monospace"


def badge(s, x, y, n, color=ACC):
    s.add(f'<circle cx="{x}" cy="{y}" r="12" fill="{color}"/>')
    s.text(x, y + 6, str(n), 17, "#FFFFFF", weight="bold")


SEQ = [3, 1, 5, 2]
CU = [0, 3, 4, 9, 11]

# ------------------------------------------------------------------ fig_groups (480 x 340)
s = SVG(480, 340, "fig_groups: grouped GEMM with seqlens / cu_seqlens")
RH, Y0 = 24, 50
XC, XW = 110, 90          # x column
CC, CW = 236, 82          # weight chips
YC, YW = 366, 90          # y column
s.text(XC + XW / 2, 38, "x  FP8", 18, INK, weight="bold")
s.text(CC + CW / 2, 38, "W_g", 18, INK, weight="bold")
s.text(YC + YW / 2, 38, "y  BF16", 18, INK, weight="bold")
s.text(100, 38, "cu_seqlens", 15, MUTED, anchor="end", family=MONO)
for g, (n, c0) in enumerate(zip(SEQ, CU)):
    ytop = Y0 + c0 * RH
    for r in range(n):
        s.rect(XC, ytop + r * RH, XW, RH, NEU_F, NEU, 1.2, 0)
        s.rect(YC, ytop + r * RH, YW, RH, "#FFFFFF", NEU, 1.2, 0)
    s.rect(XC, ytop, XW, n * RH, "none", INK, 2.4, 0)
    s.rect(YC, ytop, YW, n * RH, "none", INK, 2.4, 0)
    cy = ytop + n * RH / 2
    s.text(XC + XW / 2, cy + 6, f"E{g}", 17, INK, weight="bold", halo=True)
    s.text(YC + YW / 2, cy + 6, f"y{g}", 17, INK, halo=True)
    s.box(CC, cy - 10, CW, 20, f"W{g}", ACC_F, ACC, size=15, sw=1.6)
    s.line(XC + XW + 2, cy, CC - 2, cy, INK, 1.8, "arr")
    s.line(CC + CW + 2, cy, YC - 2, cy, INK, 1.8, "arr")
for c in CU:
    s.text(100, Y0 + c * RH + 5, str(c), 15, MUTED, anchor="end", family=MONO)
s.text(240, 334, "seqlens = [3, 1, 5, 2]", 16, INK, family=MONO)
s.save(os.path.join(HERE, "fig_groups.svg"))

# ------------------------------------------------------------------ fig_padding (870 x 270)
s = SVG(870, 270, "fig_padding: token-major tiles pad small experts")
TOK = [8, 3, 13, 1]
s.rect(190, 12, 22, 16, NEU_F, NEU, 1.5, 2)
s.text(218, 26, "token rows", 16, INK, anchor="start")
s.rect(350, 12, 22, 16, "url(#hatchPrb)", PRB, 1.5, 2)
s.text(378, 26, "padding rows", 16, INK, anchor="start")
# row 1: one 64-row tile per expert
U1, X1, W1, G1 = 2.4, 190, 64 * 2.4, 14
s.text(178, 84, "64-row tiles", 18, INK, anchor="end", weight="bold")
s.text(178, 106, "masked layout", 16, MUTED, anchor="end", style="italic")
for i, t in enumerate(TOK):
    x = X1 + i * (W1 + G1)
    s.text(x + W1 / 2, 60, f"E{i}: {t} tok", 16, INK)
    s.rect(x, 68, W1, 44, "url(#hatchPrb)", PRB, 1.8, 0)
    s.rect(x, 68, t * U1, 44, NEU_F, NEU, 1.8, 0)
# row 2: contiguous segments aligned to 128 rows
U2 = (4 * W1 + 3 * G1) / 512
s.text(178, 192, "128-row segments", 18, INK, anchor="end", weight="bold")
s.text(178, 214, "contiguous layout", 16, MUTED, anchor="end", style="italic")
for i, t in enumerate(TOK):
    x = X1 + i * 128 * U2
    s.rect(x, 176, 128 * U2, 44, "url(#hatchPrb)", PRB, 1.8, 0)
    s.rect(x, 176, max(t * U2, 2.5), 44, NEU_F, NEU, 1.8, 0)
    s.text(x + 64 * U2, 168, f"E{i}", 16, INK)
s.line(X1, 240, X1 + 4 * W1 + 3 * G1, 240, MUTED, 1.4, "arrGray", start_marker="arrGray")
s.text(X1 + (4 * W1 + 3 * G1) / 2, 262, "rows the MMA computes", 16, MUTED, style="italic")
s.save(os.path.join(HERE, "fig_padding.svg"))

# ------------------------------------------------------------------ fig_overview (870 x 310)
s = SVG(870, 310, "fig_overview: describe-groups kernel + persistent grouped GEMM")
for i, lab in enumerate(["seqlens [G]", "cu_seqlens [G+1]"]):
    s.box(10, 60 + i * 52, 168, 36, lab, "#FFFFFF", NEU, size=15)
    s.line(180, 78 + i * 52, 210, 78 + i * 52, INK, 2, "arr")
# kernel 0
s.rect(212, 30, 190, 196, ACC_F, ACC, 2.2, 6)
s.text(307, 54, "kernel 0", 18, ACC, weight="bold")
for i, lab in enumerate(["TMA desc per group", "tiles + cu_tiles scan", "task map (decode)"]):
    s.box(224, 66 + i * 50, 166, 40, lab, "#FFFFFF", ACC, size=15, sw=1.6)
badge(s, 224, 66, 1)
s.line(404, 128, 440, 128, ACC, 2.6, "arrAcc")
s.text(422, 116, "PDL", 16, ACC, weight="bold")
# kernel 1
s.rect(442, 30, 418, 196, ACC_F, ACC, 2.2, 6)
s.text(651, 54, "kernel 1: persistent, one CTA per SM", 18, ACC, weight="bold")
s.box(454, 66, 394, 34, "tile list over (group, m-tile, n-tile)", "#FFFFFF", ACC, size=15, sw=1.6)
for x, w, lab in ((454, 114, ["producer WG", "TMA loads"]), (580, 156, ["2 math WGs", "swapped WGMMA"]),
                  (748, 100, ["epilogue", "TMA store"])):
    s.box(x, 120, w, 58, lab, "#FFFFFF", ACC, size=15, sw=1.6)
s.line(569, 149, 578, 149, INK, 2, "arr")
s.line(737, 149, 746, 149, INK, 2, "arr")
badge(s, 582, 120, 2)
badge(s, 594, 204, 3)
s.text(612, 210, "per-K scale · FP32 accumulate", 15, ACC, anchor="start", style="italic")
# bottom inputs and output
for i, (x, w, lab) in enumerate([(10, 150, "x [Σt, K]"), (172, 160, "W [G, N, K]"), (344, 92, "scales")]):
    s.box(x, 254, w, 36, lab, "#FFFFFF", NEU, size=15)
s.path("M 438 272 L 520 272 L 520 229", INK, 2, "arr")
s.box(690, 254, 170, 36, "y [Σt, N] BF16", "#FFFFFF", NEU, size=15)
s.line(836, 180, 836, 251, INK, 2, "arr")
s.save(os.path.join(HERE, "fig_overview.svg"))

# ------------------------------------------------------------------ fig_tma (870 x 300)
s = SVG(870, 300, "fig_tma: seqlens + cu_seqlens become one TMA descriptor per group")


def strip(y, cell, starts, slots, label, sub):
    s.text(10, y + 18, label, 18, INK, anchor="start", weight="bold")
    s.text(10, y + 40, sub, 15, MUTED, anchor="start", family=MONO)
    x0 = 220
    for g, n in enumerate(SEQ):
        gx = x0 + starts[g] * cell
        for r in range(slots[g]):
            real = r < n
            s.rect(gx + r * cell, y, cell, 40, NEU_F if real else "#FFFFFF", NEU if real else "#BFBFBF",
                   1.2, 0, dash=None if real else "4,3")
        s.rect(gx, y, n * cell, 40, "none", INK, 2.4, 0)
        s.text(gx + n * cell / 2, y - 8, f"E{g}", 16, INK, weight="bold")
        s.box(gx + 2, y + 46, n * cell - 4, 24, f"d{g}", ACC_F, ACC, color=ACC, size=15, weight="bold", sw=1.8)


strip(40, 52, [0, 3, 4, 9], SEQ, "compact", "cu = 0 3 4 9 11")
strip(176, 30, [0, 5, 10, 15], [5, 5, 5, 5], "DeepEP layout", "cu = 0 5 10 15")
s.rect(220, 278, 22, 16, ACC_F, ACC, 1.5, 3)
s.text(248, 292, "TMA descriptor: base and rows of one group", 16, INK, anchor="start")
s.rect(640, 278, 22, 16, "#FFFFFF", "#BFBFBF", 1.5, 0, dash="4,3")
s.text(668, 292, "never loaded", 16, INK, anchor="start")
s.save(os.path.join(HERE, "fig_tma.svg"))

# ------------------------------------------------------------------ fig_swap (870 x 330)
s = SVG(870, 330, "fig_swap: tokens on the fixed 64-row M side vs on the flexible N side")
# (a) token-major
ax, ay, aw, ah = 150, 64, 200, 192
s.text(250, 26, "(a) tokens as the M side", 18, INK, weight="bold")
s.rect(ax, ay, aw, ah, "url(#hatchPrb)", PRB, 2, 0)
s.rect(ax, ay, aw, ah * 8 / 64, NEU_F, NEU, 2, 0)
s.text(ax + aw / 2, ay + 18, "8 tokens", 16, INK, weight="bold")
s.text(ax + aw / 2, ay + 112, "56 padding rows", 17, PRB, weight="bold", halo=True)
s.text(ax + aw / 2, ay - 10, "N = 128 weight cols", 16, INK)
s.text(ax - 10, ay + ah / 2 - 4, "M = 64", 17, INK, anchor="end", weight="bold")
s.text(ax - 10, ay + ah / 2 + 16, "fixed", 16, MUTED, anchor="end", style="italic")
# (b) swapped
bx, by, bw, bh = 640, 64, 48, 192
s.text(650, 26, "(b) weights as the M side", 18, ACC, weight="bold")
s.rect(bx, by, bw, bh / 2, ACC_F, ACC, 2, 0)
s.rect(bx, by + bh / 2, bw, bh / 2, ACC_F, ACC, 2, 0)
s.text(bx + bw / 2, by - 10, "N = 8 tokens", 16, INK)
s.text(bx - 10, by + bh / 4 + 6, "WG 1: 64 rows", 16, INK, anchor="end")
s.text(bx - 10, by + 3 * bh / 4 + 6, "WG 2: 64 rows", 16, INK, anchor="end")
s.text(bx + bw + 12, by + bh / 2 - 4, "128 weight", 16, INK, anchor="start")
s.text(bx + bw + 12, by + bh / 2 + 16, "rows", 16, INK, anchor="start")
s.text(bx + bw / 2, by + bh + 28, "SM90_64x8x32 FP8 atom", 15, ACC, family=MONO)
s.text(250, by + ah + 28, "y = x · Wᵀ", 16, INK, family=MONO)
s.text(650, by + bh + 52, "yᵀ = W · xᵀ", 16, ACC, family=MONO)
s.line(395, 160, 520, 160, ACC, 3, "arrAcc")
s.text(458, 148, "swap", 17, ACC, weight="bold")
s.save(os.path.join(HERE, "fig_swap.svg"))

# ------------------------------------------------------------------ fig_scales (870 x 300)
s = SVG(870, 300, "fig_scales: one mainloop, two FP8 scale granularities")
# left: granularity pictures
s.text(150, 26, "per-tensor", 18, INK, weight="bold")
s.rect(70, 40, 160, 80, NEU_F, NEU, 1.8, 0)
s.text(150, 86, "W_g", 18, INK)
s.box(196, 44, 30, 24, "s", ACC_F, ACC, size=15, sw=1.6)
s.text(150, 140, "one scale per expert", 15, MUTED, style="italic")
s.text(150, 176, "block-wise", 18, INK, weight="bold")
for j in range(4):           # W blocks 2 x 4 of 128 x 128
    for i in range(2):
        s.rect(110 + j * 40, 190 + i * 40, 40, 40, NEU_F, NEU, 1.4, 0)
        s.rect(110 + j * 40 + 24, 190 + i * 40 + 4, 12, 12, ACC_F, ACC, 1.2, 1)
for j in range(4):           # x tile rows with 1 x 128 scales
    s.rect(20, 190 + j * 20, 60, 20, NEU_F, NEU, 1.2, 0)
    s.rect(66, 190 + j * 20 + 4, 10, 12, ACC_F, ACC, 1.2, 1)
s.text(50, 292, "x: 1 × 128", 15, MUTED, style="italic")
s.text(190, 292, "W: 128 × 128", 15, MUTED, style="italic")
# right: mainloop per K block
s.line(320, 20, 320, 290, "#BFBFBF", 1.2, dash="4,4")
s.text(600, 26, "per 128-wide K block", 18, ACC, weight="bold")
s.box(344, 120, 120, 50, ["WGMMA", "FP8 → FP32"], "#FFFFFF", ACC, size=16, sw=1.8)
s.line(466, 145, 513, 145, INK, 2, "arr")
s.add(f'<circle cx="532" cy="145" r="18" fill="{ACC_F}" stroke="{ACC}" stroke-width="2"/>')
s.text(532, 152, "×", 22, ACC, weight="bold")
s.line(551, 145, 603, 145, INK, 2, "arr")
s.add(f'<circle cx="622" cy="145" r="18" fill="{ACC_F}" stroke="{ACC}" stroke-width="2"/>')
s.text(622, 152, "+", 22, ACC, weight="bold")
s.line(641, 145, 690, 145, INK, 2, "arr")
s.box(692, 120, 150, 50, ["FP32", "accumulator"], ACC_F, ACC, size=16, sw=1.8)
s.path("M 767 172 L 767 214 L 622 214 L 622 165", ACC, 2, "arrAcc")
s.text(694, 236, "next K block", 15, ACC, style="italic")
# scale inputs to the multiply node
s.line(532, 84, 532, 125, ACC, 2, "arrAcc")
s.text(532, 58, "s  or  x_s[t] · w_s[blk]", 16, INK, family=MONO)
s.text(532, 78, "scale", 15, MUTED, style="italic")
s.line(767, 118, 767, 76, INK, 2, "arr")
s.text(767, 66, "BF16 y", 16, INK, weight="bold")
s.text(767, 44, "after the last K block", 15, MUTED, style="italic")
s.save(os.path.join(HERE, "fig_scales.svg"))
