"""Figures for deck 31 (HPC-Ops fused FP8 MoE low-latency pipeline). Facts: notes/sources.md.

fig_layer: illustrative uniform top-8 routing of 16 tokens over 192 experts (Hunyuan-V3 benchmark preset),
  sampled like benchmark/fused_moe build_routing (random distinct experts per token), numpy seed 7.
fig_baseline: gather-then-GEMM chain (LMSYS blog / PR #49 description).
fig_pipeline: cp.async path in src/fuse_moe/sm90/cp_async/fuse_moe.cu (5 PDL-launched stages).
fig_count_*: running example, 4 tokens x top-2 over 4 local experts, two counting blocks.
fig_rowidx: Gate-Up GEMM loads rows of x through row_indices with cp.async (group_gemm_fp8_scatter).
fig_occupancy: TMA warp-specialized kernel (384 threads, launch_bounds(384,1), grid = #SMs, 8 stages) vs
  cp.async kernel (128 threads, launch_bounds(128, 5), grid = 5-9 x #SMs, 2 stages).
fig_pdl: schematic, not to scale.
"""
import os
import numpy as np
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
MONO = "Menlo, 'Courier New', monospace"


def badge(s, x, y, n, color=ACC):
    s.add(f'<circle cx="{x}" cy="{y}" r="12" fill="{color}"/>')
    s.text(x, y + 6, str(n), 17, "#FFFFFF", weight="bold")


# ------------------------------------------------------------------ fig_layer (480 x 330)
rng = np.random.default_rng(7)
counts = np.zeros(192, int)
for _ in range(16):
    for e in rng.permutation(192)[:8]:
        counts[e] += 1
s = SVG(480, 330, "fig_layer: 16 tokens x top-8 over 192 experts (illustrative uniform routing)")
s.box(10, 120, 96, 70, ["16 tokens", "top-8"], NEU_F, NEU, size=17)
s.line(108, 155, 140, 155, INK, 2.2, "arr")
CS, X0, Y0 = 19, 146, 34
s.text(X0 + 8 * CS, 24, "192 experts", 18, INK, weight="bold")
fills = {0: "#FFFFFF", 1: ACC_F, 2: "#9DC3E6"}
for i, c in enumerate(counts):
    r, col = divmod(i, 16)
    f = fills.get(int(c), ACC)
    s.rect(X0 + col * CS, Y0 + r * CS, CS, CS, f, "#7F7F7F", 0.8, 0)
s.text(196, 295, "tokens:", 16, INK, anchor="end")
for j, (lab, f) in enumerate([("0", "#FFFFFF"), ("1", ACC_F), ("2", "#9DC3E6"), ("3+", ACC)]):
    x = 206 + j * 66
    s.rect(x, 280, 18, 18, f, "#7F7F7F", 1, 0)
    s.text(x + 24, 295, lab, 16, INK, anchor="start")
s.text(X0 + 8 * CS, 324, "one sample of uniform routing", 15, MUTED, style="italic")
s.save(os.path.join(HERE, "fig_layer.svg"))

# ------------------------------------------------------------------ fig_baseline (870 x 300)
s = SVG(870, 300, "fig_baseline: gather-then-GEMM chain of six kernels")
YS, YH, H = 50, 186, 46
s.text(104, YS + H / 2 + 6, "SMs", 18, INK, anchor="end", weight="bold")
s.text(104, YH + H / 2 + 6, "HBM", 18, INK, anchor="end", weight="bold")
ks = [("route", 70, False), ("gather", 92, True), ("Gate-Up", 120, False), ("act + quant", 112, False),
      ("Down", 110, False), ("reduce", 86, False)]
x = 112
centers = []
for i, (lab, w, bad) in enumerate(ks):
    s.box(x, YS, w, H, lab, PRB_F if bad else NEU_F, PRB if bad else NEU, color=PRB if bad else INK,
          size=16, weight="bold" if bad else "normal")
    centers.append(x + w / 2)
    x += w
    if i < len(ks) - 1:
        s.rect(x, YS, 26, H, "url(#hatchPrb)", PRB, 1.6, 2)
        x += 26
s.text(112 + 70 + 13, YS - 10, "launch gaps", 16, PRB, weight="bold", anchor="start")
hb = [("x", 40, False), ("x × 8 copy", 112, True), ("h BF16", 92, False), ("h FP8", 80, False),
      ("y_e BF16", 100, False), ("y", 40, False)]
hx = [112, 196, 340, 476, 600, 760]
for (lab, w, bad), xx in zip(hb, hx):
    s.box(xx, YH, w, H, lab, PRB_F if bad else "#FFFFFF", PRB if bad else NEU, color=PRB if bad else INK,
          size=16, weight="bold" if bad else "normal")
s.text(252, YH + H + 26, "top-8 gathered copy", 16, PRB, weight="bold")
# writes/reads between kernels and HBM (schematic)
pairs = [(centers[1], 252, True), (centers[2], 386, False), (centers[3], 516, False),
         (centers[4], 650, False), (centers[5], 780, False)]
for kc, hc, bad in pairs:
    s.line(kc, YS + H + 2, hc, YH - 3, PRB if bad else INK, 2.2 if bad else 1.8, "arrPrb" if bad else "arr")
s.line(104, 284, 862, 284, INK, 1.6, "arr")
s.text(862, 278, "time", 15, INK, anchor="end")
s.save(os.path.join(HERE, "fig_baseline.svg"))

# ------------------------------------------------------------------ fig_pipeline (870 x 300)
s = SVG(870, 300, "fig_pipeline: five PDL-chained stages of the cp.async path")
st = [("count + index", 132), ("Gate-Up GEMM", 132), ("SiLU·mul + FP8", 140), ("Down GEMM", 124),
      ("top-k reduce", 124)]
x = 14
xs = []
for i, (lab, w) in enumerate(st):
    s.box(x, 110, w, 56, lab, ACC_F, ACC, size=16, sw=2)
    xs.append((x, w))
    if i < len(st) - 1:
        s.line(x + w + 2, 138, x + w + 34, 138, ACC, 2.6, "arrAcc")
        s.text(x + w + 18, 128, "PDL", 13 + 2, ACC, weight="bold")
    x += w + 36
badge(s, xs[0][0] + 4, 110, 1)
badge(s, xs[1][0] + 4, 110, 2)
badge(s, xs[3][0] + 4, 110, 2)
badge(s, xs[1][0] + xs[1][1] + 18, 162, 3)
# inputs
s.box(xs[0][0] + 6, 22, 120, 34, "topk_ids", "#FFFFFF", NEU, size=15)
s.line(xs[0][0] + 66, 58, xs[0][0] + 66, 108, INK, 1.8, "arr")
s.box(xs[1][0] + 6, 22, 120, 34, "x FP8 as is", "#FFFFFF", NEU, size=15)
s.line(xs[1][0] + 66, 58, xs[1][0] + 66, 108, INK, 1.8, "arr")
s.path(f"M {xs[0][0] + 66} 168 L {xs[0][0] + 66} 196 L {xs[1][0] + 40} 196 L {xs[1][0] + 40} 170",
       ACC, 2, "arrAcc", dash="6,4")
s.text(xs[0][0] + 66 + 4, 216, "row_indices", 15, ACC, anchor="start", family=MONO)
# HBM intermediates
for (sx, sw), lab in zip(xs[1:4], ["h BF16", "h FP8", "y_e BF16"]):
    s.box(sx + sw / 2 - 2, 238, 92, 30, lab, "#FFFFFF", NEU, size=15)
    s.line(sx + sw / 2 + 20, 168, sx + sw / 2 + 20, 236, MUTED, 1.6, "arrGray")
s.text(xs[2][0] + 40, 292, "intermediates in HBM", 15, MUTED, style="italic")
s.box(xs[4][0] + 10, 22, 104, 34, "y BF16", "#FFFFFF", NEU, size=15)
s.line(xs[4][0] + 62, 108, xs[4][0] + 62, 58, INK, 1.8, "arr")
s.save(os.path.join(HERE, "fig_pipeline.svg"))

# ------------------------------------------------------------------ fig_count_1 / _2 (870 x 340)
TOPK = {0: [2, 0], 1: [2, 3], 2: [0, 2], 3: [1, 2]}


def count_fig(step):
    s = SVG(870, 340, f"fig_count step {step}")
    s.text(90, 22, "topk_ids", 17, INK, weight="bold", family=MONO)
    for t in range(4):
        y = 40 + t * 44 + (14 if t >= 2 else 0)
        s.text(30, y + 26, f"t{t}", 17, INK, weight="bold")
        for k, e in enumerate(TOPK[t]):
            s.box(52 + k * 56, y, 52, 36, f"E{e}", NEU_F, NEU, size=16)
    s.rect(8, 34, 170, 96, "none", ACC, 1.6, 6, dash="5,4")
    s.rect(8, 136, 170, 96, "none", ACC, 1.6, 6, dash="5,4")
    blk = [[1, 0, 2, 1], [1, 1, 2, 0]]
    XC, PC, W = 250, 56, 50
    for e in range(4):
        s.text(XC + e * PC + W / 2, 36, f"E{e}", 15, MUTED)
    for b, y in ((0, 44), (1, 196)):
        for e in range(4):
            s.box(XC + e * PC, y, W, 36, str(blk[b][e]), ACC_F, ACC, size=17, sw=1.6)
        s.text(XC + 4 * PC + 8, y + 24, f"block {b}: shared counts", 16, ACC, anchor="start")
    s.line(180, 82, 246, 64, INK, 1.8, "arr")
    s.line(180, 184, 246, 212, INK, 1.8, "arr")
    for e in range(4):
        s.box(XC + e * PC, 120, W, 36, ["2", "1", "4", "1"][e], "#FFFFFF", INK, size=17, sw=2)
        cx = XC + e * PC + W / 2
        if blk[0][e]:
            s.line(cx - 8, 82, cx - 8, 117, ACC, 2, "arrAcc")
        if blk[1][e]:
            s.line(cx + 8, 194, cx + 8, 159, ACC, 2, "arrAcc")
    s.text(XC + 4 * PC + 8, 136, "global: one atomic", 16, INK, anchor="start", weight="bold")
    s.text(XC + 4 * PC + 8, 156, "per expert per block", 16, INK, anchor="start", weight="bold")
    s.text(XC + 4 * PC + 8, 182, "→ cu_seqlens 0 2 3 7 8", 15, INK, anchor="start", family=MONO)
    if step == 2:
        s.text(XC - 10, 290, "row_indices", 16, ACC, anchor="end", family=MONO, weight="bold")
        order = [(0, 0), (2, 0), (3, 1), (0, 2), (1, 2), (2, 2), (3, 2), (1, 3)]
        for i, (t, e) in enumerate(order):
            s.box(XC + i * 50, 268, 46, 36, f"t{t}", ACC_F if e % 2 == 0 else "#FFFFFF", ACC, size=16, sw=1.6)
        for e, (a, b) in enumerate([(0, 2), (2, 3), (3, 7), (7, 8)]):
            s.text(XC + (a + b) / 2 * 50 - 2, 326, f"E{e}", 15, ACC, weight="bold")
        s.text(XC + 8 * 50 + 16, 292, "+ topk_pos for the reduce", 15, MUTED, anchor="start", style="italic")
    return s


count_fig(1).save(os.path.join(HERE, "fig_count_1.svg"))
count_fig(2).save(os.path.join(HERE, "fig_count_2.svg"))

# ------------------------------------------------------------------ fig_rowidx (870 x 300)
s = SVG(870, 300, "fig_rowidx: Gate-Up tile of expert E2 loads x rows through row_indices")
s.text(90, 30, "x  (token order)", 17, INK, weight="bold")
for t in range(4):
    s.box(20, 44 + t * 46, 140, 40, f"t{t}", NEU_F, NEU, size=16)
s.text(250, 270, "row_indices of E2", 16, ACC, anchor="end", weight="bold")
for i, t in enumerate([0, 1, 2, 3]):
    s.box(262 + i * 40, 250, 36, 32, f"t{t}", ACC_F, ACC, size=15, sw=1.6)
s.text(640, 30, "smem tile of E2 (8 rows)", 17, INK, weight="bold")
for r in range(8):
    real = r < 4
    s.rect(560, 44 + r * 23, 160, 23, ACC_F if real else "#FFFFFF", ACC if real else "#BFBFBF", 1.4, 0,
           dash=None if real else "4,3")
    if real:
        s.text(640, 44 + r * 23 + 17, f"row {r} ← t{r}", 15, INK)
for t in range(4):
    s.path(f"M 162 {64 + t * 46} C 360 {64 + t * 46} 380 {55 + t * 23} 556 {55 + t * 23}", ACC, 2, "arrAcc")
s.text(380, 34, "cp.async per row", 16, ACC, weight="bold")
s.box(750, 112, 100, 56, ["WGMMA", "with W_E2"], "#FFFFFF", ACC, size=16, sw=1.8)
s.line(722, 140, 748, 140, INK, 2, "arr")
s.rect(560, 248, 180, 36, "#FFFFFF", PRB, 1.8, 4, dash="6,4")
s.text(650, 272, "no gathered copy", 16, PRB, weight="bold")
s.save(os.path.join(HERE, "fig_rowidx.svg"))

# ------------------------------------------------------------------ fig_occupancy (870 x 300)
s = SVG(870, 300, "fig_occupancy: one warp-specialized CTA vs many small cp.async CTAs per SM")
s.text(210, 26, "TMA + warp specialization", 18, INK, weight="bold")
s.rect(20, 40, 380, 220, "#FFFFFF", NEU, 2, 6)
s.text(30, 60, "one SM", 15, MUTED, anchor="start", style="italic")
s.rect(40, 72, 340, 172, NEU_F, NEU, 2, 4)
s.text(210, 94, "1 CTA · 384 threads", 16, INK, weight="bold")
s.box(56, 108, 96, 54, ["producer", "WG"], "#FFFFFF", NEU, size=15)
s.box(162, 108, 96, 54, ["math", "WG"], "#FFFFFF", NEU, size=15)
s.box(268, 108, 96, 54, ["math", "WG"], "#FFFFFF", NEU, size=15)
for i in range(8):
    s.rect(70 + i * 37, 182, 30, 40, "#FFFFFF" if i > 1 else NEU_F, "#7F7F7F", 1.2, 2,
           dash="4,3" if i > 1 else None)
s.text(210, 238, "8-stage ring", 15, MUTED, style="italic")
s.text(645, 26, "cp.async, no specialization", 18, ACC, weight="bold")
s.rect(470, 40, 380, 220, "#FFFFFF", ACC, 2, 6)
s.text(480, 60, "one SM", 15, MUTED, anchor="start", style="italic")
for i in range(5):
    x = 486 + i * 72
    s.rect(x, 72, 64, 172, ACC_F, ACC, 1.8, 4)
    s.text(x + 32, 96, "CTA", 15, ACC, weight="bold")
    s.box(x + 6, 108, 52, 54, ["1", "WG"], "#FFFFFF", ACC, size=15, sw=1.4)
    for j in range(2):
        s.rect(x + 10 + j * 24, 182, 20, 40, "#FFFFFF", ACC, 1.2, 2)
s.text(210, 288, "1 CTA per SM · 8 stages", 17, INK, weight="bold")
s.text(660, 288, "≥ 5 CTAs per SM · 2 stages each", 17, ACC, weight="bold")
s.save(os.path.join(HERE, "fig_occupancy.svg"))

# ------------------------------------------------------------------ fig_pdl (870 x 250)
s = SVG(870, 250, "fig_pdl: five stages without and with PDL (schematic)")
labs = ["count", "Gate-Up", "act·quant", "Down", "reduce"]
ws = [70, 150, 80, 140, 80]
s.text(126, 64, "stream order", 18, INK, anchor="end", weight="bold")
x = 136
for i, (lab, w) in enumerate(zip(labs, ws)):
    s.box(x, 40, w, 40, lab, NEU_F, NEU, size=16)
    x += w
    if i < 4:
        s.rect(x, 40, 30, 40, "url(#hatchPrb)", PRB, 1.6, 2)
        x += 30
end1 = x
s.text(126, 154, "with PDL", 18, ACC, anchor="end", weight="bold")
x = 136
for i, (lab, w) in enumerate(zip(labs, ws)):
    s.box(x, 130, w, 40, lab, ACC_F, ACC, size=16)
    x += w + 6
end2 = x - 6
s.line(end1, 24, end1, 214, MUTED, 1.6, dash="5,5")
s.line(end2 + 2, 196, end1 - 2, 196, ACC, 2.4, "arrAcc", start_marker="arrAcc")
s.text((end1 + end2) / 2, 222, "time saved", 17, ACC, weight="bold")
s.line(128, 244, 862, 244, INK, 1.6, "arr")
s.text(862, 238, "time", 15, INK, anchor="end")
s.save(os.path.join(HERE, "fig_pdl.svg"))
