"""Mechanism figures for deck 13 (HPC-Ops fused AllReduce + residual + RMSNorm).

Facts drawn (see notes/sources.md):
- fig_overview: two modes, both two-shot (PR #48; README). HT = one kernel: multimem.ld_reduce ->
  +residual, RMSNorm in registers -> multimem.st (fuse_allreduce_rmsnorm_high_throughput.cu).
  LL = twoshotAllreduceKernel (P2P scatter to owner, Lamport poll, sum, multicast broadcast) +
  rmsNormLamport, launched with PDL (fuse_allreduce_rmsnorm_low_latency.cu).
- fig_twoshot: token ownership; 4 GPUs x 4 tokens example (tests use world_size 4 and 8).
- fig_ht_*: owner GPU pulls the cross-GPU sum of its tokens through the switch, normalizes, pushes.
- fig_lamport_*: scatter slots pre-filled with -0; sender maps -0 to +0; owner spins until no -0;
  3 rotating Lamport buffers (LamportBufferLayout::sNumLamportBuffers = 3).
- fig_pdl: kernel 1 triggers programmatic launch completion early; kernel 2 prefetches residual and
  gamma with cp.async, polls the broadcast buffer, then normalizes. Schematic, not to scale.
"""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))


def badge(s, x, y, n, color=ACC):
    s.add(f'<circle cx="{x}" cy="{y}" r="13" fill="{color}"/>')
    s.text(x, y + 6.3, str(n), 18, "#FFFFFF", weight="bold")


# ------------------------------------------------------------------ fig_overview (870 x 300)
s = SVG(870, 300, "fig_overview: one op, two modes, both two-shot")
# row labels
badge(s, 20, 82, 2)
s.text(40, 88, "high-throughput", 19, ACC, anchor="start", weight="bold")
s.text(40, 112, "many tokens", 17, MUTED, anchor="start", style="italic")
badge(s, 20, 218, 3)
s.text(40, 224, "low-latency", 19, ACC, anchor="start", weight="bold")
s.text(40, 248, "few tokens", 17, MUTED, anchor="start", style="italic")
# HT: one kernel
s.rect(204, 30, 656, 116, "#FFFFFF", ACC, 2, 6, dash="7,4")
s.text(214, 50, "one kernel", 16, ACC, anchor="start", weight="bold")
for x, lab, tag in ((216, ["multicast", "ld_reduce"], "shot 1"),
                    (432, ["+ residual", "RMSNorm"], "norm on owned tokens"),
                    (648, ["multicast", "store"], "shot 2")):
    s.box(x, 60, 200, 56, lab, ACC_F, ACC)
    s.text(x + 100, 138, tag, 16, INK, style="italic")
for x in (416, 632):
    s.line(x + 1, 88, x + 15, 88, INK, 2.2, "arr")
# LL: two kernels chained by PDL
s.rect(204, 168, 426, 116, "#FFFFFF", ACC, 2, 6, dash="7,4")
s.text(214, 188, "kernel 1", 16, ACC, anchor="start", weight="bold")
s.rect(676, 168, 184, 116, "#FFFFFF", ACC, 2, 6, dash="7,4")
s.text(686, 188, "kernel 2", 16, ACC, anchor="start", weight="bold")
for x, w, lab, tag in ((216, 200, ["P2P scatter", "+ sum on owner"], "shot 1"),
                       (432, 186, ["multicast", "broadcast"], "shot 2"),
                       (686, 164, ["+ residual", "RMSNorm"], "norm on all tokens")):
    s.box(x, 198, w, 56, lab, ACC_F, ACC)
    s.text(x + w / 2, 276, tag, 16, INK, style="italic")
s.line(417, 226, 431, 226, INK, 2.2, "arr")
s.line(619, 226, 684, 226, ACC, 2.6, "arrAcc")
s.text(653, 212, "PDL", 18, ACC, weight="bold", halo=True)
s.save(os.path.join(HERE, "fig_overview.svg"))

# ------------------------------------------------------------------ fig_twoshot (870 x 300)
CW, CH = 44, 40
PX = (96, 384, 672)
Y0 = 72


def grid(s, x0, kind):
    for t in range(4):
        s.text(x0 + t * CW + CW / 2, Y0 - 10, f"t{t}", 17, INK)
    for r in range(4):
        for t in range(4):
            x, y = x0 + t * CW, Y0 + r * CH
            if kind == "partial":
                s.box(x, y, CW, CH, "p", NEU_F, NEU, size=17, rx=0, sw=1.5)
            elif kind == "scatter":
                if r == t:
                    s.box(x, y, CW, CH, "Σ", ACC_F, ACC, color=ACC, size=18, weight="bold", rx=0, sw=2.5)
                else:
                    s.box(x, y, CW, CH, "", "#FFFFFF", "#BFBFBF", size=17, rx=0, sw=1.2)
            else:
                s.box(x, y, CW, CH, "Σ", ACC_F, ACC, color=ACC, size=18, weight="bold", rx=0, sw=1.5)


s = SVG(870, 300, "fig_twoshot: token ownership in a two-shot all-reduce (4 GPUs, 4 tokens)")
for r in range(4):
    s.text(PX[0] - 10, Y0 + r * CH + CH / 2 + 6.3, f"GPU {r}", 17, INK, anchor="end", weight="bold")
grid(s, PX[0], "partial")
grid(s, PX[1], "scatter")
grid(s, PX[2], "gather")
for (xa, xb), l1, l2 in (((PX[0] + 4 * CW + 8, PX[1] - 8), "shot 1", "reduce-scatter"),
                         ((PX[1] + 4 * CW + 8, PX[2] - 8), "shot 2", "all-gather")):
    s.line(xa, Y0 + 2 * CH, xb, Y0 + 2 * CH, ACC, 2.6, "arrAcc")
    s.text((xa + xb) / 2, Y0 + 2 * CH - 34, l1, 17, ACC, weight="bold")
    s.text((xa + xb) / 2, Y0 + 2 * CH - 14, l2, 16, ACC)
for x0, cap in zip(PX, ("(a) partial sums", "(b) after shot 1", "(c) after shot 2")):
    s.text(x0 + 2 * CW, Y0 + 4 * CH + 30, cap, 18, INK, weight="bold")
s.text(PX[1] + 2 * CW, Y0 + 4 * CH + 54, "GPU r owns token r", 17, MUTED, style="italic")
s.save(os.path.join(HERE, "fig_twoshot.svg"))

# ------------------------------------------------------------------ fig_ht_1 / fig_ht_2 (870 x 316)
GW, GH, GY = 58, 46, 248
SY, SH = 52, 46


def gpus(s, x0, owner=1):
    for i in range(4):
        x = x0 + 20 + i * 67
        own = i == owner
        s.box(x, GY, GW, GH, f"GPU{i}", ACC_F if own else NEU_F, ACC if own else NEU,
              color=ACC if own else INK, size=16, weight="bold" if own else "normal", sw=2.5 if own else 2)
    return [x0 + 20 + i * 67 + GW / 2 for i in range(4)]


def switch(s, x0):
    s.box(x0 + 20, SY, 259, SH, "NVSwitch", NEU_F, NEU, size=18, weight="bold")


def panel_ld(s):
    x0 = 0
    s.text(150, 26, "pull Σ: multicast ld_reduce", 18, ACC, weight="bold")
    switch(s, x0)
    cx = gpus(s, x0)
    for c in cx:
        s.line(c - 10, GY - 2, c - 10, SY + SH + 3, INK, 2, "arr")
    s.line(cx[1] + 10, SY + SH + 2, cx[1] + 10, GY - 3, ACC, 3, "arrAcc")
    s.text(cx[1] + 18, 176, "Σ", 22, ACC, anchor="start", weight="bold", halo=True)
    s.text(150, 314, "partials of token t1", 16, MUTED, style="italic")


def panel_norm(s):
    s.text(450, 26, "normalize in registers", 18, ACC, weight="bold")
    s.rect(352, 52, 196, 242, ACC_F, ACC, 2.5, 6)
    s.text(450, 80, "GPU1", 18, ACC, weight="bold")
    s.box(368, 98, 164, 44, "Σ + residual", "#FFFFFF", ACC, size=18)
    s.line(450, 143, 450, 160, INK, 2.2, "arr")
    s.box(368, 162, 164, 44, "RMSNorm", "#FFFFFF", ACC, size=18)
    s.line(450, 207, 450, 224, INK, 2.2, "arr")
    s.box(368, 226, 164, 44, "× weight", "#FFFFFF", ACC, size=18)
    # hand-off arrows between panels
    s.line(300, 173, 346, 173, ACC, 3, "arrAcc")
    s.line(553, 173, 588, 173, ACC, 3, "arrAcc")


def panel_st(s):
    x0 = 572
    s.text(x0 + 150, 26, "push: multicast store", 18, ACC, weight="bold")
    switch(s, x0)
    cx = gpus(s, x0)
    s.line(cx[1] - 10, GY - 2, cx[1] - 10, SY + SH + 3, ACC, 3, "arrAcc")
    for c in cx:
        s.line(c + 10, SY + SH + 2, c + 10, GY - 3, ACC, 2.4, "arrAcc")
    s.text(x0 + 150, 314, "normalized t1 to all GPUs", 16, ACC, style="italic")


s = SVG(870, 320, "fig_ht_1: owner GPU1 pulls the sum of token t1 through the switch")
panel_ld(s)
s.save(os.path.join(HERE, "fig_ht_1.svg"))
s = SVG(870, 320, "fig_ht_2: pull, normalize in registers, push to all GPUs")
panel_ld(s)
panel_norm(s)
panel_st(s)
s.save(os.path.join(HERE, "fig_ht_2.svg"))

# ------------------------------------------------------------------ fig_lamport_1 / _2 (870 x 316)
RY = [60, 120, 180, 240]
RH = 46


def lamport(done):
    s = SVG(870, 316, f"fig_lamport ({'all arrived' if done else 'one partial missing'})")
    s.text(80, 34, "partials of token t", 17, MUTED, style="italic")
    s.text(415, 34, "GPU1 scatter slots", 18, INK, weight="bold")
    s.text(775, 34, "broadcast buffers", 18, INK, weight="bold")
    for i, y in enumerate(RY):
        cy = y + RH / 2
        arrived = done or i != 3
        s.box(20, y, 120, RH, f"GPU{i}", NEU_F, NEU, size=17)
        if arrived:
            s.line(142, cy, 328, cy, ACC, 2.4, "arrAcc")
            s.box(330, y, 170, RH, f"from GPU{i}", ACC_F, ACC, size=17)
        else:
            s.line(142, cy, 300, cy, MUTED, 2, dash="6,5")
            s.rect(330, y, 170, RH, "url(#hatchGray)", MUTED, 2, 4)
            s.text(415, cy + 7, "−0", 20, PRB, weight="bold", halo=True)
        # fan-in to the sum node
        s.line(502, cy, 560, 172 + (cy - 172) * 0.25, ACC if arrived else MUTED, 2.2,
               dash=None if arrived else "5,4")
    s.text(235, 100, "P2P store", 17, INK, halo=True)
    col = ACC if done else PRB
    s.add(f'<circle cx="588" cy="172" r="27" fill="{ACC_F if done else PRB_F}" stroke="{col}" stroke-width="2.5"/>')
    s.text(588, 180, "Σ", 24, col, weight="bold")
    if not done:
        s.text(588, 230, "spin while", 17, PRB, weight="bold")
        s.text(588, 252, "any −0", 17, PRB, weight="bold")
    for i, y in enumerate(RY):
        cy = y + RH / 2
        if done:
            s.line(616, 172 + (cy - 172) * 0.25, 698, cy, ACC, 2.2, "arrAcc")
            s.box(700, y, 150, RH, f"GPU{i}", ACC_F, ACC, size=17)
        else:
            s.box(700, y, 150, RH, f"GPU{i}", "#FFFFFF", "#BFBFBF", color=MUTED, size=17, dash="6,4")
    if done:
        s.text(588, 250, "multicast", 16, ACC, weight="bold")
        s.text(588, 270, "store", 16, ACC, weight="bold")
    s.text(80, 308, "sender maps −0 to +0", 16, MUTED, style="italic")
    s.text(415, 308, "3 rotating buffers", 16, MUTED, style="italic")
    return s


lamport(False).save(os.path.join(HERE, "fig_lamport_1.svg"))
lamport(True).save(os.path.join(HERE, "fig_lamport_2.svg"))

# ------------------------------------------------------------------ fig_pdl (870 x 270)
s = SVG(870, 292, "fig_pdl: plain stream order vs PDL launch of the RMSNorm kernel (schematic)")
X0 = 170
s.text(158, 58, "stream order", 18, INK, anchor="end", weight="bold")
s.box(X0, 34, 300, 42, "kernel 1: all-reduce", NEU_F, NEU, size=17)
s.rect(X0 + 300, 34, 60, 42, "url(#hatchPrb)", PRB, 2, 3)
s.text(X0 + 330, 26, "launch", 16, PRB, weight="bold")
s.box(X0 + 360, 34, 130, 42, "prologue", NEU_F, NEU, size=17)
s.box(X0 + 490, 34, 200, 42, "+ res · RMSNorm", NEU_F, NEU, size=17)
s.line(X0 + 690, 22, X0 + 690, 236, MUTED, 1.6, dash="5,5")
# PDL lanes
s.text(158, 136, "with PDL", 18, ACC, anchor="end", weight="bold")
s.text(158, 156, "kernel 1", 16, MUTED, anchor="end", style="italic")
s.box(X0, 116, 300, 40, "kernel 1: all-reduce", NEU_F, NEU, size=17)
s.text(158, 196, "kernel 2", 16, MUTED, anchor="end", style="italic")
s.box(X0 + 24, 172, 170, 40, "prefetch res γ", ACC_F, ACC, size=17)
s.rect(X0 + 194, 172, 106, 40, "url(#hatchGray)", MUTED, 2, 3)
s.text(X0 + 247, 198, "poll", 17, INK, halo=True)
s.box(X0 + 300, 172, 200, 40, "+ res · RMSNorm", ACC_F, ACC, size=17)
s.line(X0 + 24, 158, X0 + 24, 170, ACC, 2.2, "arrAcc")
s.text(X0 + 30, 108, "early trigger", 15, ACC, anchor="start", style="italic")
s.line(X0 + 500, 222, X0 + 690, 222, ACC, 2.4, "arrAcc", start_marker="arrAcc")
s.text(X0 + 595, 246, "time saved", 17, ACC, weight="bold")
s.line(X0 - 8, 266, 862, 266, INK, 1.6, "arr")
s.text(862, 286, "time", 16, INK, anchor="end")
s.save(os.path.join(HERE, "fig_pdl.svg"))
