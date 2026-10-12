"""fig_dualkernel: batch-invariant decoding attention (DeepSeek-V4 §3.3). Without split-KV, one SM computes a whole
sequence; a partial last wave leaves SMs idle (wave quantization). Kernel 2 spreads each tail sequence over the SMs
of a thread-block cluster, exchanging data through distributed shared memory, with the same accumulation order.
Schematic: 8 SMs, 10 sequences, 4-SM clusters; durations illustrative."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 292
s = SVG(W, H, "fig_dualkernel")

RH, RG = 22, 4
Y0 = 34
U = 1.5


def frame(px, title, accent=False):
    s.text(px + 4, 20, title, 19, ACC if accent else INK, anchor="start", weight="bold")
    for r in range(8):
        s.text(px + 34, Y0 + r * (RH + RG) + RH / 2 + 5.3, f"SM{r}", 15, MUTED, anchor="end")
    s.line(px + 40, 248, px + 40 + 250 * U, 248, INK, 1.8, "arr")
    s.text(px + 40 + 250 * U, 266, "time", 15, MUTED, anchor="end", style="italic")


def blk(px, r0, nrows, t0, t1, label, kind):
    x = px + 40 + t0 * U
    y = Y0 + r0 * (RH + RG)
    h = nrows * RH + (nrows - 1) * RG
    if kind == "k1":
        s.box(x, y, (t1 - t0) * U, h, label, NEU_F, NEU, INK, 15, sw=1.5, rx=3)
    elif kind == "k2":
        s.box(x, y, (t1 - t0) * U, h, label, ACC_F, ACC, ACC, 16, weight="bold", sw=2, rx=3)
    else:
        s.rect(x, y, (t1 - t0) * U, h, "url(#hatchPrb)", PRB, 1.5, 3)


# ---- (a) one SM per sequence only
PA = 0
frame(PA, "(a) One SM per sequence")
for r in range(8):
    blk(PA, r, 1, 0, 100, f"s{r}", "k1")
blk(PA, 0, 1, 100, 200, "s8", "k1")
blk(PA, 1, 1, 100, 200, "s9", "k1")
blk(PA, 2, 6, 100, 200, "", "idle")
s.text(PA + 40 + 150 * U, Y0 + 5 * (RH + RG) + 4, "idle SMs", 18, PRB, weight="bold", halo=True)

s.line(436, 6, 436, 270, "#BFBFBF", 1.2)

# ---- (b) dual kernel
PB = 446
frame(PB, "(b) Dual kernel", accent=True)
for r in range(8):
    blk(PB, r, 1, 0, 100, f"s{r}", "k1")
blk(PB, 0, 4, 100, 134, "s8", "k2")
blk(PB, 4, 4, 100, 134, "s9", "k2")
s.text(PB + 40 + 140 * U, Y0 + 2 * (RH + RG) - 6, "one cluster", 16, ACC, anchor="start", style="italic")
s.text(PB + 40 + 140 * U, Y0 + 2 * (RH + RG) + 14, "per sequence", 16, ACC, anchor="start", style="italic")
s.text(PB + 40 + 140 * U, Y0 + 6 * (RH + RG) - 6, "same order", 16, ACC, anchor="start", style="italic")
s.text(PB + 40 + 140 * U, Y0 + 6 * (RH + RG) + 14, "via DSMEM", 16, ACC, anchor="start", style="italic")

# legend
s.rect(40, 274, 22, 14, NEU_F, NEU, 1.5, 2)
s.text(70, 286, "kernel 1: whole sequence on one SM", 15, INK, anchor="start")
s.rect(446 + 40, 274, 22, 14, ACC_F, ACC, 1.5, 2)
s.text(446 + 70, 286, "kernel 2: tail sequences across SMs", 15, INK, anchor="start")
s.save(os.path.join(HERE, "fig_dualkernel.svg"))
