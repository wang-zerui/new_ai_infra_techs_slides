"""fig_thd: packed stream under Ulysses SP (P = 4, as in VeOmni's SequenceParallelCollator): pad to a multiple
of P, slice for per-token ops, all-to-all so attention sees the full stream for H/P heads with the same
cu_seqlens. Right: the Megatron-Core THD + CP counterpart."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 300, "fig_thd")
toks = list("ABCDEFGHIJ") + ["", ""]
seg = [0, 0, 0, 0, 1, 1, 2, 2, 2, 3, 4, 4]
CW, CG = 30, 3

def chip(x, y, i):
    if toks[i]:
        s.box(x, y, CW, 28, toks[i], NEU_F if seg[i] % 2 == 0 else "#FFFFFF", NEU, INK, 16, "bold", 1.5, 3)
    else:
        s.rect(x, y, CW, 28, "url(#hatchGray)", NEU, 1.5, 3)

# row 1: packed + padded stream
s.text(136, 36, "packed", 17, INK, anchor="end", weight="bold")
for i in range(12):
    chip(148 + i * (CW + CG), 16, i)
s.text(148 + 12 * (CW + CG) + 4, 36, "pad to 12", 16, MUTED, anchor="start", style="italic")

# row 2: per-rank slices
s.text(136, 112, "per token", 17, INK, anchor="end", weight="bold")
s.text(136, 132, "ops", 17, INK, anchor="end", weight="bold")
BX, BWID, BG = 148, 112, 8
for r in range(4):
    x = BX + r * (BWID + BG)
    s.rect(x, 90, BWID, 54, "#FFFFFF", NEU, 1.6, 4)
    s.text(x + BWID / 2, 160, f"rank {r}", 15, MUTED)
    for k in range(3):
        chip(x + 6 + k * (CW + CG), 103, 3 * r + k)
s.line(BX + 2 * (BWID + BG) - BG / 2, 166, BX + 2 * (BWID + BG) - BG / 2, 194, ACC, 2.4, "arrAcc")
s.text(BX + 2 * (BWID + BG) + 8, 186, "all-to-all", 17, ACC, anchor="start", weight="bold")

# row 3: attention on rank 0: full stream, H/P heads, cu_seqlens segments
s.text(136, 226, "attention", 17, ACC, anchor="end", weight="bold")
s.text(136, 246, "(rank 0)", 15, MUTED, anchor="end")
for i in range(12):
    chip(148 + i * (CW + CG), 206, i)
s.text(148 + 12 * (CW + CG) + 4, 226, "H/P heads", 16, ACC, anchor="start", weight="bold")
# segment brackets
starts = [0, 4, 6, 9, 10]
for a, b in zip(starts, starts[1:]):
    xa = 148 + a * (CW + CG)
    xb = 148 + b * (CW + CG) - CG
    s.path(f"M{xa + 1},{240} L{xa + 1},{248} L{xb - 1},{248} L{xb - 1},{240}", ACC, 2)
s.text(148 + 5 * (CW + CG), 274, "segments from cu_seqlens", 16, ACC, style="italic")

# right: Megatron counterpart
RX = 652
s.rect(RX, 86, 212, 196, "#FFFFFF", ACC, 2, 6, dash="7,4")
s.text(RX + 106, 114, "Megatron-Core", 18, ACC, weight="bold")
for i, t in enumerate(["THD [T, heads, dim]", "ring CP over T", "padded + unpadded", "cu_seqlens",
                       "Dynamic-CP: deck 68"]):
    s.text(RX + 106, 146 + i * 27, t, 16, INK if i < 4 else ACC, weight="bold" if i == 4 else "normal")
s.save(os.path.join(HERE, "fig_thd.svg"))
