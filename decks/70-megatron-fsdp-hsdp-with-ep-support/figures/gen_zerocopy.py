"""fig_zerocopy: (a) per-parameter uniform sharding (FSDP2) needs copies into a staging buffer;
(b) Megatron-FSDP flattens a module's parameters into one buffer and cuts it into equal contiguous shards
(uneven per parameter); collectives run in place on NCCL-registered buffers, two of which alternate."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 286, "fig_zerocopy: per-parameter shards + copies vs flat buffer shards in place")
P = [("P1", 240), ("P2", 160), ("P3", 100)]
H = 30

# ---------------- (a) per-parameter shards
s.text(8, 24, "(a) Per-parameter shards (FSDP2)", 19, INK, anchor="start", weight="bold")
x0, y = 48, 52
r0_pieces = []
for name, L in P:
    s.text(x0 - 8, y + 21, name, 16, INK, anchor="end", weight="bold")
    q = L / 4
    for j in range(4):
        own = j == 0
        s.rect(x0 + j * q, y, q, H, ACC_F if own else "#FFFFFF", ACC if own else "#BFBFBF", 1.6 if own else 1, 0,
               dash=None if own else "3,2")
    r0_pieces.append((x0 + q / 2, y + H))
    y += H + 18
# staging buffer
bx, by, bw = 300, 82, 100
s.box(bx, by, bw, 50, ["staging", "buffer"], PRB_F, PRB, PRB, 16)
for (px, py) in r0_pieces:
    s.path(f"M{px:.1f},{py - H / 2:.1f} C{px + 120:.1f},{py - H / 2:.1f} {bx - 60},{by + 25} {bx - 2},{by + 25}",
           PRB, 1.8, "arrPrb")
s.text(bx + bw / 2, by + 74, "copy in and out", 17, PRB, weight="bold")
s.text(bx + bw / 2, by + 94, "every collective", 17, PRB, weight="bold")
s.text(48, 218, "rank 0 owns a quarter of each parameter", 15, MUTED, anchor="start", style="italic")

s.line(432, 12, 432, 276, "#BFBFBF", 1.2)

# ---------------- (b) flat buffer
s.text(448, 24, "(b) Flat per-module buffer", 19, INK, anchor="start", weight="bold")
fx, fy, scale = 466, 78, 0.76
total = sum(L for _, L in P) * scale
xx = fx
for name, L in P:
    w = L * scale
    s.text(xx + w / 2, fy - 10, name, 16, INK, weight="bold")
    xx += w
q = total / 4
for j in range(4):
    own = j == 0
    s.rect(fx + j * q, fy, q, H + 8, ACC_F if own else "#FFFFFF", ACC if own else "#BFBFBF", 1.8 if own else 1, 0,
           dash=None if own else "3,2")
    s.text(fx + j * q + q / 2, fy + H + 30, f"rank {j}", 15, MUTED)
# parameter boundaries inside the buffer
xx = fx
for name, L in P[:-1]:
    xx += L * scale
    s.line(xx, fy - 4, xx, fy + H + 12, INK, 2.2, dash="4,3")
s.text(fx + total / 2, 176, "AG and RS run in place", 17, ACC, weight="bold")
s.text(fx + total / 2, 196, "on NCCL-registered memory", 17, ACC, weight="bold")
# double buffer
s.box(fx + 40, 222, 130, 40, "buffer A", ACC_F, ACC, ACC, 16, "bold")
s.box(fx + 210, 222, 130, 40, "buffer B", ACC_F, ACC, ACC, 16, "bold")
s.line(fx + 172, 242, fx + 208, 242, ACC, 2, "arrAcc", start_marker="arrAcc")
s.text(fx + total / 2, 282, "units alternate between two buffers", 15, MUTED, style="italic")
s.save(os.path.join(HERE, "fig_zerocopy.svg"))
