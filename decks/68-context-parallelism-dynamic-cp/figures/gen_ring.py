"""fig_ring: CP=4 ring; KV chunk passes GPU r -> r+1 each step; GPU1's timeline overlaps next-step send/recv
with current-step attention. At step i GPU r holds KV of rank (r - i) mod 4 (TE AttnFuncWithCPAndKVP2P)."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 268, "fig_ring: ring of 4 GPUs + GPU1 timeline (compute vs comm streams)")

# ---------------- ring of four GPUs
BW, BH = 104, 56
pos = {0: (24, 40), 1: (224, 40), 2: (224, 186), 3: (24, 186)}
for g, (x, y) in pos.items():
    hl = g == 1
    s.rect(x, y, BW, BH, ACC_F if hl else NEU_F, ACC if hl else NEU, 2.4 if hl else 2, 5)
    s.text(x + BW / 2, y + 24, f"GPU{g}", 18, ACC if hl else INK, weight="bold")
    s.text(x + BW / 2, y + 46, f"Q{g} · KV{g}", 16, INK)
# arrows clockwise 0->1->2->3->0
s.line(24 + BW + 4, 68, 224 - 4, 68, ACC, 2.4, "arrAcc")
s.line(224 + BW / 2, 40 + BH + 4, 224 + BW / 2, 186 - 4, ACC, 2.4, "arrAcc")
s.line(224 - 4, 214, 24 + BW + 4, 214, ACC, 2.4, "arrAcc")
s.line(24 + BW / 2, 186 - 4, 24 + BW / 2, 40 + BH + 4, ACC, 2.4, "arrAcc")
for (x, y) in ((176, 58), (176, 236)):
    s.text(x, y, "KV", 17, ACC, weight="bold", halo=True)
for (x, y) in ((76 + 18, 140), (276 + 18, 140)):
    s.text(x, y, "KV", 17, ACC, anchor="start", weight="bold", halo=True)

s.line(362, 14, 362, 256, "#BFBFBF", 1.2)

# ---------------- GPU1 timeline
X0, SW = 470, 80            # step width
YC, YM, H = 70, 128, 40
s.text(380, 30, "GPU1 timeline in one attention layer", 19, INK, anchor="start", weight="bold")
s.text(X0 - 10, YC + H / 2 + 6, "compute", 17, INK, anchor="end")
s.text(X0 - 10, YM + H / 2 + 6, "comm", 17, INK, anchor="end")
held = [1, 0, 3, 2]          # KV chunk held by GPU1 at step i: (1 - i) mod 4
for i, kv in enumerate(held):
    x = X0 + i * (SW + 4)
    s.text(x + SW / 2, YC - 10, f"step {i}", 15, MUTED)
    s.box(x, YC, SW, H, f"Q1 · KV{kv}", NEU_F, NEU, size=16)
    if i < 3:
        nxt = held[i + 1]
        s.box(x + 4, YM, SW - 8, H, [f"get KV{nxt}"], ACC_F, ACC, ACC, 16)
xm = X0 + 4 * (SW + 4)
s.box(xm, YC, 44, H, "+", ACC_F, ACC, ACC, 22, "bold")
s.text(xm + 22, YC - 10, "merge", 15, MUTED)
s.text(X0, YM + H + 30, "next KV arrives while this step computes", 16, MUTED, anchor="start", style="italic")
s.line(X0, 236, 862, 236, INK, 2, "arr")
s.text(862, 260, "time", 18, INK, anchor="end")
s.save(os.path.join(HERE, "fig_ring.svg"))
