"""fig_bf16: MoE gradient sync (§3.4.1). (a) tree/ring reduce-scatter in BF16 accumulates partial sums with
low-precision adders at every hop; (b) V4: all-to-all of BF16 (stochastically rounded) gradients, then each rank
sums the pieces of its own shard in FP32. DP = 4 shown."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 292
s = SVG(W, H, "fig_bf16")
BH = 38
ys = [60 + i * 54 for i in range(4)]

# ---------------- (a)
s.text(205, 22, "(a) Ring reduce-scatter in BF16", 20, INK, weight="bold")
labels = ["R0: g0", "R1: + g1", "R2: + g2", "R3: + g3"]
for i, y in enumerate(ys):
    s.box(70, y, 150, BH, labels[i], NEU_F, NEU, size=18)
    if i < 3:
        s.line(145, y + BH + 2, 145, ys[i + 1] - 3, INK, 2.2, "arr")
        s.text(240, y + BH + 15, "BF16 add + round", 17, PRB, anchor="start", weight="bold")
s.text(205, 282, "rounding error at every hop", 19, PRB, weight="bold")

s.line(420, 10, 420, H - 10, "#BFBFBF", 1.2)

# ---------------- (b)
s.text(645, 22, "(b) V4: all-to-all, then FP32 sum", 20, ACC, weight="bold")
s.text(446, 48, "BF16 with stochastic rounding", 17, MUTED, anchor="start")
for i, y in enumerate(ys):
    yy = y
    s.box(446, yy, 148, BH, f"R{i} gradient", "#FFFFFF", NEU, size=18)
RX, RY, RW, RHh = 690, 112, 170, 96
for i, y in enumerate(ys):
    yy = y + BH / 2
    s.line(596, yy, RX - 3, RY + 18 + i * 20, ACC, 2.2, "arrAcc")
s.box(RX, RY, RW, RHh, ["R2 shard", "Σ in FP32"], ACC_F, ACC, color=ACC, weight="bold", sw=2.4)
s.text(RX + RW / 2, RY + RHh + 26, "same on every rank", 17, MUTED)
s.text(645, 282, "one rounding, FP32 accumulation", 19, ACC, weight="bold")
s.save(os.path.join(HERE, "fig_bf16.svg"))
