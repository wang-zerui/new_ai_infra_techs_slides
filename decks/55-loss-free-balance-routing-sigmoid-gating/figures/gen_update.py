"""fig_update: sign feedback (Algorithm 1 of arXiv 2408.15664). Token counts of step t vs their mean;
overloaded experts get b -= u, underloaded b += u; the new bias is used from step t+1 on (schematic)."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 290
s = SVG(W, H, "fig_update: per-expert load vs mean -> +-u bias steps; applied at the next step")

BASE, BW, STEP, X0 = 196, 50, 76, 70
heights = [140, 60, 108, 46, 124, 74]                 # schematic token counts
mean = sum(heights) / len(heights)
s.add(f'<text x="24" y="{BASE - 75}" text-anchor="middle" font-size="16" fill="{MUTED}" font-style="italic" '
      f'transform="rotate(-90 24 {BASE - 75})">tokens at step t</text>')
s.line(44, BASE, X0 + 5 * STEP + BW + 14, BASE, INK, 1.6)
for i, h in enumerate(heights):
    x = X0 + i * STEP
    over = h > mean
    s.rect(x, BASE - h, BW, h, PRB_F if over else "#FFFFFF", PRB if over else NEU, 2, 2)
    s.box(x - 4, BASE + 10, BW + 8, 34, "−u" if over else "+u", ACC_F, ACC, color=ACC, size=19, weight="bold",
          sw=1.8)
    s.text(x + BW / 2, BASE + 72, f"E{i + 1}", 18, INK)
ym = BASE - mean
s.line(52, ym, X0 + 5 * STEP + BW + 14, ym, INK, 2, dash="7,5")
s.text(X0 + 5 * STEP + BW + 20, ym + 6, "mean", 17, INK, anchor="start", style="italic")
s.text(10, BASE + 32, "bias", 17, ACC, anchor="start", weight="bold")

# right: one feedback round, in time order
RX, RW = 568, 296
s.box(RX, 14, RW, 46, "step t: route with b", NEU_F, NEU)
s.line(RX + RW / 2, 62, RX + RW / 2, 106, INK, 2.2, "arr")
s.text(RX + RW / 2 + 10, 90, "count c per expert", 16, MUTED, anchor="start", style="italic")
s.box(RX, 108, RW, 56, "b ← b + u · sign(mean − c)", ACC_F, ACC, color=ACC, size=19, weight="bold")
s.line(RX + RW / 2, 166, RX + RW / 2, 210, INK, 2.2, "arr")
s.text(RX + RW / 2 + 10, 194, "no gradient", 16, MUTED, anchor="start", style="italic")
s.box(RX, 212, RW, 46, "step t+1: route with new b", NEU_F, NEU)
s.save(os.path.join(HERE, "fig_update.svg"))
