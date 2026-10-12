"""fig_lengths: one batch with sequences of 4K, 8K and 32K tokens, drawn to scale (Megatron-Core report §6.4 example)."""
import os
from svgkit import SVG, ACC, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(480, 250, "fig_lengths")
X0, W = 70, 390           # 32K tokens = 390 pt
def x(k):
    return X0 + W * k / 32
rows = [("seq 1", 4), ("seq 2", 8), ("seq 3", 32)]
for i, (name, k) in enumerate(rows):
    y = 30 + i * 58
    s.text(X0 - 10, y + 27, name, 18, INK, anchor="end", weight="bold")
    s.rect(X0, y, x(k) - X0, 38, NEU_F, NEU, 2, 3)
    s.text(x(k) + 8 if k < 32 else x(k) - 10, y + 26, f"{k}K", 18, INK, anchor="start" if k < 32 else "end",
           weight="bold")
# axis
YA = 212
s.line(X0, YA, X0 + W + 6, YA, INK, 2, "arr")
for k in (0, 8, 16, 24, 32):
    s.line(x(k), YA - 5, x(k), YA + 5, INK, 1.6)
    s.text(x(k), YA + 26, f"{k}K" if k else "0", 16, MUTED)
s.text(X0 + W, YA - 10, "tokens", 16, MUTED, anchor="end", style="italic")
s.save(os.path.join(HERE, "fig_lengths.svg"))
