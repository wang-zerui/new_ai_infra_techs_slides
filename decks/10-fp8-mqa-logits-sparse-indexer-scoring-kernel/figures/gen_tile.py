"""Slide 7 (mechanism 1): SM90 tile mapping, H = 64 heads: keys on the MMA's M side (64 per warpgroup),
(token, head) pairs on N (2 tokens x 64 heads = 128), K = head_dim = 128 (sm90_fp8_mqa_logits.cuh)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 282
s = SVG(W, H, "MQA logits tile mapping")
LIGHT = "#F4F8FC"
# geometry
kx, kw = 96, 64          # key tile (A)
ax, aw = 196, 260        # accumulator / Q tile width (N = 128)
y0, hh = 70, 184         # rows = 256 keys
qy, qh = 8, 46           # Q tile (B) height (K = 128)
# Q tile (B): two tokens x 64 heads
s.rect(ax, qy, aw / 2, qh, ACC_F, ACC, 2, 3)
s.rect(ax + aw / 2, qy, aw / 2, qh, ACC_F, ACC, 2, 3)
s.text(ax + aw / 4, qy + 20, "token t", 16, INK, weight="bold")
s.text(ax + aw / 4, qy + 39, "64 heads", 15, INK)
s.text(ax + 3 * aw / 4, qy + 20, "token t+1", 16, INK, weight="bold")
s.text(ax + 3 * aw / 4, qy + 39, "64 heads", 15, INK)
s.text(ax - 10, qy + 22, "Q tile", 17, INK, anchor="end", weight="bold")
s.text(ax - 10, qy + 42, "K = 128", 16, MUTED, anchor="end")
# key tile (A): 4 warpgroup slabs
for i in range(4):
    y = y0 + i * hh / 4
    s.rect(kx, y, kw, hh / 4, ACC_F if i % 2 == 0 else "#C9DDF0", ACC, 2, 2)
    s.text(kx + kw / 2, y + hh / 8 + 6, f"WG{i}", 16, INK, weight="bold")
    s.text(kx - 8, y + hh / 8 + 6, f"{64*i}–{64*i+63}", 15, MUTED, anchor="end")
s.text(kx + kw / 2 - 20, y0 + hh + 18, "256 keys × 128", 16, MUTED)
# accumulators
for i in range(4):
    y = y0 + i * hh / 4
    for j in range(2):
        s.rect(ax + j * aw / 2, y, aw / 2, hh / 4, LIGHT, "#9DB7D5", 1.2, 0)
s.rect(ax, y0, aw, hh, "none", NEU, 2, 0)
s.text(ax + aw / 2, y0 + hh / 2 - 6, "FP32 accumulators", 18, INK, weight="bold")
s.text(ax + aw / 2, y0 + hh / 2 + 16, "256 × 128 (token, head)", 16, INK)
s.text(ax + aw / 2, y0 + hh + 18, "registers (SM90) · TMEM (SM100)", 16, MUTED)
# reduction arrow + logits
s.arrow(ax + aw + 4, y0 + hh / 2, 562, y0 + hh / 2, ACC, 2.6)
s.text(510, y0 + hh / 2 - 36, "ReLU · w", 17, ACC, weight="bold")
s.text(510, y0 + hh / 2 - 14, "Σ 64 heads", 17, ACC, weight="bold")
lx = 568
for j in range(2):
    s.rect(lx + j * 34, y0, 34, hh, ACC_F, ACC, 2, 2)
    s.text(lx + j * 34 + 17, y0 - 8, "t" if j == 0 else "t+1", 16, INK, weight="bold")
s.text(lx + 34, y0 + hh + 18, "256 × 2 logits", 16, INK)
# side notes
nx = 664
s.text(nx, y0 + 30, "M = keys", 17, INK, anchor="start", weight="bold")
s.text(nx, y0 + 52, "64 per warpgroup", 16, MUTED, anchor="start")
s.text(nx, y0 + 92, "N = token × head", 17, INK, anchor="start", weight="bold")
s.text(nx, y0 + 114, "128 = (128 / H) × H", 16, MUTED, anchor="start")
s.text(nx, y0 + 154, "one key read", 17, INK, anchor="start", weight="bold")
s.text(nx, y0 + 176, "feeds 128 scores", 16, MUTED, anchor="start")
s.save(os.path.join(HERE, "fig_tile.svg"))
