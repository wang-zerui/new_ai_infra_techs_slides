"""fig_compose: the three memory levers for optimizer state compose: precision (12 B -> 6 B per parameter with
BF16 moments + remainder), sharding over d data-parallel ranks (6/d B per rank), optional offload to host.
Block heights proportional to bytes (12 B = 120 pt)."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 236, "fig_compose: 12 B -> 6 B -> 6/d B per rank -> host")

U = 10.0            # pt per byte
BASE = 172          # common baseline
W = 120


def block(x, b, lab, sub, fill, stroke, col):
    h = b * U
    s.rect(x, BASE - h, W, h, fill, stroke, 2.2, 3)
    s.text(x + W / 2, BASE + 24, lab, 17, col, weight="bold")
    s.text(x + W / 2, BASE + 44, sub, 15, MUTED)


block(10, 12, "12 B", "FP32 state", PRB_F, PRB, PRB)
block(250, 6, "6 B", "BF16 m v + remainder", ACC_F, ACC, ACC)
# sharded: d slices, one owned
xs = 490
for j in range(4):
    own = j == 0
    s.rect(xs, BASE - 6 * U + j * 15, W, 15, ACC_F if own else "#FFFFFF", ACC if own else "#BFBFBF",
           2 if own else 1, 0, dash=None if own else "3,2")
s.text(xs + W / 2, BASE + 24, "6/d B", 17, ACC, weight="bold")
s.text(xs + W / 2, BASE + 44, "per rank", 15, MUTED)
s.rect(730, BASE - 60, W, 60, "#FFFFFF", NEU, 2, 3, dash="6,4")
s.text(730 + W / 2, BASE - 24, "host", 17, INK, weight="bold")
s.text(730 + W / 2, BASE + 24, "optional", 17, INK, weight="bold")
s.text(730 + W / 2, BASE + 44, "offload", 15, MUTED)


def step(x1, x2, top, bottom):
    y = BASE - 40
    s.line(x1 + 6, y, x2 - 6, y, ACC, 2.6, "arrAcc")
    s.text((x1 + x2) / 2, y - 30, top, 16, ACC, weight="bold")
    s.text((x1 + x2) / 2, y - 12, bottom, 15, MUTED)


step(130, 250, "precision", "BF16 / FP8")
step(370, 490, "shard over d", "dist-opt / FSDP")
step(610, 730, "move", "fewer bytes")
s.save(os.path.join(HERE, "fig_compose.svg"))
