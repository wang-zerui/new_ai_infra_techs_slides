"""fig_bytes: 18 bytes of model state per parameter with BF16 weights, FP32 main grads and FP32 Adam
(Megatron dist_optimizer.md table; 12-byte optimizer split from ZeRO K = 12). Heights proportional to bytes."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(480, 330, "fig_bytes: stacked bar of 18 bytes per parameter")

U = 16.5                      # pt per byte
X, W, Y0 = 8, 96, 12
segs = [("BF16 weight", 2, NEU_F), ("FP32 gradient", 4, NEU_F), ("FP32 master weight", 4, "#D9D9D9"),
        ("Adam first moment", 4, "#D9D9D9"), ("Adam second moment", 4, "#D9D9D9")]
y = Y0
mids = []
for name, b, fill in segs:
    h = b * U
    s.rect(X, y, W, h, fill, NEU, 1.8, 0)
    s.text(X + W / 2, y + h / 2 + 6.3, f"{b} B", 18, INK, weight="bold")
    s.text(X + W + 16, y + h / 2 + 6.3, name, 18, INK, anchor="start")
    mids.append((y, y + h))
    y += h


def brace(x, y1, y2, lab1, lab2):
    s.path(f"M{x},{y1 + 2} q8,0 8,8 L{x + 8},{(y1 + y2) / 2 - 8} q0,8 8,8 q-8,0 -8,8 "
           f"L{x + 8},{y2 - 10} q0,8 -8,8", INK, 1.8)
    s.text(x + 22, (y1 + y2) / 2 - 3, lab1, 17, INK, anchor="start", weight="bold")
    s.text(x + 22, (y1 + y2) / 2 + 17, lab2, 17, INK, anchor="start")


brace(300, mids[0][0], mids[1][1], "6 B", "weight + grad")
brace(300, mids[2][0], mids[4][1], "12 B", "optimizer state")
s.save(os.path.join(HERE, "fig_bytes.svg"))
