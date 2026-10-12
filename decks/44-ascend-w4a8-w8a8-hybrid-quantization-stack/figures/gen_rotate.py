"""fig_rotate: why rotating the model removes the outlier channels (QuaRot).

Left: per-channel activation magnitudes with a few outlier channels (illustrative, not measured).
Right: the same activations after a Hadamard rotation that is fused into the weights, so the
network's output is unchanged. Bar heights are schematic.
"""
import os
import random
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 300, "fig_rotate: Hadamard rotation removes outlier activation channels")

random.seed(7)
BASE_Y = 212                      # baseline of both bar panels
PW = 300                          # panel width


def panel(x0, title, heights, colors, note, note_color):
    s.text(x0 + PW / 2, 34, title, 19, INK, weight="bold")
    s.line(x0, BASE_Y, x0 + PW, BASE_Y, NEU, 2)
    n = len(heights)
    bw = PW / n
    for i, (h, c) in enumerate(zip(heights, colors)):
        fill, stroke = (PRB_F, PRB) if c == "p" else (ACC_F, ACC) if c == "a" else ("#FFFFFF", NEU)
        s.rect(x0 + i * bw + 1.5, BASE_Y - h, bw - 3, h, fill, stroke, 1.4, 1)
    s.text(x0 + PW / 2, BASE_Y + 24, "activation channels", 16, MUTED)
    s.text(x0 + PW / 2, BASE_Y + 48, note, 17, note_color, weight="bold")


small = [random.uniform(10, 26) for _ in range(18)]
before = list(small)
for i in (4, 11):
    before[i] = 150
panel(20, "before", before, ["p" if h > 100 else "n" for h in before],
      "a few channels dominate the range", PRB)

after = [random.uniform(44, 62) for _ in range(18)]
panel(550, "after rotation", after, ["a"] * 18, "every channel in the same range", ACC)

# the quantization-grid hint on both panels
for k in range(1, 4):
    y = BASE_Y - k * 38
    s.line(20, y, 320, y, "#D9D9D9", 1.2, dash="4,3")
    s.line(550, y, 850, y, "#D9D9D9", 1.2, dash="4,3")
s.text(326, BASE_Y - 114, "INT4", 15, MUTED, anchor="start")
s.text(326, BASE_Y - 96, "levels", 15, MUTED, anchor="start")

# the transform in the middle
s.path("M 374 120 L 470 120 L 470 104 L 512 136 L 470 168 L 470 152 L 374 152 Z", INK, 1.8,
       fill="#FFFFFF")
s.text(443, 92, "rotate", 19, ACC, weight="bold")
s.box(348, 196, 190, 74, ["(X Q) (Qᵀ W) = X W", "Q orthogonal", "Qᵀ folded into W"], ACC_F, ACC,
      color=INK, size=16, sw=2.5)
s.save(os.path.join(HERE, "fig_rotate.svg"))
