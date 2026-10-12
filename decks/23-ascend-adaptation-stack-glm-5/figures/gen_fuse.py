"""fig_fuse: 13 MLA pre-processing operators collapse into one super operator using both unit types."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(480, 300, "fig_fuse: MLAPO fuses 13 pre-processing operators; Vector and Cube units run in parallel")

s.text(240, 22, "MLA pre-processing", 20, INK, weight="bold")

# before: 13 small ops in two rows
x0, y0, w, h, g = 60, 36, 28, 32, 6
for i in range(13):
    col, row = i % 7, i // 7
    s.rect(x0 + col * (w + g), y0 + row * (h + 8), w, h, NEU_F, NEU, 1.8, 2)
s.text(240, 134, "13 operators · 13 launches", 18, INK)

# fat hollow arrow down
s.path("M 222 146 L 258 146 L 258 166 L 272 166 L 240 192 L 208 166 L 222 166 Z",
       INK, 1.6, fill="#FFFFFF")

# after: one super operator over two unit lanes
s.rect(20, 200, 440, 86, ACC_F, ACC, 2.4, 5)
s.text(240, 224, "MLAPO super operator", 19, ACC, weight="bold")
s.box(34, 234, 200, 42, "Cube unit", "#FFFFFF", ACC, color=ACC, size=18)
s.box(246, 234, 200, 42, "Vector unit", "#FFFFFF", ACC, color=ACC, size=18)
s.save(os.path.join(HERE, "fig_fuse.svg"))
