"""fig_context: one kernel stack serves pre-training, RL and serving; the same token must give the same bits
(DeepSeek-V4 §3.3: bitwise alignment among pre-training, post-training and inference)."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED, GOOD

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 480, 330
s = SVG(W, H, "fig_context")

xs = [6, 166, 326]
BW = 148
# outputs row
for i, x in enumerate(xs):
    s.box(x + 34, 4, 80, 36, "y(t)", "#FFFFFF", GOOD, GOOD, 19, weight="bold", sw=2)
    if i < 2:
        s.text(x + BW + 9, 29, "=", 24, GOOD, weight="bold")
# consumers
labels = [["pre-training"], ["RL rollout", "+ trainer"], ["inference", "serving"]]
for x, lab in zip(xs, labels):
    s.box(x, 82, BW, 62, lab, NEU_F, NEU, INK, 18, sw=2)
    s.line(x + BW / 2, 80, x + BW / 2, 42, INK, 2.2, "arr")
# shared kernel stack
s.rect(6, 196, 468, 126, ACC_F, ACC, 2, 6)
s.text(240, 220, "one shared kernel stack", 18, ACC, weight="bold")
s.box(22, 236, 210, 72, ["TileLang", "fused kernels"], "#FFFFFF", ACC, INK, 18, sw=2)
s.box(248, 236, 210, 72, ["DeepGEMM", "GEMMs"], "#FFFFFF", ACC, INK, 18, sw=2)
for x in xs:
    s.line(x + BW / 2, 194, x + BW / 2, 146, ACC, 2.4, "arrAcc")
s.save(os.path.join(HERE, "fig_context.svg"))
