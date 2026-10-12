"""fig_quant: which GLM-5 blocks are INT8 and which are INT4 (area follows weight share)."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(480, 306, "fig_quant: attention/MLP in W8A8, 256 experts per layer in W4A8")

s.text(240, 22, "One GLM-5 layer", 20, INK, weight="bold")

# attention + dense MLP: small area, INT8
s.box(16, 38, 200, 54, ["Attention (MLA)", "W8A8 · INT8"], NEU_F, NEU, size=18)
s.box(16, 100, 200, 54, ["Dense MLP", "W8A8 · INT8"], NEU_F, NEU, size=18)
s.text(116, 176, "~2% of weights", 17, MUTED, style="italic")

# experts: large area, INT4
s.rect(240, 38, 224, 190, ACC_F, ACC, 2.4, 5)
s.text(352, 64, "256 experts", 20, ACC, weight="bold")
s.text(352, 88, "W4A8 · INT4", 19, ACC, weight="bold")
ex, ey = 256, 104
for r in range(3):
    for c in range(4):
        s.rect(ex + c * 48, ey + r * 34, 40, 26, "#FFFFFF", ACC, 1.6, 2)
s.text(352, 212, "top-8 routing + 1 shared", 17, ACC)
s.text(352, 248, "~98% of weights", 17, ACC, style="italic", weight="bold")

# router arrow
s.line(220, 127, 236, 127, INK, 2.2, "arr")

s.text(240, 288, "Calibration: QuaRot · Flex_AWQ_SSZ · msModelSlim", 17, MUTED, style="italic")
s.save(os.path.join(HERE, "fig_quant.svg"))
