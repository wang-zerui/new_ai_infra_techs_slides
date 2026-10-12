"""fig_calib: the scaling-calibration step (AWQ-style search), as named by Flex_AWQ_SSZ.

AWQ's own search: s = s_X^alpha per input channel, alpha chosen by grid search on calibration
statistics (arXiv 2306.00978 Eq. 5). The error curve is schematic, not measured.
"""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 292, "fig_calib: per-channel scaling search and asymmetric weight quantization")

Y, BH = 56, 104

# calibration statistics
s.rect(10, Y, 196, BH, NEU_F, NEU, 2, 4)
s.text(108, Y + 24, "activation stats", 17, INK, weight="bold")
for i, h in enumerate([18, 30, 24, 40, 22, 34, 26]):
    s.rect(28 + i * 24, Y + BH - 12 - h, 16, h, "#FFFFFF", NEU, 1.4, 1)
s.text(108, Y + BH + 20, "from calibration prompts", 16, MUTED)
s.line(206, Y + BH / 2, 238, Y + BH / 2, INK, 2.2, "arr")

# the search
s.rect(240, Y, 210, BH, ACC_F, ACC, 3, 5)
s.text(306, Y + 36, "s = s", 21, INK, anchor="start")
s.text(353, Y + 42, "X", 14, INK, anchor="start")
s.text(364, Y + 26, "α", 16, ACC, anchor="start", weight="bold")
s.text(345, Y + 64, "per input channel", 17, INK)
s.text(345, Y + 88, "α by grid search", 17, ACC, weight="bold")
# schematic error curve under the search box
AX0, AX1, AY0, AY1 = 252, 438, 262, 196
s.line(AX0, AY1, AX0, AY0, INK, 1.8)
s.line(AX0, AY0, AX1, AY0, INK, 1.8)
s.path(f"M {AX0 + 6} {AY1 + 4} Q {AX0 + 58} {AY0 + 2} {AX0 + 96} {AY0 - 2} "
       f"Q {AX0 + 140} {AY0 - 2} {AX1 - 6} {AY1 - 10}", ACC, 2.4)
s.add(f'<circle cx="{AX0 + 96}" cy="{AY0 - 2}" r="5" fill="{ACC}"/>')
s.text(AX0 + 96, AY0 - 12, "α*", 16, ACC, weight="bold")
s.text(AX0 - 4, AY1 - 6, "error", 15, MUTED, anchor="start")
s.text(AX1, AY0 + 18, "α = 1", 15, MUTED, anchor="end")
s.text(AX0, AY0 + 18, "0", 15, MUTED)
s.line(450, Y + BH / 2, 482, Y + BH / 2, INK, 2.2, "arr")

# apply
s.box(484, Y, 170, BH, ["fold s into W", "and into the", "previous op"], ACC_F, ACC, color=INK,
      size=17, sw=3)
s.line(654, Y + BH / 2, 686, Y + BH / 2, INK, 2.2, "arr")

# quantize
s.box(688, Y, 172, BH, ["quantize W", "scale + offset", "per channel"], ACC_F, ACC, color=INK,
      size=17, sw=3)
s.text(774, Y + BH + 20, "\"SSZ\" in the report", 16, MUTED, style="italic")
s.text(774, Y + BH + 42, "is not defined there", 16, MUTED, style="italic")
s.save(os.path.join(HERE, "fig_calib.svg"))
