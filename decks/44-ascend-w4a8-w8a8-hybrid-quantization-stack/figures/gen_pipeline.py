"""fig_pipeline: the four steps that turn a BF16 checkpoint into a served W4A8/W8A8 model.

Steps 1-3 are the msModelSlim offline path named in GLM-5 report section 5; step 4 is what the
vLLM-Ascend runtime does with the result. INT4 QAT (report section 2.4.3) feeds the checkpoint
that enters step 1.
"""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 300, "fig_pipeline: rotate, calibrate, assign, serve")

Y, BH = 92, 92
XS = [136, 320, 504, 688]
W = 160
TITLES = [
    ("❶ Rotate", "QuaRot Hadamard", "fused into W"),
    ("❷ Calibrate", "Flex_AWQ_SSZ", "per-channel scales"),
    ("❸ Assign", "W4A8 experts", "W8A8 attn + MLP"),
    ("❹ Serve", "vLLM-Ascend", "int4 W · int8 X"),
]
for x, (a, b, c) in zip(XS, TITLES):
    s.rect(x, Y, W, BH, ACC_F, ACC, 3, 5)
    s.text(x + W / 2, Y + 28, a, 19, ACC, weight="bold")
    s.text(x + W / 2, Y + 54, b, 17, INK)
    s.text(x + W / 2, Y + 76, c, 17, INK)
# fat hollow arrows between the steps
for x in XS[:-1]:
    x0 = x + W + 2
    s.path(f"M {x0} {Y + 36} L {x0 + 12} {Y + 36} L {x0 + 12} {Y + 24} L {x0 + 22} {Y + 46} "
           f"L {x0 + 12} {Y + 68} L {x0 + 12} {Y + 56} L {x0} {Y + 56} Z", INK, 1.6, fill="#FFFFFF")

# inputs and output
s.box(10, Y + 14, 110, 64, ["BF16", "checkpoint"], "#FFFFFF", NEU, size=17)
s.line(120, Y + 46, 134, Y + 46, INK, 2.2, "arr")
s.box(864 - 110, Y + BH + 56, 110, 60, ["served", "model"], "#FFFFFF", NEU, size=17)
s.line(XS[3] + W / 2, Y + BH + 2, XS[3] + W / 2, Y + BH + 54, INK, 2.2, "arr")

# calibration data feeds step 2
s.box(300, Y + BH + 52, 200, 58, ["calibration prompts"], NEU_F, NEU, size=16)
s.line(400, Y + BH + 52, 400, Y + BH + 2, MUTED, 2.2, "arrGray")

# QAT feeds the incoming checkpoint
s.box(10, 24, 240, 46, ["INT4 QAT during SFT"], ACC_F, ACC, color=INK, size=16, dash="6,4", sw=2.5)
s.line(130, 70, 130, Y - 2, ACC, 2.2, "arrAcc")
s.text(258, 52, "one kernel, train = infer", 16, MUTED, anchor="start", style="italic")

# scope note
s.text(864, 40, "msModelSlim offline", 16, MUTED, anchor="end", style="italic")
s.text(864, 62, "steps ❶❷❸", 16, MUTED, anchor="end", style="italic")
s.save(os.path.join(HERE, "fig_pipeline.svg"))
