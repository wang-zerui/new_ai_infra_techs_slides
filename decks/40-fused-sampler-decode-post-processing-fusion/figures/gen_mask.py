"""fig_mask: penalty state. (a) benchmark's vLLM-style path rebuilds two B x V presence masks from host token-ID
lists every step (H2D copy + zeros + scatter) before applying the penalty; (b) HPC-Ops keeps a 1-bit-per-token mask
row per request slot on the GPU: K1 reads the bit, K2 sets the sampled token's bit with atomicOr."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 290
s = SVG(W, H, "fig_mask")

# ---- (a) baseline
s.text(8, 24, "vLLM-style: rebuilt every step", 19, PRB, anchor="start", weight="bold")
AX, AW = 40, 340
s.box(AX, 44, AW, 50, ["token-ID lists on host", "prompt + output"], "#FFFFFF", NEU, size=17)
s.line(AX + AW / 2, 96, AX + AW / 2, 128, PRB, 2.2, "arrPrb")
s.text(AX + AW / 2 + 10, 117, "H2D copy", 16, PRB, anchor="start")
s.box(AX, 130, AW, 54, ["zeros + scatter", "2 bool masks of B × V"], PRB_F, PRB, size=17)
s.line(AX + AW / 2, 186, AX + AW / 2, 214, PRB, 2.2, "arrPrb")
s.box(AX, 216, AW, 44, "apply penalty to logits", PRB_F, PRB, size=17)

s.line(432, 14, 432, H - 10, "#BFBFBF", 1)

# ---- (b) HPC-Ops
s.text(452, 24, "HPC-Ops: resident 1-bit mask", 19, ACC, anchor="start", weight="bold")
s.box(500, 44, 330, 56, ["bitmask row per request slot", "1 bit per vocab token"], ACC_F, ACC, size=17,
      weight="bold", color=ACC)
s.box(458, 176, 182, 64, ["K1", "read bit · scale logit"], ACC_F, ACC, size=17)
s.box(672, 176, 192, 64, ["K2", "atomicOr sampled bit"], ACC_F, ACC, size=17)
s.line(560, 102, 560, 174, ACC, 2.2, "arrAcc")
s.line(770, 174, 770, 102, ACC, 2.2, "arrAcc")
s.line(642, 208, 670, 208, INK, 2.2, "arr")
s.text(665, 266, "row = V / 8 = 15,104 bytes · no host copy", 16, MUTED, style="italic")
s.save(os.path.join(HERE, "fig_mask.svg"))
