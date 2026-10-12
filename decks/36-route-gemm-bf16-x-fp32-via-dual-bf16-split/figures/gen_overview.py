"""fig_overview: Route GEMM = split W once at load (1), one fused kernel with two BF16 MMAs and FP32 accumulators (2),
small-M tile mapping (3)."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED, badge

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 298
s = SVG(W, H, "fig_overview: load-time split + one fused kernel")

# containers
s.rect(6, 6, 262, 286, "#FFFFFF", ACC, 1.8, 8, dash="7,4")
s.text(20, 32, "model load · once", 19, ACC, anchor="start", weight="bold")
s.rect(290, 6, 574, 286, "#FFFFFF", ACC, 1.8, 8, dash="7,4")
badge(s, 316, 25, "2")
s.text(338, 32, "every forward · one kernel", 19, ACC, anchor="start", weight="bold")

# load side
s.box(18, 108, 84, 60, ["W", "FP32"], NEU_F, NEU, size=19)
s.box(154, 44, 100, 56, ["W_hi", "BF16"], ACC_F, ACC, size=19)
s.box(154, 176, 100, 56, ["W_lo", "BF16"], ACC_F, ACC, size=19)
s.path("M102,138 L126,138 L126,72 L152,72", INK, 2.2, "arr")
s.path("M126,138 L126,204 L152,204", INK, 2.2, "arr")
badge(s, 126, 138, "1")
s.text(18, 268, "cached with the weight", 17, MUTED, anchor="start", style="italic")

# kernel side
s.box(300, 110, 136, 56, ["x tile (BF16)", "loaded once"], NEU_F, NEU, size=18)
s.box(476, 44, 146, 56, ["BF16 MMA", "→ acc_hi FP32"], ACC_F, ACC, size=18)
s.box(476, 176, 146, 56, ["BF16 MMA", "→ acc_lo FP32"], ACC_F, ACC, size=18)
s.box(662, 96, 188, 84, ["epilogue", "acc_hi + acc_lo/256"], ACC_F, ACC, color=ACC, weight="bold", size=18)
s.box(690, 226, 132, 50, "FP32 scores", NEU_F, NEU, size=18)
# weights into MMAs (enter upper part of each MMA box)
s.line(254, 64, 474, 64, ACC, 2.2, "arrAcc")
s.line(254, 218, 474, 218, ACC, 2.2, "arrAcc")
# x tile feeds both MMAs (enter lower part)
s.path("M436,138 L456,138 L456,86 L474,86", INK, 2.2, "arr")
s.path("M456,138 L456,190 L474,190", INK, 2.2, "arr")
# MMAs into epilogue
s.path("M622,72 L642,72 L642,118 L660,118", INK, 2.2, "arr")
s.path("M622,204 L642,204 L642,158 L660,158", INK, 2.2, "arr")
# store once
s.line(756, 180, 756, 224, INK, 2.2, "arr")
s.text(764, 207, "store once", 16, MUTED, anchor="start", style="italic")
# (3) small-M mapping strip
s.rect(306, 246, 316, 36, "#FFFFFF", ACC, 1.5, 6, dash="4,3")
badge(s, 326, 264, "3")
s.text(346, 270, "small-M tiles: swap-AB + split-K", 17, ACC, anchor="start")
s.save(os.path.join(HERE, "fig_overview.svg"))
