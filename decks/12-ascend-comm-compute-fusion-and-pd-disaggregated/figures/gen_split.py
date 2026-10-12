"""fig_split: AllReduce (before) versus ReduceScatter + sharded window + AllGather (after)."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 252, "fig_split: the sharded window between ReduceScatter and AllGather")

H = 46

# ---------------- before
s.text(12, 47, "Before", 19, INK, anchor="start", weight="bold")
s.box(118, 18, 170, H, "Attention + o_proj", NEU_F, NEU, size=18)
s.rect(296, 18, 150, H, PRB_F, PRB, 2.4, 3)
s.text(371, 47, "AllReduce", 18, PRB, weight="bold")
s.box(454, 18, 190, H, "RMSNorm + router", NEU_F, NEU, size=18)
s.box(652, 18, 206, H, "All-to-all + experts", NEU_F, NEU, size=18)
s.line(288, 41, 294, 41, INK, 2.2)
s.line(446, 41, 452, 41, INK, 2.2)
s.line(644, 41, 650, 41, INK, 2.2)
s.text(549, 86, "every rank holds all tokens", 18, PRB, weight="bold")

# divider
s.line(12, 108, 858, 108, MUTED, 1.2, dash="6,4")

# ---------------- after
s.text(12, 157, "After", 19, ACC, anchor="start", weight="bold")
s.box(118, 128, 170, H, "Attention + o_proj", NEU_F, NEU, size=18)
s.rect(296, 128, 150, H, ACC_F, ACC, 2.4, 3)
s.text(371, 157, "ReduceScatter", 18, ACC, weight="bold")
s.box(454, 128, 190, H, "RMSNorm + router", ACC_F, ACC, color=ACC, size=18)
s.box(652, 128, 112, H, "All-to-all", ACC_F, ACC, color=ACC, size=18)
s.rect(772, 128, 86, H, ACC_F, ACC, 2.4, 3)
s.text(815, 157, "AllGather", 17, ACC, weight="bold")
s.line(288, 151, 294, 151, INK, 2.2)
s.line(446, 151, 452, 151, INK, 2.2)
s.line(644, 151, 650, 151, INK, 2.2)
s.line(764, 151, 770, 151, INK, 2.2)

# sharded window bracket
s.line(452, 192, 452, 206, ACC, 2)
s.line(766, 192, 766, 206, ACC, 2)
s.line(452, 206, 766, 206, ACC, 2)
s.text(609, 230, "sharded window: 1/TP of the tokens", 18, ACC, weight="bold")

s.save(os.path.join(HERE, "fig_split.svg"))
