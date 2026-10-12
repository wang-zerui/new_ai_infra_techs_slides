"""fig_bound: two status-quo checkpoint styles. (a) per-rank files fit only the writing layout; (b) gathering a
full state dict funnels every byte through one rank."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 282, "fig_bound")
MONO = "Menlo, monospace"

def fileicon(x, y, w, h, label, stroke=NEU, color=INK, fill="#FFFFFF"):
    f = 12
    s.path(f"M{x},{y} L{x + w - f},{y} L{x + w},{y + f} L{x + w},{y + h} L{x},{y + h} Z", stroke, 2, fill=fill)
    s.path(f"M{x + w - f},{y} L{x + w - f},{y + f} L{x + w},{y + f}", stroke, 1.6)
    s.text(x + w / 2, y + h / 2 + 6, label, 16, color, family=MONO)

# ---- (a) per-rank files
s.text(10, 26, "(a) One file per rank", 20, INK, anchor="start", weight="bold")
s.text(20, 58, "written by 4 GPUs", 17, MUTED, anchor="start", style="italic")
for i in range(4):
    fileicon(20, 72 + i * 48, 130, 38, f"rank{i}.pt")
s.text(260, 58, "resumed on 2 GPUs", 17, MUTED, anchor="start", style="italic")
for j in range(2):
    y = 84 + j * 96
    s.box(260, y, 110, 52, f"GPU {j}", NEU_F, NEU, INK, 18, "bold")
    s.text(392, y + 34, "?", 28, PRB, weight="bold")
for i in range(4):
    s.line(152, 91 + i * 48, 256, 110 + (i // 2) * 96, PRB, 1.8, "arrPrb", dash="5,4")
s.box(240, 236, 170, 36, "offline converter", "#FFFFFF", PRB, PRB, 17, "bold", 2, 4, dash="6,4")

s.line(435, 10, 435, 272, "#BFBFBF", 1.5)

# ---- (b) gather to rank 0
s.text(456, 26, "(b) Gather a full copy", 20, INK, anchor="start", weight="bold")
for i in range(4):
    s.box(466, 64 + i * 50, 100, 40, f"GPU {i}", NEU_F, NEU, INK, 18, "bold")
s.box(640, 136, 96, 46, "rank 0", PRB_F, PRB, PRB, 18, "bold")
for i in range(4):
    s.line(568, 84 + i * 50, 638, 159, PRB, 2, "arrPrb")
fileicon(770, 138, 86, 42, "full.pt")
s.line(738, 159, 768, 159, INK, 2, "arr")
s.text(756, 216, "every byte", 17, PRB, weight="bold")
s.text(756, 238, "through one rank", 17, PRB, weight="bold")
s.text(756, 266, "VeOmni: HF export only", 16, MUTED, style="italic")
s.save(os.path.join(HERE, "fig_bound.svg"))
