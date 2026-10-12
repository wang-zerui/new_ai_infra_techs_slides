"""fig_model: the DeepSeek-V3 stack in pipeline order — embedding, 3 dense, 58 MoE, MTP, loss.

Heights are qualitative cost classes from [MC] Fig. 41 legend (light: embedding/loss/dense; heavy: MoE, MTP).
"""
import os
from svgkit import SVG, ACC, PRB, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 214, "fig_model: DeepSeek-V3 layer sequence with cost classes")
BASE, HL, HH = 104, 38, 80
x = 112
GAP = 12

def blk(x, w, h, fill, stroke, label=None, color=INK, bold=False, size=17):
    s.rect(x, BASE - h, w, h, fill, stroke, 1.6, 2)
    if label:
        s.text(x + w / 2, BASE - h / 2 + 6, label, size, color, weight="bold" if bold else "normal")

# cost-class guide
s.text(96, BASE - HL + 6, "light", 16, MUTED, anchor="end", style="italic")
s.text(96, BASE - HH + 6, "heavy", 16, MUTED, anchor="end", style="italic")
groups = []
blk(x, 40, HL, "#FFFFFF", NEU, "E", bold=True); groups.append((x, 40, "embed")); x += 40 + GAP
gx = x
for i in range(3):
    blk(x, 22, HL, NEU_F, NEU); x += 22 + 3
groups.append((gx, x - 3 - gx, "3 dense")); x += GAP - 3
gx = x
for i in range(58):
    blk(x, 6.8, HH, "#BFBFBF", NEU); x += 6.8 + 1.3
groups.append((gx, x - 1.3 - gx, "58 MoE layers")); x += GAP - 1.3
blk(x, 52, HH, "#BFBFBF", NEU, "MTP", bold=True); groups.append((x, 52, "1 MTP")); x += 52 + GAP
blk(x, 40, HL, "#FFFFFF", NEU, "L", bold=True); groups.append((x, 40, "loss")); x += 40
for gx, gw, lab in groups:
    s.text(gx + gw / 2, BASE + 26, lab, 17, INK)
s.line(108, BASE + 1, x + 4, BASE + 1, INK, 1.5)
s.text(112, 172, "61 decoder layers ÷ 32 virtual stages = 1.9 per stage", 19, PRB, anchor="start", weight="bold")
s.text(112, 200, "61 is prime: no uniform split exists", 17, PRB, anchor="start")
s.save(os.path.join(HERE, "fig_model.svg"))
print("strip ends at", x)
