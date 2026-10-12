"""fig_statusquo: (a) long window on every token, mostly packed short docs; (b) short-only training plus a RoPE
rescale, never trained on long recall. Schematic, no numbers beyond the window sizes."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 300, "fig_statusquo: two failed suitors for long context")

# divider
s.line(435, 14, 435, 290, "#BFBFBF", 1.2)

# ---------------- panel (a)
s.text(16, 30, "(a) Long window from the start", 20, INK, anchor="start", weight="bold")
s.rect(20, 52, 395, 44, PRB_F, PRB, 2, 4)
s.text(170, 80.3, "whole run at 128K", 18, INK)
s.rect(330, 52, 40, 44, "url(#hatchPrb)", PRB, 2, 0)
s.line(330, 96, 22, 150, MUTED, 1.4, dash="5,4")
s.line(370, 96, 413, 150, MUTED, 1.4, dash="5,4")
s.rect(20, 150, 395, 84, "#FFFFFF", NEU, 1.8, 5)
x = 30
widths = [22, 30, 18, 26, 20, 96, 24, 18, 30, 22, 26]
for w in widths:
    if w == 96:
        s.box(x, 172, w, 44, "long doc", "#FFFFFF", NEU, size=16)
    else:
        s.rect(x, 172, w, 44, NEU_F, NEU, 1.6, 2)
    x += w + 4
s.text(217, 258, "one 128K training sequence", 18, MUTED)
s.text(217, 284, "mostly packed short docs", 18, PRB, weight="bold")

# ---------------- panel (b)
s.text(451, 30, "(b) Short only + RoPE stretch", 20, INK, anchor="start", weight="bold")
s.box(455, 52, 268, 44, "whole run at 4K", NEU_F, NEU, color=INK)
s.line(725, 74, 744, 74, INK, 2.2, "arr")
s.box(747, 52, 113, 44, ["RoPE", "rescale"], "#FFFFFF", NEU, size=17, dash="6,4")
s.line(803, 96, 803, 148, INK, 2.2, "arr")
s.rect(455, 150, 405, 84, "#FFFFFF", NEU, 1.8, 5)
s.box(466, 186, 52, 36, "fact", "#FFFFFF", INK, size=16, weight="bold")
x = 526
for w in [26, 20, 30, 18, 24, 28, 22, 18, 26]:
    s.rect(x, 186, w, 36, NEU_F, NEU, 1.6, 2)
    x += w + 4
s.box(796, 186, 54, 36, "query", "#FFFFFF", INK, size=16, weight="bold")
s.path("M 823,186 C 823,156 492,156 492,184", PRB, 2.2, "arrPrb", dash="7,5")
s.text(657, 170, "✗", 24, PRB, weight="bold", halo=True)
s.text(657, 258, "128K prompt at inference", 18, MUTED)
s.text(657, 284, "never trained on long recall", 18, PRB, weight="bold")
s.save(os.path.join(HERE, "fig_statusquo.svg"))
