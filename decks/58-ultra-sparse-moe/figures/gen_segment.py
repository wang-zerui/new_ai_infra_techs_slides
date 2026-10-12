"""fig_segment: fine-grained expert segmentation (DeepSeekMoE §3.1). Same total expert width, same active width
(2 x f = 8 x f/4); combinations C(16,2)=120 vs C(64,8)=4,426,165,368."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 268
s = SVG(W, H, "fig_segment: 16 experts top-2 vs 64 quarter-width experts top-8")
X0, L = 196, 660

def strip(y, h, n, lit, color, fill):
    w = L / n
    for i in range(n):
        if i not in lit:
            s.rect(X0 + i * w, y, w, h, "#FFFFFF", "#8C8C8C", 1.2, 0)
    for i in sorted(lit):
        s.rect(X0 + i * w, y, w, h, fill, color, 1.6, 0)
    s.rect(X0, y, L, h, "none", NEU, 1.8, 0)

# row 1: coarse
Y1, Y2, SH = 8, 150, 70
s.text(4, Y1 + 30, "16 experts", 20, INK, anchor="start", weight="bold")
s.text(4, Y1 + 56, "width f · top-2", 19, INK, anchor="start")
strip(Y1, SH, 16, {3, 11}, INK, "#7F7F7F")
s.text(X0, Y1 + SH + 28, "120 possible expert sets", 19, PRB, anchor="start", weight="bold")
# split arrow
s.add(f'<path d="M58,96 h18 v18 h12 l-21,22 l-21,-22 h12 z" fill="#FFFFFF" stroke="{INK}" stroke-width="1.8"/>')
s.text(98, 124, "split ×4", 19, INK, anchor="start", style="italic")
# row 2: fine-grained
s.text(4, Y2 + 30, "64 experts", 20, ACC, anchor="start", weight="bold")
s.text(4, Y2 + 56, "width f/4 · top-8", 19, ACC, anchor="start")
strip(Y2, SH, 64, {5, 13, 22, 30, 37, 44, 51, 60}, ACC, ACC)
s.text(X0, Y2 + SH + 28, "4,426,165,368 possible expert sets", 19, ACC, anchor="start", weight="bold")
s.text(X0 + L, Y2 + SH + 28, "same active width: 2f = 8 · f/4", 18, MUTED, anchor="end", style="italic")
s.save(os.path.join(HERE, "fig_segment.svg"))
