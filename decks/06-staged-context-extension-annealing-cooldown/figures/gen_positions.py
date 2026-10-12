"""fig_positions: how each model makes positions work at the longer window (schematic).
(a) K2: RoPE, trained to 32K, YaRN-rescaled to 128K (K2 §2.5). (b) GLM-4.5: RoPE base 10,000 -> 1,000,000 when
the window moves to 32K (GLM §2.4). (c) K3: KDA layers carry order through their decay, MLA layers use NoPE, so
nothing changes from 8K to 1M (K3 §2.1.2, §3.4). Wave periods are illustrative only."""
import math, os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 282, "fig_positions: YaRN / RoPE base / NoPE+KDA")
for x in (290, 580):
    s.line(x, 14, x, 272, "#BFBFBF", 1.2)

def title(x, t, sub):
    s.text(x, 26, t, 20, INK, anchor="start", weight="bold")
    s.text(x, 52, sub, 18, ACC, anchor="start", weight="bold")

# (a) Kimi K2
title(10, "(a) Kimi K2", "RoPE + YaRN")
x0, x1 = 16, 274
xm = x0 + (x1 - x0) * 32 / 128
s.rect(x0, 120, xm - x0, 40, ACC_F, ACC, 2, 3)
s.rect(xm, 120, x1 - xm, 40, "#FFFFFF", ACC, 2, 3, dash="6,4")
s.text(x0, 108, "trained", 18, ACC, anchor="start", weight="bold")
s.text((xm + x1) / 2, 145.3, "YaRN-rescaled", 18, ACC)
for x, t in ((x0, "0"), (xm, "32K"), (x1, "128K")):
    s.line(x, 160, x, 168, INK, 1.6)
s.text(x0, 188, "0", 17, INK, anchor="start")
s.text(xm, 188, "32K", 17, INK)
s.text(x1, 188, "128K", 17, INK, anchor="end")
s.text(145, 222, "window position", 16, MUTED, style="italic")
s.text(145, 264, "128K from a 32K stage", 18, INK)

# (b) GLM-4.5
title(300, "(b) GLM-4.5", "RoPE base 10K → 1M")
def wave(xa, xb, cy, amp, cycles, color, sw):
    pts = []
    for i in range(121):
        t = i / 120
        pts.append(f"{xa + (xb - xa) * t:.1f},{cy - amp * math.sin(2 * math.pi * cycles * t):.1f}")
    s.path("M " + " L ".join(pts), color, sw)
s.text(306, 92, "base 10K", 18, MUTED, anchor="start")
wave(306, 570, 122, 18, 5, "#A6A6A6", 2.4)
s.text(306, 162, "base 1M", 18, ACC, anchor="start", weight="bold")
wave(306, 570, 192, 18, 0.7, ACC, 3)
s.line(306, 214, 568, 214, INK, 1.6, "arr")
s.text(438, 235, "position within 32K", 16, MUTED, style="italic")
s.text(438, 264, "slower rotation per position", 18, INK)

# (c) Kimi K3
title(590, "(c) Kimi K3", "NoPE + KDA")
bx = 596; w = 56; g = 6
for i in range(3):
    s.box(bx + i * (w + g), 104, w, 46, "KDA", ACC_F, ACC, color=ACC, weight="bold")
mx = bx + 3 * (w + g)
s.box(mx, 104, 864 - mx, 46, "MLA", "#FFFFFF", NEU, weight="bold")
s.path(f"M {bx},158 L {bx},166 L {bx + 3 * w + 2 * g},166 L {bx + 3 * w + 2 * g},158", ACC, 1.8)
s.text(bx + (3 * w + 2 * g) / 2, 188, "decay orders tokens", 17, ACC)
s.text((mx + 864) / 2, 188, "no RoPE", 17, INK)
s.text(729, 222, "same layers at every window", 16, MUTED, style="italic")
s.text(729, 264, "8K to 1M with no change", 18, INK)
s.save(os.path.join(HERE, "fig_positions.svg"))
