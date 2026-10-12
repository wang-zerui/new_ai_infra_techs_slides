"""fig_onoff: where the paper's best configurations enable 1F1B EP overlap (App. B.2 of arXiv 2603.07685)."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(480, 290, "fig_onoff: 1F1B overlap ON/OFF per model and platform")
cols = [("H100", "NVL8"), ("GB200", "NVL72"), ("GB300", "NVL72")]
rows = [("DeepSeek-V3", [True, False, True]), ("Qwen3-235B", [True, True, False])]
CX, CW, GAP = 150, 104, 6
RY, RH = 74, 62
for j, (g, nv) in enumerate(cols):
    cx = CX + j * (CW + GAP) + CW / 2
    s.text(cx, 30, g, 18, INK, weight="bold")
    s.text(cx, 54, nv, 16, MUTED)
for i, (m, on) in enumerate(rows):
    y = RY + i * (RH + 8)
    s.text(140, y + RH / 2 + 6.3, m, 18, INK, anchor="end", weight="bold")
    for j, v in enumerate(on):
        x = CX + j * (CW + GAP)
        if v:
            s.box(x, y, CW, RH, "ON", ACC_F, ACC, color=ACC, weight="bold", size=20)
        else:
            s.box(x, y, CW, RH, "OFF", "#FFFFFF", "#A6A6A6", color=MUTED, size=20, dash="6,4")
yb = RY + 2 * RH + 8 + 16
x1 = CX
s.path(f"M{x1 + 2},{yb} v8 h{CW - 4} v-8", MUTED, 1.6)
s.text(x1 + CW / 2, yb + 30, "EP across", 16, MUTED, style="italic")
s.text(x1 + CW / 2, yb + 50, "nodes", 16, MUTED, style="italic")
x2 = CX + CW + GAP
s.path(f"M{x2 + 2},{yb} v8 h{2 * CW + GAP - 4} v-8", MUTED, 1.6)
s.text(x2 + CW + GAP / 2, yb + 30, "EP inside NVLink domain", 16, MUTED, style="italic")
s.save(os.path.join(HERE, "fig_onoff.svg"))
