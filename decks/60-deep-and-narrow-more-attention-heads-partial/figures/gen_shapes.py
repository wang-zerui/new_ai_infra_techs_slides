"""fig_shapes: silhouettes of DeepSeek-V3, Kimi K2, GLM-4.5 — height = layers, width = hidden dim (to scale).

Config numbers from GLM-4.5 Table 1 (arXiv 2508.06471): layers 61/61/92 (dense 3/1/3), hidden 7168/7168/5120,
routed experts 256/384/160, total params 671B/1043B/355B.
"""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 480, 360
s = SVG(W, H, "fig_shapes: model silhouettes, height = layers, width = hidden dim")

PITCH = 2.55          # pt per layer
WPX = 116 / 7168      # pt per hidden unit
BASE = 268            # bottom of every stack

models = [
    # name, layers, dense, hidden, experts, total, ours
    ("DeepSeek-V3", 61, 3, 7168, 256, "671B", False),
    ("Kimi K2", 61, 1, 7168, 384, "1043B", False),
    ("GLM-4.5", 92, 3, 5120, 160, "355B", True),
]
cxs = [86, 240, 394]

for (name, L, dense, hid, exp, tot, ours), cx in zip(models, cxs):
    w = hid * WPX
    h = L * PITCH
    x, y = cx - w / 2, BASE - h
    fill, stroke = (ACC_F, ACC) if ours else (NEU_F, NEU)
    s.rect(x, y, w, h, fill, stroke, 2.2 if ours else 1.8, rx=2)
    # one thin line per layer boundary (texture = layer count)
    for i in range(1, L):
        yy = BASE - i * PITCH
        s.line(x + 1.5, yy, x + w - 1.5, yy, "#A9C4E0" if ours else "#C8C8C8", 0.8)
    # dense layers at the bottom, darker band
    s.rect(x, BASE - dense * PITCH, w, dense * PITCH, ACC if ours else NEU, None, 0, rx=0)
    col = ACC if ours else INK
    s.text(cx, y - 10, f"{tot} total", 18, col, weight="bold" if ours else "normal")
    s.text(cx, BASE + 26, name, 20, col, weight="bold")
    s.text(cx, BASE + 50, f"{L} layers × {hid}", 18, INK)
    s.text(cx, BASE + 73, f"{exp} experts", 18, INK)

# axis key (left edge): what height and width mean
for i, t in enumerate(["height = layers", "width = hidden dim", "dark band = dense layers"]):
    s.text(6, 20 + i * 19, t, 15, MUTED, anchor="start", style="italic")
s.save(os.path.join(HERE, "fig_shapes.svg"))
