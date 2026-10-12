"""fig_platforms: GLM-5 and the seven domestic accelerator platforms it reports adaptation for."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(480, 304, "fig_platforms: GLM-5 on seven Chinese chip platforms, Ascend detailed in the report")

# model box
s.box(40, 8, 400, 50, ["GLM-5 · 744B MoE (40B active)"], ACC_F, ACC, color=ACC, weight="bold", size=19)

# fan-out arrow
s.line(240, 60, 240, 86, INK, 2.2, "arr")
s.text(252, 80, "full-stack adaptation", 17, MUTED, anchor="start", style="italic")

PLAT = ["Huawei Ascend", "Moore Threads", "Hygon", "Cambricon", "Kunlunxin", "MetaX", "Enflame"]
X, Y, W, H, GX, GY = 12, 96, 222, 42, 12, 10
for i, p in enumerate(PLAT):
    col, row = i % 2, i // 2
    x, y = X + col * (W + GX), Y + row * (H + GY)
    if i == 0:
        s.box(x, y, W, H, p, ACC_F, ACC, color=ACC, weight="bold", size=18)
    else:
        s.box(x, y, W, H, p, NEU_F, NEU, size=18)

# note on the case study, in the free cell beside Enflame
nx, ny = X + (W + GX), Y + 3 * (H + GY)
s.text(nx + 6, ny + 16, "Report details only", 17, ACC, anchor="start", style="italic")
s.text(nx + 6, ny + 38, "Ascend Atlas (§5)", 17, ACC, anchor="start", style="italic")
s.save(os.path.join(HERE, "fig_platforms.svg"))
