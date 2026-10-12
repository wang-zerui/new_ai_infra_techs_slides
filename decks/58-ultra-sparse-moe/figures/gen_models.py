"""fig_models: the shared FFN shape of 2026 open frontier MoEs (config table, numbers are model configs)."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 282
s = SVG(W, H, "fig_models: total/active, routed experts, routed per token, shared expert for V3, GLM-5, K2, V4-Flash, V4-Pro")

cols = [("Model", 10, "start"), ("Total / active", 330, "middle"), ("Routed experts", 520, "middle"),
        ("Routed / token", 683, "middle"), ("Shared", 812, "middle")]
rows = [
    ("DeepSeek-V3 (2024)", "671B / 37B", "256", "8", "1", True),
    ("GLM-5", "744B / 40B", "256", "8", "1", False),
    ("Kimi K2", "1.04T / 32.6B", "384", "8", "1", False),
    ("DeepSeek-V4-Flash", "284B / 13B", "256", "6", "1", False),
    ("DeepSeek-V4-Pro", "1.6T / 49B", "384", "6", "1", False),
]
HEAD_Y, ROW0, RH = 30, 44, 46
# accent band behind the three "shape" columns
s.rect(436, ROW0 + 2, 432, RH * len(rows) - 4, ACC_F, ACC, 1.5, 6)
for name, x, anc in cols:
    s.text(x, HEAD_Y, name, 19, ACC, anchor=anc, weight="bold")
s.line(4, ROW0, W - 4, ROW0, INK, 1.6)
for i, (m, ta, r, k, sh, ref) in enumerate(rows):
    y = ROW0 + i * RH + RH / 2 + 7
    col = MUTED if ref else INK
    s.text(10, y, m, 19, col, anchor="start", weight="normal" if ref else "bold")
    s.text(330, y, ta, 19, col)
    s.text(520, y, r, 20, col, weight="bold")
    s.text(683, y, k, 20, col, weight="bold")
    s.text(812, y, sh, 20, col, weight="bold")
    if i < len(rows) - 1:
        s.line(4, ROW0 + (i + 1) * RH, 434, ROW0 + (i + 1) * RH, "#BFBFBF", 1)
        s.line(438, ROW0 + (i + 1) * RH, W - 4, ROW0 + (i + 1) * RH, "#9DC3E6", 1)
s.save(os.path.join(HERE, "fig_models.svg"))
