"""fig_recipe: routing hyper-parameters of three production models.
DeepSeek-V3: arXiv 2412.19437 §2.1.2, §4.2 (p.22-23). GLM-4.5: arXiv 2508.06471 §2.1 Table 1, §2.4.
DeepSeek-V4: arXiv 2606.19348 §2.1, §4.2.1-4.2.2 (Flash 256 routed, Pro 384 routed, both top-6)."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 282
s = SVG(W, H, "fig_recipe: production loss-free routing settings")

cols = ["DeepSeek-V3", "GLM-4.5", "DeepSeek-V4"]
rows = [
    ("Score function", ["Sigmoid", "Sigmoid", "√Softplus"], False),
    ("Routed experts", ["256 · top-8", "160 · top-8", "256–384 · top-6"], False),
    ("Bias update rate", ["0.001", "0.001", "0.001"], True),
    ("Rate set to 0", ["last 500B tokens", "after 15T tokens", "not stated"], False),
    ("Sequence loss weight", ["1e-4", "1e-4", "1e-4"], True),
]
LW, CW, HH, RH = 228, 214, 42, 46
x0 = 2
for j, c in enumerate(cols):
    s.text(x0 + LW + j * CW + CW / 2, 28, c, 20, INK, weight="bold")
s.line(x0, HH, W - 2, HH, INK, 2)
for r, (lab, vals, hi) in enumerate(rows):
    y = HH + 4 + r * RH
    if hi:
        s.rect(x0, y, W - 4, RH - 4, ACC_F, ACC_F, 0, 3)
    s.text(x0 + 8, y + RH / 2 + 4, lab, 19, ACC if hi else INK, anchor="start", weight="bold")
    for j, v in enumerate(vals):
        col = MUTED if v == "not stated" else (ACC if hi else INK)
        s.text(x0 + LW + j * CW + CW / 2, y + RH / 2 + 4, v, 19, col, weight="bold" if hi else "normal",
               style="italic" if v == "not stated" else "normal")
    if r < len(rows) - 1:
        s.line(x0, y + RH - 2, W - 2, y + RH - 2, "#D9D9D9", 1.2)
s.line(x0, HH + 4 + len(rows) * RH - 2, W - 2, HH + 4 + len(rows) * RH - 2, INK, 2)
s.save(os.path.join(HERE, "fig_recipe.svg"))
