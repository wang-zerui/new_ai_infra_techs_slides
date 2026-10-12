"""fig_prope: partial RoPE in a GLM-4.5 head (128 dims: first 64 rotated, last 64 unrotated) vs the MLA head of
DeepSeek-V3 / Kimi K2 (128 'nope' dims + 64 decoupled RoPE dims). 3 pt per dim, to scale.

Sources: HF config zai-org/GLM-4.5 (head_dim 128, partial_rotary_factor 0.5); transformers modeling_glm4_moe.py
apply_rotary_pos_emb (first rotary_dim dims rotated, rest passed through); DeepSeek-V3 Sec. 4.2 (d_h = 128, d_h^R = 64).
"""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 760, 232
s = SVG(W, H, "fig_prope: rotated vs unrotated dims per head")

X, D = 176, 3.0          # bar start, pt per dim
BH = 32
r0, r1, u1 = X, X + 64 * D, X + 128 * D

# column headers
s.text((r0 + r1) / 2, 20, "dims 0–63", 17, MUTED)
s.text((r1 + u1) / 2, 20, "dims 64–127", 17, MUTED)
s.text(0, 20, "GLM-4.5 head", 18, ACC, anchor="start", weight="bold")

for y, lab, pos in ((30, "q at position m", "m"), (70, "k at position n", "n")):
    s.text(0, y + BH / 2 + 6, lab, 18, INK, anchor="start")
    s.box(r0, y, r1 - r0, BH, f"rotated by {pos}·θ", ACC_F, ACC, ACC, size=18, weight="bold", rx=0, sw=2)
    s.box(r1, y, u1 - r1, BH, "unrotated", "#FFFFFF", ACC, INK, size=18, rx=0, sw=2)

# brackets: what each half contributes to q·k
by = 110
for (a, b), txt, col in (((r0, r1), "depends on n − m", ACC), ((r1, u1), "position-free", INK)):
    s.path(f"M{a + 3},{by} L{a + 3},{by + 7} L{b - 3},{by + 7} L{b - 3},{by}", col, 1.8)
    s.text((a + b) / 2, by + 29, txt, 18, col, weight="bold" if col == ACC else "normal")
s.text(0, by + 29, "q·k terms", 18, INK, anchor="start")

# divider
s.line(0, 160, W - 4, 160, "#BFBFBF", 1.2)

# MLA head of DeepSeek-V3 / Kimi K2: 128 nope + 64 decoupled RoPE dims
my = 186
s.text(0, my + BH / 2 + 6, "q, k", 18, INK, anchor="start")
m1, m2 = X + 128 * D, X + 192 * D
s.box(X, my, m1 - X, BH, "128 dims unrotated", "#FFFFFF", NEU, INK, size=18, rx=0, sw=2)
s.box(m1, my, m2 - m1, BH, "64 rotated", NEU_F, NEU, INK, size=18, rx=0, sw=2)
s.text(0, my - 10, "DeepSeek-V3 / Kimi K2 MLA head", 18, INK, anchor="start", weight="bold")

s.save(os.path.join(HERE, "fig_prope.svg"))
