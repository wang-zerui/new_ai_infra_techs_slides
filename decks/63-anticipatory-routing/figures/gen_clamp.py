"""fig_clamp: SwiGLU clamping in DeepSeek-V4 (§4.2.3): the linear (up) component is clamped to [-10, 10] and the
gate component is capped at 10, in every expert, throughout V4-Flash and V4-Pro training."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED, opnode

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 480, 312
s = SVG(W, H, "fig_clamp")

s.box(190, 4, 100, 34, "token x", "#FFFFFF", NEU, INK, 18, sw=1.8)
L, R, CW = 24, 266, 190
s.path(f"M240,40 L240,48 L{L + CW / 2},48 L{L + CW / 2},56", INK, 2, "arr")
s.path(f"M240,48 L{R + CW / 2},48 L{R + CW / 2},56", INK, 2, "arr")
s.box(L, 58, CW, 38, "gate projection", NEU_F, NEU, INK, 17, sw=1.6)
s.box(R, 58, CW, 38, "up projection", NEU_F, NEU, INK, 17, sw=1.6)
s.box(L, 114, CW, 38, "cap at 10", ACC_F, ACC, ACC, 18, weight="bold", sw=2.2)
s.box(R, 114, CW, 38, "clamp to [−10, 10]", ACC_F, ACC, ACC, 18, weight="bold", sw=2.2)
s.box(L, 170, CW, 38, "SiLU", NEU_F, NEU, INK, 17, sw=1.6)
for x in (L + CW / 2, R + CW / 2):
    s.line(x, 98, x, 112, INK, 2, "arr")
s.line(L + CW / 2, 154, L + CW / 2, 168, INK, 2, "arr")
# multiply
cx, cy = 240, 236
s.path(f"M{L + CW / 2},210 L{L + CW / 2},{cy} L{cx - 15},{cy}", INK, 2, "arr")
s.path(f"M{R + CW / 2},154 L{R + CW / 2},{cy} L{cx + 15},{cy}", INK, 2, "arr")
opnode(s, cx, cy, "×")
s.line(cx, cy + 14, cx, 266, INK, 2, "arr")
s.box(150, 268, 180, 38, "down projection", NEU_F, NEU, INK, 17, sw=1.6)
s.save(os.path.join(HERE, "fig_clamp.svg"))
