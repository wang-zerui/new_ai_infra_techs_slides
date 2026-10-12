"""fig_mla: why QK-Norm fits GQA but not MLA decode.

GQA (GLM-4.5): per-head keys are materialized, normalized, cached (transformers modeling_glm4_moe.py).
MLA (DeepSeek-V3 / Kimi K2): only the 512-dim latent c and the 64-dim shared RoPE key are cached (DeepSeek-V3 Sec. 2.1.1);
per-head keys k_h = W_UK,h c are "not fully materialized during inference" (Kimi K2 Sec. 2.1) because the up-projection
is folded into the query: q_h . (W_UK,h c) = (W_UK,h^T q_h) . c.
"""
import os
from svgkit import SVG, rich, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 276
s = SVG(W, H, "fig_mla: QK-Norm needs materialized keys")
BH = 56


def chain(y, items, x0=170, gap=30):
    xs, x = [], x0
    for w, parts, fill, st, col, wt in items:
        s.rect(x, y, w, BH, fill, st, 2.4 if fill == ACC_F else 2, rx=4)
        rich(s, x + w / 2, y + BH / 2 + 6.5, parts, 18, color=col, weight=wt)
        xs.append((x, w))
        x += w + gap
    for (a, wa), (b, _) in zip(xs, xs[1:]):
        s.line(a + wa + 2, y + BH / 2, b - 2, y + BH / 2, INK, 2.2, "arr")
    return xs


# (a) GQA
YA = 26
s.text(0, YA + 24, "GQA", 20, ACC, anchor="start", weight="bold")
s.text(0, YA + 48, "GLM-4.5", 17, INK, anchor="start")
a = chain(YA, [
    (54, [("x",)], NEU_F, NEU, INK, "normal"),
    (130, [("k = W",), ("k", "sub"), (" x",)], NEU_F, NEU, INK, "normal"),
    (150, [("RMSNorm(k)",)], ACC_F, ACC, ACC, "bold"),
    (110, [("cache k",)], "#FFFFFF", NEU, INK, "normal"),
    (130, [("score q·k",)], NEU_F, NEU, INK, "normal"),
])
# (b) MLA decode
YB = 196
s.text(0, YB + 24, "MLA decode", 20, PRB, anchor="start", weight="bold")
s.text(0, YB + 48, "DeepSeek-V3, K2", 17, INK, anchor="start")
b = chain(YB, [
    (54, [("x",)], NEU_F, NEU, INK, "normal"),
    (150, [("c = W",), ("DKV", "sub"), (" x",)], NEU_F, NEU, INK, "normal"),
    (150, [("cache c (576)",)], "#FFFFFF", NEU, INK, "normal"),
    (220, [("score (W",), ("UK", "sub"), ("ᵀ", ""), ("q)·c",)], NEU_F, NEU, INK, "normal"),
])
# ghost: per-head key never formed
gx, gw, gy, gh = b[2][0], b[3][0] + b[3][1] - b[2][0], 106, 70
s.rect(gx, gy, gw, gh, PRB_F, PRB, 2, rx=4, dash="6,4")
rich(s, gx + gw / 2, gy + 29, [("per-head key k",), ("h", "sub"), (" = W",), ("UK,h", "sub"), (" c is never formed",)], 18,
     color=PRB)
s.text(gx + gw / 2, gy + 55, "no k to normalize", 18, PRB, weight="bold")
s.save(os.path.join(HERE, "fig_mla.svg"))
