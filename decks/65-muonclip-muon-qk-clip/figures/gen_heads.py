"""fig_heads: (a) naive QK-Clip: one gamma from the hottest head shrinks every head; (b) per-head: gamma_h = min(1, tau/S_max^h)
pulls only the heads above tau down to tau. Logit heights are illustrative (no data). Source: K2 §2.1, App. D."""
import os
from svgkit import rich, SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 298, "fig_heads")
s.line(435, 14, 435, 290, "#BFBFBF", 1.2)
vals = [40, 70, 160, 55, 30, 120]
TAU = 100
BASE = 236

def panel(x0, title, sub, new, colors, note, note_color):
    s.text(x0 + 4, 26, title, 20, INK, anchor="start", weight="bold")
    rich(s, x0 + 4, 58, sub, 22, ACC, anchor="start")
    bw, gap = 42, 22
    bx0 = x0 + 52
    for i, (v, nv, c) in enumerate(zip(vals, new, colors)):
        x = bx0 + i * (bw + gap)
        fill, stroke = c
        if abs(nv - v) > 1e-6:
            s.rect(x, BASE - v, bw, v, "none", MUTED, 1.6, 2, dash="5,4")
        s.rect(x, BASE - nv, bw, nv, fill, stroke, 2, 2)
        rich(s, x + bw / 2, BASE + 24, [("h", ""), (str(i + 1), "sub")], 22, INK)
    s.line(x0 + 30, BASE - TAU, x0 + 425, BASE - TAU, ACC, 2, dash="8,5")
    s.text(x0 + 24, BASE - TAU + 7, "τ", 22, ACC, anchor="end", weight="bold")
    s.line(x0 + 34, BASE, x0 + 425, BASE, INK, 1.8)
    s.text(x0 + 230, 288, note, 19, note_color, weight="bold")

g = TAU / max(vals)
naive = [v * g for v in vals]
cn = [(NEU_F, NEU) if v == max(vals) else (PRB_F, PRB) for v in vals]
panel(0, "(a) Naive: one γ for all heads", [("γ = τ / max", ""), ("h", "sub"), (" S", ""), ("max", "sub")],
      naive, cn, "cold heads shrink too", PRB)
per = [min(v, TAU) for v in vals]
cp = [(ACC_F, ACC) if v > TAU else (NEU_F, NEU) for v in vals]
panel(440, "(b) QK-Clip: one γ per head", [("γ", ""), ("h", "sub"), (" = min(1, τ / S", ""), ("max", "sub"), ("h", "sup"), (")", "")],
      per, cp, "only hot heads shrink", ACC)
s.rect(722, 44, 26, 18, "none", MUTED, 1.6, 2, dash="5,4")
s.text(756, 59, "before clip", 17, MUTED, anchor="start")
s.save(os.path.join(HERE, "fig_heads.svg"))
