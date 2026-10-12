"""fig_suitors: (a) logit soft-cap caps the logit after QK^T, while W_q/W_k (and q.k) keep growing underneath;
(b) QK-Norm normalizes per-head Q and K, but MLA inference never materializes per-head K (only latent c_KV and shared
k_R are cached). Sources: K2 §2.1 p.3; DeepSeek-V3 §2.1.1."""
import os
from svgkit import rich, SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 270, "fig_suitors")
s.line(430, 14, 430, 262, "#BFBFBF", 1.2)

# (a) soft-cap
s.text(12, 30, "(a) Logit soft-cap", 20, INK, anchor="start", weight="bold")
Y = 74; H = 50
s.rect(16, Y, 108, H, PRB_F, PRB, 2, 4)
rich(s, 70, Y + 32, [("QK", ""), ("T", "sup")], 22, PRB, weight="bold")
s.line(126, Y + H / 2, 152, Y + H / 2, INK, 2.2, "arr")
s.box(154, Y, 118, H, "soft-cap", "#FFFFFF", NEU, size=20, weight="bold")
s.line(274, Y + H / 2, 300, Y + H / 2, INK, 2.2, "arr")
s.box(302, Y, 112, H, "softmax", "#FFFFFF", NEU, size=20)
s.rect(16, 190, 108, 50, PRB_F, PRB, 2, 4)
rich(s, 70, 222, [("W", ""), ("q", "sub"), (", W", ""), ("k", "sub")], 22, PRB, weight="bold")
s.line(70, 188, 70, Y + H + 3, PRB, 2.4, "arrPrb")
s.text(140, 210, "keep growing", 19, PRB, anchor="start", weight="bold")
s.text(140, 234, "beneath the cap", 19, PRB, anchor="start", weight="bold")
s.text(213, 152, "bounds only the output", 18, MUTED, style="italic")

# (b) QK-Norm
s.text(442, 30, "(b) QK-Norm", 20, INK, anchor="start", weight="bold")
s.box(446, 62, 60, 42, "Q", "#FFFFFF", NEU, size=20, weight="bold")
s.box(446, 124, 60, 42, "K", "#FFFFFF", PRB, size=20, weight="bold", dash="6,4")
s.line(508, 83, 534, 83, INK, 2.2, "arr")
s.line(508, 145, 534, 145, INK, 2.2, "arr")
s.box(536, 62, 92, 42, "norm", "#FFFFFF", NEU, size=20)
s.box(536, 124, 92, 42, "norm", "#FFFFFF", PRB, size=20, dash="6,4")
s.text(628, 134, "✗", 30, PRB, weight="bold", halo=True)
s.line(476, 188, 476, 170, MUTED, 1.8, "arrGray", dash="5,4")
s.path("M 630,83 L 660,83 L 660,104", INK, 2.2)
s.path("M 630,145 L 660,145 L 660,124", INK, 2.2)
s.line(660, 114, 690, 114, INK, 2.2, "arr")
s.rect(692, 92, 100, 44, "#FFFFFF", NEU, 2, 4)
rich(s, 742, 121, [("QK", ""), ("T", "sup")], 22, INK, weight="bold")
s.rect(446, 190, 346, 46, NEU_F, NEU, 2, 4)
rich(s, 619, 220, [("MLA cache: latent c", ""), ("KV", "sub"), (" + shared k", ""), ("R", "sub")], 20, INK)
s.text(619, 262, "no per-head K at inference", 19, PRB, weight="bold")
s.save(os.path.join(HERE, "fig_suitors.svg"))
