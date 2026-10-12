"""fig_bytes: values one token sends through dispatch, drawn to scale: one chip per selected expert, chip width
proportional to the routed width. K2: 8 x 7168; K3 if full width: 16 x 7168; K3 latent: 16 x 3584
(K3 Table 1; volume proportional to K x width, LatentMoE §2.2). Analytic, no measurements."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(490, 300, "fig_bytes: per-token dispatch volume to scale")

X0, FULL = 10, 400          # FULL px = 16 x 7168 values
px_per = FULL / (16 * 7168)
rows = [
    ("Kimi K2: 8 × 7168", 8, 7168, NEU_F, NEU, INK, "1×"),
    ("K3, full width: 16 × 7168", 16, 7168, PRB_F, PRB, PRB, "2×"),
    ("K3, latent: 16 × 3584", 16, 3584, ACC_F, ACC, ACC, "1×"),
]
y = 30
for label, k, w, fill, stroke, tcol, ratio in rows:
    s.text(X0, y, label, 19, tcol, anchor="start", weight="bold")
    cw = w * px_per
    for i in range(k):
        s.rect(X0 + i * cw, y + 12, cw - 2.5, 40, fill, stroke, 1.6, 2)
    s.text(X0 + FULL + 70, y + 41, ratio, 26, tcol, anchor="end", weight="bold")
    y += 92
s.text(X0, 290, "one chip = one selected expert's copy of the token", 16, MUTED, anchor="start", style="italic")
s.save(os.path.join(HERE, "fig_bytes.svg"))
