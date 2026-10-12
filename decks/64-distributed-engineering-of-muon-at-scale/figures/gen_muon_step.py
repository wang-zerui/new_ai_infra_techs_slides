"""fig_muon_step: DeepSeek-V4 Muon update for one logically independent matrix W (n x m), Algorithm 1 + §2.4:
Nesterov momentum (mu = 0.95), hybrid Newton-Schulz (8 + 2 iterations), RMS rescale (0.18), weight decay + update."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 268
s = SVG(W, H, "fig_muon_step")

Y, BH = 34, 70
boxes = [  # x, w, lines, highlight
    (2, 116, ["Gradient", "G (n × m)"], False),
    (146, 160, ["Nesterov", "momentum 0.95"], False),
    (334, 176, ["Newton-Schulz", "10 iterations"], True),
    (538, 150, ["Rescale", "RMS to 0.18"], False),
    (716, 152, ["Decay +", "update W"], False),
]
for i, (x, w, lines, hi) in enumerate(boxes):
    if hi:
        s.box(x, Y, w, BH, lines, ACC_F, ACC, color=ACC, weight="bold", sw=2.6)
    else:
        s.box(x, Y, w, BH, lines, NEU_F, NEU)
    if i < len(boxes) - 1:
        nx = boxes[i + 1][0]
        s.line(x + w + 2, Y + BH / 2, nx - 3, Y + BH / 2, INK, 2.2, "arr")
s.text(2, 20, "per logically independent matrix W", 18, MUTED, anchor="start", style="italic")

# zoom into Newton-Schulz
ZX, ZY, ZW, ZH = 120, 140, 630, 118
s.path(f"M334,{Y + BH} L{ZX},{ZY}", ACC, 1.4, dash="5,4")
s.path(f"M510,{Y + BH} L{ZX + ZW},{ZY}", ACC, 1.4, dash="5,4")
s.rect(ZX, ZY, ZW, ZH, "#FFFFFF", ACC, 1.8, 6, dash="6,4")
s.text(ZX + ZW / 2, ZY + 34, "X ← aX + b(XXᵀ)X + c(XXᵀ)²X", 23, INK, weight="bold")
s.text(ZX + ZW / 2, ZY + 64, "8 iterations (3.4445, −4.7750, 2.0315) then 2 iterations (2, −1.5, 0.5)", 17, INK)
s.text(ZX + ZW / 2, ZY + 98, "XXᵀ needs the whole matrix at once", 19, PRB, weight="bold")
s.save(os.path.join(HERE, "fig_muon_step.svg"))
