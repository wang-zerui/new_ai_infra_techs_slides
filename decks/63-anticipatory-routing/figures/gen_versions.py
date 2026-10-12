"""fig_versions: which parameter version each part of training step t uses (DeepSeek-V4 §4.2.3).
Standard: everything from theta(t). Anticipatory: routing indices from theta(t - dt), cached; features, loss,
gradients and the update from theta(t). The paper does not state the version used for gate values."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 300
s = SVG(W, H, "fig_versions")

CA, CB = 360, 620       # column x
CW = 230
s.text(CA + CW / 2, 22, "(a) Standard step t", 19, INK, weight="bold")
s.text(CB + CW / 2, 22, "(b) Anticipatory step t", 19, ACC, weight="bold")

rows = [
    ("which experts (top-6 indices)", "θ(t)", ("θ(t−Δt) · cached", "acc")),
    ("hidden states and expert outputs", "θ(t)", ("θ(t)", "neu")),
    ("loss and gradients", "θ(t)", ("θ(t)", "neu")),
    ("optimizer update", "θ(t) → θ(t+1)", ("θ(t) → θ(t+1)", "neu")),
]
for i, (lab, a, (b, kb)) in enumerate(rows):
    y = 40 + i * 58
    s.rect(4, y - 4, W - 8, 52, "#FFFFFF" if i % 2 else "#FAFAFA", "#FFFFFF", 0, 0)
    s.text(14, y + 28, lab, 18, INK, anchor="start", weight="bold" if i == 0 else "normal")
    s.box(CA, y + 4, CW, 38, a, NEU_F, NEU, INK, 18, sw=1.6)
    if kb == "acc":
        s.box(CB, y + 4, CW, 38, b, ACC_F, ACC, ACC, 18, weight="bold", sw=2.4)
    else:
        s.box(CB, y + 4, CW, 38, b, NEU_F, NEU, INK, 18, sw=1.6)
s.line(CB - 20, 34, CB - 20, 268, "#BFBFBF", 1.2)
s.text(14, 290, "Gate values (expert weights): version not stated in the paper", 15, MUTED, anchor="start",
       style="italic")
s.save(os.path.join(HERE, "fig_versions.svg"))
