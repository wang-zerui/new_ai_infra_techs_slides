"""fig_module: worked example of MTP depth 1 in training on "The cat sat on the mat" (DeepSeek-V3 Eq. 21-24):
at position i the MTP block gets h_i and the embedding of the true next token t_{i+1}, and is trained on t_{i+2}."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 262
s = SVG(W, H, "fig_module: per-position inputs and targets of main model and MTP depth 1")

words = ["The", "cat", "sat", "on", "the", "mat"]
LX, X0, CW, RH, GAP, Y0 = 200, 216, 150, 44, 8, 34
rows = [("token tᵢ", lambda i: words[i], "plain"),
        ("main target tᵢ₊₁", lambda i: words[i + 1], "plain"),
        ("MTP input", lambda i: f"h({words[i]}) + {words[i + 1]}", "acc"),
        ("MTP target tᵢ₊₂", lambda i: words[i + 2], "accb")]
for c in range(4):
    s.text(X0 + c * CW + CW / 2, 22, f"i = {c + 1}", 16, MUTED, style="italic")
for r, (lab, fn, kind) in enumerate(rows):
    y = Y0 + r * (RH + GAP)
    s.text(LX, y + RH / 2 + 6.5, lab, 19, ACC if kind != "plain" else INK, anchor="end", weight="bold")
    for c in range(4):
        x = X0 + c * CW
        if kind == "plain":
            s.box(x + 4, y, CW - 8, RH, fn(c), "#FFFFFF", "#7F7F7F", size=19, sw=1.6)
        elif kind == "acc":
            s.box(x + 4, y, CW - 8, RH, fn(c), ACC_F, ACC, color=ACC, size=18, sw=1.6)
        else:
            s.box(x + 4, y, CW - 8, RH, fn(c), "#FFFFFF", ACC, color=ACC, size=19, weight="bold", sw=1.6)
# highlight position i = 2 ("cat")
hx = X0 + CW
s.rect(hx - 1, Y0 - 6, CW + 2, 4 * (RH + GAP) + 4, "none", ACC, 3, 6)
s.save(os.path.join(HERE, "fig_module.svg"))
