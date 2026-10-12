"""fig_pick: worked example (illustrative values) of biased Top-2 selection with unbiased gate weights.
Sigmoid gate as in DeepSeek-V3 Eq. 13-16: g = selected s normalised to sum 1."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, NEU, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 290
s = SVG(W, H, "fig_pick: s, b, s+b, top-2 pick, gate weights for one token over 4 experts (illustrative)")

score = [0.90, 0.85, 0.60, 0.20]
bias = [-0.30, 0.00, 0.10, 0.00]
sb = [a + b for a, b in zip(score, bias)]
pick = sorted(range(4), key=lambda i: -sb[i])[:2]                  # -> E2, E3
den = sum(score[i] for i in pick)
gate = {i: score[i] / den for i in pick}

LX, CX0, CW, RH, GAP, Y0 = 190, 206, 118, 40, 6, 52
cols = [CX0 + i * CW for i in range(4)]
for i, x in enumerate(cols):
    name = "E1 · hot" if i == 0 else f"E{i + 1}"
    s.text(x + CW / 2, 36, name, 19, PRB if i == 0 else INK, weight="bold")
rows = ["score s", "bias b", "s + b", "top-2 pick", "gate weight"]
for r, lab in enumerate(rows):
    y = Y0 + r * (RH + GAP)
    s.text(LX, y + RH / 2 + 6.5, lab, 19, ACC if r in (1, 3) else INK, anchor="end", weight="bold")
    for i, x in enumerate(cols):
        fill, stroke, txt, col, wt = "#FFFFFF", "#7F7F7F", "", INK, "normal"
        if r == 0:
            txt = f"{score[i]:.2f}"
        elif r == 1:
            fill, stroke, col = ACC_F, ACC, ACC
            txt = (f"{bias[i]:+.2f}".replace("-", "−")) if bias[i] else "0"
        elif r == 2:
            txt = f"{sb[i]:.2f}"
            if i in pick:
                col, wt = ACC, "bold"
        elif r == 3:
            if i in pick:
                fill, stroke, txt, col, wt = ACC_F, ACC, "selected", ACC, "bold"
            else:
                txt, col = "–", MUTED
        else:
            if i in pick:
                txt, col, wt = f"{gate[i]:.2f}", ACC, "bold"
            else:
                txt, col = "–", MUTED
        s.box(x + 3, y, CW - 6, RH, txt, fill, stroke, color=col, size=19, weight=wt, sw=1.6)
# gate weights come from s (row 0), not from s + b
yr0 = Y0 + RH / 2
yr4 = Y0 + 4 * (RH + GAP) + RH / 2
xr = cols[-1] + CW
s.path(f"M {xr + 2} {yr0} L {xr + 30} {yr0} L {xr + 30} {yr4} L {xr + 4} {yr4}", ACC, 2.5, "arrAcc")
s.text(xr + 42, 152, "weights", 18, ACC, anchor="start", weight="bold")
s.text(xr + 42, 174, "use s only", 18, ACC, anchor="start", weight="bold")
s.save(os.path.join(HERE, "fig_pick.svg"))
print("pick", pick, "gate", gate)
