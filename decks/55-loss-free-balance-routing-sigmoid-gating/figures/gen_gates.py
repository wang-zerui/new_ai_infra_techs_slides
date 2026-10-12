"""fig_gates: why sigmoid suits an additive routing bias (arXiv 2408.15664 App. C.2): raising one expert's logit
changes every softmax score (coupled), but only its own sigmoid score. Illustrative logits (2.0, 1.5, 0.5) -> E3 to 2.5."""
import math
import os
from svgkit import SVG, ACC, ACC_F, PRB, NEU, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 276
s = SVG(W, H, "fig_gates: softmax couples expert scores, sigmoid keeps them independent (illustrative)")

before, after = [2.0, 1.5, 0.5], [2.0, 1.5, 2.5]


def softmax(l):
    e = [math.exp(v) for v in l]
    return [v / sum(e) for v in e]


def sigmoid(l):
    return [1 / (1 + math.exp(-v)) for v in l]


BASE, SCALE, BW = 206, 168, 42
panels = [(0, softmax, "(a) Softmax: E1 and E2 shrink", PRB), (450, sigmoid, "(b) Sigmoid: E1 and E2 unchanged", ACC)]
for x0, fn, label, col in panels:
    vb, va = fn(before), fn(after)
    s.line(x0 + 20, BASE, x0 + 410, BASE, INK, 1.6)
    for i in range(3):
        cx = x0 + 85 + i * 125
        hb, ha = vb[i] * SCALE, va[i] * SCALE
        s.rect(cx - BW - 3, BASE - hb, BW, hb, "#FFFFFF", NEU, 1.8, 2, dash="5,3")
        changed = abs(va[i] - vb[i]) > 1e-6 and i < 2
        fill = ACC_F if fn is sigmoid else "#D9D9D9"
        stroke = PRB if changed else (ACC if fn is sigmoid else NEU)
        s.rect(cx + 3, BASE - ha, BW, ha, fill, stroke, 2.6 if changed else 1.8, 2)
        s.text(cx, BASE + 24, f"E{i + 1}", 18, INK)
    s.text(x0 + 215, 266, label, 19, col, weight="bold")
s.line(436, 30, 436, 276, "#BFBFBF", 1.2)
# legend
s.rect(258, 6, 26, 20, "#FFFFFF", NEU, 1.8, 2, dash="5,3")
s.text(292, 22, "before", 17, INK, anchor="start")
s.rect(372, 6, 26, 20, "#D9D9D9", NEU, 1.8, 2)
s.text(406, 22, "after raising the logit of E3", 17, INK, anchor="start")
s.save(os.path.join(HERE, "fig_gates.svg"))
print("softmax", [round(v, 3) for v in softmax(before)], [round(v, 3) for v in softmax(after)])
print("sigmoid", [round(v, 3) for v in sigmoid(before)], [round(v, 3) for v in sigmoid(after)])
