"""Slide 9 (mechanism 3): multi-layer distillation — the F layer's indexer is trained toward the average attention
distribution of all layers that will reuse its top-k (IndexCache Eq. 1-3). Bar heights are illustrative."""
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 280
s = SVG(W, H, "multi-layer distillation")
random.seed(5)
base = [0.2, 0.9, 0.3, 0.15, 0.7, 0.25, 0.5, 0.1, 0.8, 0.2, 0.35, 0.6]


def spark(x, y, w, h, vals, color, stroke):
    n = len(vals); bw = w / n
    s.rect(x, y, w, h, "#FFFFFF", stroke, 1.6, 4)
    for i, v in enumerate(vals):
        bh = v * (h - 10)
        s.rect(x + i * bw + 2, y + h - 4 - bh, bw - 4, bh, color, color, 0.5, 0)


layers = ["ℓ+3", "ℓ+2", "ℓ+1", "ℓ"]
dists = []
y0, rh, gap = 6, 52, 14
for i, lab in enumerate(layers):
    vals = [max(0.05, min(1.0, b + random.uniform(-0.18, 0.18))) for b in base]
    dists.append(vals)
    y = y0 + i * (rh + gap)
    s.text(0, y + 22, f"layer {lab}", 17, INK, anchor="start")
    s.rich(0, y + 44, [("attention p", ""), (lab, "sup")], 16, anchor="start", color=MUTED)
    spark(120, y, 190, rh, vals, "#A6A6A6", NEU)
    s.arrow(312, y + rh / 2, 349, 120 + i * 12)
avg = [sum(d[i] for d in dists) / 4 for i in range(len(base))]
# averaging node
s.rect(352, 110, 58, 56, "#FFFFFF", INK, 2, 5)
s.text(381, 144, "mean", 17, INK, weight="bold")
s.arrow(412, 138, 427, 138)
spark(430, 112, 170, 52, avg, "#7F7F7F", NEU)
s.rich(515, 100, [("target p̄", "b"), (" = mean of 4", "")], 17)
# KL node and indexer
s.rect(650, 110, 190, 56, "#FFFFFF", ACC, 2.5, 6)
s.rich(745, 145, [("KL( p̄ ‖ q", ""), ("ℓ", "sup"), (" )", "")], 19)
s.arrow(602, 138, 647, 138)
s.rect(650, 206, 190, 66, ACC_F, ACC, 2.5, 6)
s.text(745, 228, "Indexer of layer ℓ", 17, INK, weight="bold")
spark(680, 236, 130, 30, [max(0.05, a + random.uniform(-0.08, 0.08)) for a in avg], ACC, ACC)
s.arrow(718, 204, 718, 169)
s.path("M772,169 V203", ACC, 2.2, "arrAcc", dash="6,4")
s.text(780, 192, "grad", 16, ACC, anchor="start", weight="bold")
s.text(745, 92, "same gradient as the 4 KL terms", 16, MUTED, style="italic")
s.save(os.path.join(HERE, "fig_distill.svg"))
