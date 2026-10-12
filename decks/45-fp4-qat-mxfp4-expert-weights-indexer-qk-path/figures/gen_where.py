"""Hook: the two FP4 islands in an otherwise FP8 V4 model (precision per [cfg] and [V4] §5.2.1)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svgkit import SVG, ACC, ACC_FILL, GRAY, GRAY_FILL, INK

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 282
s = SVG(W, H)


def tag(x, y, t, hot):
    w = 12 + 10.5 * len(t)
    s.rect(x - w, y - 12, w, 24, ACC if hot else "#fff", ACC if hot else GRAY, 2, 12)
    s.text(x - w / 2, y + 6, t, 16, "bold", "#fff" if hot else GRAY)


def box(x, y, w, h, label, hot=False, t=None):
    s.rect(x, y, w, h, ACC_FILL if hot else GRAY_FILL, ACC if hot else GRAY, 2.5 if hot else 2, 6)
    s.text(x + w / 2, y + h / 2 + 6, label, 18, "bold", INK)
    if t:
        tag(x + w - 6, y, t, t == "FP4")


# CSA layer
s.rect(0, 0, 420, 282, "#fff", GRAY, 1.5, 6, "7,5")
s.text(14, 26, "CSA attention layer", 18, "bold", GRAY, "start")
box(18, 58, 176, 50, "Compressors", t="FP8")
box(226, 58, 176, 50, "Indexer Q · K", True, "FP4")
box(226, 140, 176, 50, "Index scores", True, "BF16")
box(18, 140, 176, 50, "Top-k", False)
box(18, 218, 384, 50, "Shared-KV MQA", t="FP8")
s.arrow(196, 83, 224, 83)
s.arrow(314, 110, 314, 138)
s.arrow(224, 165, 196, 165)
s.arrow(106, 192, 106, 216)

# MoE layer
s.rect(450, 0, 420, 282, "#fff", GRAY, 1.5, 6, "7,5")
s.text(464, 26, "MoE layer", 18, "bold", GRAY, "start")
box(468, 58, 176, 50, "Router")
box(676, 58, 176, 50, "Shared expert", t="FP8")
s.rect(468, 140, 384, 128, ACC_FILL, ACC, 2.5, 6)
s.text(660, 166, "Routed experts × 384", 18, "bold", INK)
tag(846, 140, "FP4", True)
for r in range(2):
    for c in range(14):
        s.rect(484 + c * 25.5, 184 + r * 36, 20, 28, "#fff", ACC, 1.5, 2)
s.arrow(556, 110, 556, 138)
s.save(os.path.join(HERE, "fig_where.svg"))
