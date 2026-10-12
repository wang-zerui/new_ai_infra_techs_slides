"""Overview: CSA data flow for one query token (left-to-right lanes). Shapes use V4 notation:
n = context length, m = 4, c = 512 (KV entry), c_I = 128 (indexer key)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svgkit import SVG, ACC, ACC_FILL, GRAY, GRAY_FILL, INK

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 314
s = SVG(W, H)

C0, C1, C2, C3, C4 = (0, 150), (186, 160), (392, 166), (598, 150), (782, 86)
A, B, C, D = (14, 60), (108, 60), (194, 52), (256, 56)


def at(col, lane):
    return col[0], lane[0], col[1], lane[1]


def two(col, lane, l1, l2, fill="#fff", stroke=GRAY, w1="bold"):
    x, y, w, h = at(col, lane)
    s.rect(x, y, w, h, fill, stroke, 2.5 if stroke == ACC else 2, 6)
    s.text(x + w / 2, y + h / 2 - 4, l1, 18, w1, INK)
    s.text(x + w / 2, y + h / 2 + 18, l2, 18, "normal", INK)


# inputs (pre-existing, neutral)
two(C0, A, "Context", "n tokens")
x, y, w, h = at(C0, C); s.box(x, y, w, h, "Query token")
two(C0, D, "Last 128", "tokens")

# CSA parts (accent)
two(C1, A, "Compressors", "m = 4", ACC_FILL, ACC)
two(C1, B, "Indexer keys", "n/4 × 128", ACC_FILL, ACC)
two(C2, A, "KV entries", "n/4 × 512", ACC_FILL, ACC)
two(C2, B, "Lightning", "indexer + top-k", ACC_FILL, ACC)
two(C3, A, "Selected", "k entries", ACC_FILL, ACC)
two(C3, D, "Window", "128 raw entries", ACC_FILL, ACC)
x, y, w, h = C4[0], A[0], C4[1], D[0] + D[1] - A[0]
s.rect(x, y, w, h, ACC_FILL, ACC, 2.5, 6)
s.lines(x + w / 2, y + h / 2, ["Shared", "KV", "MQA"], 20, "bold", INK)
x, y, w, h = at(C1, C); s.box(x, y, w, h, "Query latent", "#fff", GRAY, 2, 18, "bold")

# arrows: lane A
s.arrow(152, 44, 184, 44)
s.arrow(348, 44, 390, 44)
s.arrow(560, 44, 596, 44)
s.arrow(750, 44, 780, 44)
# compressors -> indexer keys -> indexer
s.arrow(266, 76, 266, 106)
s.arrow(348, 138, 390, 138)
# indexer -> selected (top-k ids)
s.path("M560,138 H673 V78")
s.text(680, 116, "top-k ids", 16, "normal", ACC, "start")
# query -> latent -> indexer queries + attention queries
s.arrow(152, 220, 184, 220)
s.path("M348,220 H778")
s.path("M475,220 V172")
s.text(486, 196, "indexer q", 16, "normal", INK, "start")
s.text(600, 212, "attention q", 16, "normal", INK, "start")
# window branch
s.arrow(152, 284, 596, 284)
s.arrow(750, 284, 780, 284)

# badges
s.badge(186, 14, 1, ACC)
s.badge(392, 108, 2, ACC)
s.badge(782, 14, 3, ACC)
s.save(os.path.join(HERE, "fig_overview.svg"))
