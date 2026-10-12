"""Overview: one HCA layer (left) and where it sits in the V4-Pro stack (right)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svgkit import SVG, ACC, ACC_FILL, GRAY, GRAY_FILL, INK

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 300
s = SVG(W, H)

# container for one HCA layer
s.rect(0, 0, 664, 300, "#fff", GRAY, 1.5, 6, "7,5")
s.text(14, 26, "One HCA layer", 18, "bold", GRAY, "start")

C0, C1, C2, C3 = (16, 140), (190, 160), (392, 160), (576, 76)
A, B, C = (46, 60), (142, 54), (224, 60)


def two(col, lane, l1, l2, fill="#fff", stroke=GRAY):
    x, w = col; y, h = lane
    s.rect(x, y, w, h, fill, stroke, 2.5 if stroke == ACC else 2, 6)
    if l2:
        s.text(x + w / 2, y + h / 2 - 4, l1, 18, "bold", INK)
        s.text(x + w / 2, y + h / 2 + 18, l2, 18, "normal", INK)
    else:
        s.text(x + w / 2, y + h / 2 + 6, l1, 18, "bold", INK)


two(C0, A, "Context", "n tokens")
two(C1, A, "Compressor", "m′ = 128", ACC_FILL, ACC)
two(C2, A, "KV entries", "n/128 × 512", ACC_FILL, ACC)
two(C0, B, "Query token", None)
two(C1, B, "Query latent", None)
two(C0, C, "Last 128", "tokens")
two(C2, C, "Window", "128 raw entries", ACC_FILL, ACC)
x, w = C3
s.rect(x, A[0], w, C[0] + C[1] - A[0], ACC_FILL, ACC, 2.5, 6)
s.lines(x + w / 2, (A[0] + C[0] + C[1]) / 2, ["Dense", "shared", "KV", "MQA"], 18, "bold", INK)

s.arrow(158, 76, 188, 76); s.arrow(352, 76, 390, 76); s.arrow(554, 76, 574, 76)
s.arrow(158, 169, 188, 169); s.path("M352,169 H574")
s.text(392, 162, "attention q", 16, "normal", INK, "start")
s.arrow(158, 254, 390, 254); s.arrow(554, 254, 574, 254)
s.badge(190, 46, 1, ACC)
s.badge(576, 46, 2, ACC)

# V4-Pro stack column
SX, SW = 736, 130
s.text(SX + SW / 2, 26, "V4-Pro stack", 18, "bold", INK)
cells = [("L0", "HCA"), ("L1", "HCA"), ("L2", "CSA"), ("L3", "HCA"), ("L4", "CSA"), None, ("L60", "CSA")]
y = 40
for c in cells:
    if c is None:
        s.text(SX + SW / 2, y + 14, "⋮", 20, "bold", GRAY)
        y += 26
        continue
    li, k = c
    hca = k == "HCA"
    s.rect(SX, y, SW, 30, ACC_FILL if hca else GRAY_FILL, ACC if hca else GRAY, 2.5 if hca else 2, 4)
    s.text(SX + SW / 2, y + 21, f"{li} {k}", 18, "bold" if hca else "normal", ACC if hca else INK)
    if li == "L3":
        zy0, zy1 = y, y + 30
    y += 38
s.path(f"M666,2 L{SX - 2},{zy0}", GRAY, 1.5, head=False, dash="5,4")
s.path(f"M666,298 L{SX - 2},{zy1}", GRAY, 1.5, head=False, dash="5,4")
s.badge(SX, 40, 3, ACC)
s.save(os.path.join(HERE, "fig_overview.svg"))
