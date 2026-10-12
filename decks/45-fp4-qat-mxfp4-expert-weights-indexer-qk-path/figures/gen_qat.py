"""Mechanism 1: expert-weight QAT forward/backward ([V4] §5.2.1; MX transposition note from [MX] §3)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svgkit import SVG, ACC, ACC_FILL, GRAY, GRAY_FILL, INK, RED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 284
s = SVG(W, H)
X = [0, 236, 472, 708]
BW, Y0, BH = 160, 20, 70


def two(i, y, l1, l2, hot=False, white=False):
    x = X[i]
    s.rect(x, y, BW, BH, ACC_FILL if hot else ("#fff" if white else GRAY_FILL), ACC if hot else GRAY,
           2.5 if hot else 2, 6)
    s.text(x + BW / 2, y + BH / 2 - 4, l1, 18, "bold", INK)
    s.text(x + BW / 2, y + BH / 2 + 18, l2, 18, "normal", INK)


two(0, Y0, "FP32 master", "expert weight W", white=True)
two(1, Y0, "MXFP4 W", "E2M1 + E8M0/32", True)
two(2, Y0, "FP8 W", "E4M3 + 128×128", False)
two(3, Y0, "FP8 GEMM", "X · W", False)
for i, lab in enumerate(["quantize", "exact", "forward"]):
    xa, xb = X[i] + BW + 2, X[i + 1] - 2
    s.arrow(xa, Y0 + BH / 2, xb, Y0 + BH / 2)
    s.text((xa + xb) / 2, Y0 + BH / 2 - 10, lab, 16, "normal", INK)

# backward
s.path(f"M{X[3] + BW / 2},{Y0 + BH + 2} V{150}", INK, 2, dash="6,4")
s.rect(X[3], 152, BW, 58, GRAY_FILL, GRAY, 2, 6)
s.text(X[3] + BW / 2, 176, "∂L / ∂W", 18, "bold", INK)
s.text(X[3] + BW / 2, 198, "w.r.t. FP8 W", 18, "normal", INK)
s.path(f"M{X[3] - 2},181 H{X[0] + BW / 2} V{Y0 + BH + 4}", ACC, 2.5)
s.text((X[1] + X[2] + BW) / 2, 172, "straight to the master (STE)", 18, "bold", ACC)
# skipped step
s.rect(250, 218, 360, 52, "#fff", GRAY, 2, 6, "6,4")
s.text(430, 250, "re-quantize Wᵀ into FP4", 18, "normal", GRAY)
s.text(626, 254, "✗ skipped", 18, "bold", RED, "start")
s.save(os.path.join(HERE, "fig_qat.svg"))
