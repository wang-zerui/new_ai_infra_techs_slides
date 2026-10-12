"""fig_inverse.svg: chunk size and the in-chunk triangular inverse. FLA uses 64-token chunks with secondary 16-token
sub-chunks (log-space rescaling); FlashKDA v1 uses 16-token chunks because g >= -5 keeps exp(cumsum g) in BF16 range,
and inverts the 16x16 lower-triangular matrix as two 8x8 FP32 forward substitutions plus two BF16 HMMA merge GEMMs.
Sources: FlashKDA deep dive Sec. 1 and 3; K3 Sec. 2.1.1; Kimi Linear Sec. 6.2."""
import pathlib, sys; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from svgkit import *
s = SVG(870, 285)
# left: 64-token chunk
s.t(170, 24, "64-token chunk (FLA)", 19, "bold")
x0, y0, b = 90, 40, 38
for i in range(4):
    for j in range(4):
        if j > i:
            s.box(x0 + j * b, y0 + i * b, b, b, "#FFFFFF", "#BFBFBF", 1, 0)
for i in range(4):
    for j in range(i + 1):
        if i == j:
            s.box(x0 + j * b, y0 + i * b, b, b, "url(#hr)", RED, 1.5, 0)
        else:
            s.box(x0 + j * b, y0 + i * b, b, b, GRAY_F, GRAY, 1.5, 0)
s.t(170, 218, "16-token sub-chunks", 17, "normal", RED)
s.t(170, 240, "rescaled in log space", 17, "normal", RED)
s.t(170, 266, "64×64 triangular inverse", 17, "normal", RED)
# middle arrow
s.line(320, 116, 470, 116, ACC, 3, "b")
s.t(395, 104, "g ≥ −5", 19, "bold", ACC)
s.t(395, 142, "16 tokens fit", 16)
s.t(395, 162, "in BF16 range", 16)
# right: 16x16 = 2x2 blocks of 8
s.t(640, 24, "16-token chunk (FlashKDA)", 19, "bold", ACC)
x1, y1, B = 488, 40, 76
s.box(x1 + B, y1, B, B, "#FFFFFF", "#BFBFBF", 1, 0)
s.box(x1, y1, B, B, ACC_F, ACC, 2, 0); s.t(x1 + B / 2, y1 + B / 2 + 7, "A", 20, "bold", ACC)
s.box(x1 + B, y1 + B, B, B, ACC_F, ACC, 2, 0); s.t(x1 + 1.5 * B, y1 + 1.5 * B + 7, "C", 20, "bold", ACC)
s.box(x1, y1 + B, B, B, "#FFFFFF", ACC, 2, 0); s.t(x1 + B / 2, y1 + 1.5 * B + 7, "B", 20, "bold", ACC)
s.t(x1 + B, y1 + 2 * B + 20, "I + L  (16×16)", 16, "normal", GRAY)
s.t(660, 70, "A, C: 8×8 blocks", 17, "bold", ACC, "start")
s.t(660, 92, "FP32 forward", 17, "normal", INK, "start")
s.t(660, 112, "substitution", 17, "normal", INK, "start")
s.t(660, 150, "B: merged by two", 17, "bold", ACC, "start")
s.t(660, 172, "BF16 HMMA GEMMs", 17, "normal", INK, "start")
s.t(640, 262, "off-diagonal block of the inverse = −C⁻¹ · B · A⁻¹", 17, "normal", INK)
s.save(pathlib.Path(__file__).with_name("fig_inverse.svg"))
