"""fig_pipeline: the decode-step sampling pipeline (stage order from hpc/sampler.py docstring), with the
penalty state carried to the next step."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 480, 340
s = SVG(W, H, "fig_pipeline")
BW, BH, GAP = 196, 50, 28
XL, XR, Y0 = 40, 276, 10
left = [("logits  B × V", True), ("repetition penalty", False), ("temperature", False), ("top-k", False)]
right = [("softmax", False), ("top-p", False), ("random draw", False), ("penalty-mask update", False)]
for i, (lab, inp) in enumerate(left):
    y = Y0 + i * (BH + GAP)
    s.box(XL, y, BW, BH, lab, "#FFFFFF" if inp else NEU_F, NEU, size=18, weight="bold" if inp else "normal")
    if i < 3:
        s.line(XL + BW / 2, y + BH + 2, XL + BW / 2, y + BH + GAP - 2, INK, 2.2, "arr")
for i, (lab, _) in enumerate(right):
    y = Y0 + i * (BH + GAP)
    s.box(XR, y, BW, BH, lab, NEU_F, NEU, size=18)
    if i < 3:
        s.line(XR + BW / 2, y + BH + 2, XR + BW / 2, y + BH + GAP - 2, INK, 2.2, "arr")
# top-k (left bottom) -> softmax (right top)
yk = Y0 + 3 * (BH + GAP) + BH / 2
ys = Y0 + BH / 2
s.path(f"M{XL + BW + 2},{yk} L{(XL + BW + XR) / 2},{yk} L{(XL + BW + XR) / 2},{ys} L{XR - 2},{ys}", INK, 2.2, "arr")
# next-step loop: mask update -> repetition penalty
ym = Y0 + 3 * (BH + GAP) + BH
yp = Y0 + 1 * (BH + GAP) + BH / 2
s.path(f"M{XR + BW / 2},{ym + 2} L{XR + BW / 2},{H - 10} L{16},{H - 10} L{16},{yp} L{XL - 2},{yp}", MUTED, 2.2,
       "arrGray", dash="7,4")
s.text(XR - 30, H - 16, "state for the next step", 17, MUTED, anchor="end", style="italic")
s.save(os.path.join(HERE, "fig_pipeline.svg"))
