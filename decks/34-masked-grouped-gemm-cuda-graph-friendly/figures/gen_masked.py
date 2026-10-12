"""fig_masked: the masked layout. A = G fixed-capacity slots of max_m rows; masked_m (device int32[G]) marks how many
rows of each slot are valid; the kernel computes only tiles that hold valid rows; D has the same slots."""
import os
from tilekit import SVG, seg, hlines, outline, mono, EXP, INK, MUTED, ACC, ACC_F, NEU, NEU_F

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 292, "fig_masked: fixed slots + device-side masked_m")

MAXM, BM = 512, 128
Y0, H = 66, 196                  # slot top, slot height (= max_m rows)
px = H / MAXM
mm = [300, 60, 200]              # running example masked_m
SW, GAP = 62, 10


def slots(x0, w, label, out=False):
    s.text(x0 + (3 * w + 2 * GAP) / 2, Y0 - 34, label, 17, INK, weight="bold")
    for g, m in enumerate(mm):
        x = x0 + g * (w + GAP)
        h = m * px
        seg(s, x, Y0, w, h, g, None, sw=1.2)
        ntile = -(-m // BM)
        # tiles that hold no valid row: never scheduled
        for t in range(ntile, MAXM // BM):
            s.rect(x, Y0 + t * BM * px, w, BM * px, "#F7F7F7", "#BFBFBF", 1, rx=0, dash="3,3")
        # rows past masked_m inside the last scheduled tile: computed, ignored
        s.rect(x, Y0 + h, w, ntile * BM * px - h, "#FFFFFF", "#BFBFBF", 0.8, rx=0)
        hlines(s, x, 0, w, [Y0 + k * BM * px for k in range(1, MAXM // BM)], sw=1.4)
        s.line(x - 3, Y0 + h, x + w + 3, Y0 + h, ACC, 3)
        outline(s, x, Y0, w, H, INK, 1.8)
        s.text(x + w / 2, Y0 + H + 18, f"E{g}", 16, INK, weight="bold")


XA, XB, XD = 108, 382, 520
slots(XA, SW, "A (G × max_m × K)")
slots(XD, SW, "D (G × max_m × N)")

# masked_m device tensor, one cell per slot, pointing at the valid boundary
s.text(58, Y0 - 34 + 0, "masked_m", 15, ACC, weight="bold", family="Menlo, Consolas, monospace")
s.text(58, Y0 - 16, "(on GPU)", 15, ACC)
for g, m in enumerate(mm):
    x = XA + g * (SW + GAP)
    s.box(x + 8, Y0 - 24, SW - 16, 20, str(m), ACC_F, ACC, ACC, 15, "bold")
    s.line(x + SW / 2, Y0 - 4, x + SW / 2, Y0 + m * px - 3, ACC, 1.6, "arrAcc", dash="3,2")

# B slabs
s.text(XB + 40, Y0 - 34, "B (G × N × K)", 17, INK, weight="bold")
for g in range(3):
    seg(s, XB + 10, Y0 + 8 + g * 66, 60, 54, g, f"B[{g}]", size=16)

s.text((XA + 3 * SW + 2 * GAP + XB + 10) / 2, Y0 + H / 2 + 9, "×", 26, MUTED)
s.text((XB + 70 + XD) / 2, Y0 + H / 2 + 9, "=", 26, MUTED)
# row-size bracket
s.text(XA - 8, Y0 + 6, "0", 15, MUTED, anchor="end", family="Menlo, Consolas, monospace")
s.text(XA - 8, Y0 + H + 4, "max_m", 15, MUTED, anchor="end", family="Menlo, Consolas, monospace")

# legend
LX = 744
s.rect(LX, 80, 22, 16, EXP[0][0], EXP[0][1], 1.2, rx=0)
s.text(LX + 30, 93, "valid rows", 15, INK, anchor="start")
s.rect(LX, 112, 22, 16, "#FFFFFF", "#BFBFBF", 1, rx=0)
s.text(LX + 30, 125, "computed", 15, INK, anchor="start")
s.text(LX + 30, 143, "then ignored", 15, INK, anchor="start")
s.rect(LX, 160, 22, 16, "#F7F7F7", "#BFBFBF", 1, rx=0, dash="3,3")
s.text(LX + 30, 173, "tile never", 15, INK, anchor="start")
s.text(LX + 30, 191, "scheduled", 15, INK, anchor="start")
s.line(LX, 216, LX + 22, 216, ACC, 3)
s.text(LX + 30, 221, "masked_m", 15, ACC, anchor="start", family="Menlo, Consolas, monospace")
s.save(os.path.join(HERE, "fig_masked.svg"))
