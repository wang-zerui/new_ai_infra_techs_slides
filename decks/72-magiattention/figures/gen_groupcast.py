"""fig_groupcast: causal mask, 4 ranks, zigzag (rank r holds chunks r and 7-r). KV5 lives on rank2 and is needed
only by Q6 (rank1) and Q7 (rank0) ([BLOG] Group Collective Primitives).
(a) ring P2P forwards KV5 rank2 -> rank3 (unused) -> rank0 -> rank1.
(b) GroupCast sends KV5 straight to rank0 and rank1; in backward GroupReduce sums their partial dKV5 on rank2."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 294, "fig_groupcast: ring vs GroupCast/GroupReduce for KV5")

W, GAP = 140, 50
XS = [150 + i * (W + GAP) for i in range(4)]
CX = [x + W / 2 for x in XS]


def ranks(y, hot=None, dim=None):
    for r in range(4):
        fill, stroke = NEU_F, NEU
        if r == hot:
            fill, stroke = ACC_F, ACC
        s.rect(XS[r], y, W, 60, fill, stroke, 2, 5)
        s.text(CX[r], y + 25, f"rank{r}", 18, ACC if r == hot else INK, weight="bold")
        s.text(CX[r], y + 49, f"chunks {r}, {7 - r}", 16, INK)


# ---------------- (a) ring
YA = 30
s.text(10, YA + 28, "(a) Ring P2P", 18, INK, anchor="start", weight="bold")
ranks(YA, hot=2)
ym = YA + 30
s.line(XS[2] + W + 3, ym, XS[3] - 3, ym, PRB, 2.4, "arrPrb", dash="7,4")
s.text((XS[2] + W + XS[3]) / 2, ym - 10, "KV5", 16, PRB, weight="bold", halo=True)
s.path(f"M{CX[3]},{YA + 63} L{CX[3]},{YA + 90} L{CX[0]},{YA + 90} L{CX[0]},{YA + 64}", MUTED, 2.2, "arrGray")
s.text(CX[0] + 60, YA + 86, "KV5", 16, MUTED, weight="bold", halo=True)
s.line(XS[0] + W + 3, ym, XS[1] - 3, ym, MUTED, 2.2, "arrGray")
s.text((XS[0] + W + XS[1]) / 2, ym - 10, "KV5", 16, MUTED, weight="bold", halo=True)
s.text(XS[3] + W / 2, YA - 8, "never uses KV5", 16, PRB, weight="bold")

# ---------------- (b) GroupCast / GroupReduce
YB = 186
s.text(10, YB + 24, "(b) GroupCast", 18, ACC, anchor="start", weight="bold")
s.text(10, YB + 46, "+ GroupReduce", 17, ACC, anchor="start")
ranks(YB, hot=2)
# cast: above the row
s.path(f"M{CX[2] - 10},{YB - 3} C{CX[2] - 10},{YB - 26} {CX[1] + 10},{YB - 26} {CX[1] + 10},{YB - 3}", ACC, 2.4, "arrAcc")
s.path(f"M{CX[2] + 20},{YB - 3} C{CX[2] + 20},{YB - 56} {CX[0]},{YB - 56} {CX[0]},{YB - 3}", ACC, 2.4, "arrAcc")
s.text(CX[0] + 4, YB - 40, "KV5", 16, ACC, anchor="end", weight="bold")
# reduce: below the row
s.path(f"M{CX[0]},{YB + 63} C{CX[0]},{YB + 100} {CX[2] - 30},{YB + 100} {CX[2] - 30},{YB + 63}", ACC, 2.2,
       "arrAcc", dash="6,4")
s.path(f"M{CX[1]},{YB + 63} C{CX[1]},{YB + 86} {CX[2] - 10},{YB + 86} {CX[2] - 10},{YB + 63}", ACC, 2.2,
       "arrAcc", dash="6,4")
s.text(CX[3], YB + 82, "partial dKV5", 16, ACC, weight="bold")
s.text(CX[3], YB + 100, "summed on rank2", 16, ACC, weight="bold")
s.save(os.path.join(HERE, "fig_groupcast.svg"))
