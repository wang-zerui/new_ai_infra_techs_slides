"""fig_merge: plain 1F1B steady state vs merged FWD-BWD steps (one pipeline stage, schematic)."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 296, "fig_merge: F1 B0 F2 B1 F3 B2 with exposed a2a  vs  (F1+B0)(F2+B1)(F3+B2) with a2a on a second stream")

X0 = 150
FC, BC, A = 56, 112, 34            # forward compute, backward compute, a2a (per pass, all layers)
YA, HA = 44, 46                    # row A
YB1, YB2, HB = 146, 196, 42        # row B lanes

# legend
s.rect(470, 10, 26, 18, PRB_F, PRB, 2, 3)
s.text(504, 25, "exposed all-to-all", 17, INK, anchor="start")
s.rect(666, 10, 26, 18, ACC_F, ACC, 2, 3)
s.text(700, 25, "hidden all-to-all", 17, INK, anchor="start")

# row A: plain 1F1B
s.text(136, YA + HA / 2 + 6.3, "plain 1F1B", 18, INK, anchor="end", weight="bold")
x = X0
for k in range(3):
    for tag, comp in ((f"F{k + 1}", FC), (f"B{k}", BC)):
        s.box(x, YA, comp, HA, tag, NEU_F, NEU)
        s.rect(x + comp, YA, A, HA, PRB_F, PRB, 2, 3)
        x += comp + A
endA = x

# row B: merged steps on two streams
s.text(10, YB1 - 14, "merged FWD-BWD", 18, ACC, anchor="start", weight="bold")
s.text(136, YB1 + HB / 2 + 6.3, "compute", 18, INK, anchor="end")
s.text(136, YB2 + HB / 2 + 6.3, "comm", 18, INK, anchor="end")
x = X0
for k in range(3):
    w = FC + BC
    s.rect(x - 3, YB1 - 6, w + 6, YB2 + HB - YB1 + 12, "none", ACC, 1.5, 6, dash="6,4")
    s.box(x, YB1, w, HB, f"F{k + 1} + B{k}", ACC_F, ACC, color=ACC, weight="bold")
    s.box(x + (w - 2 * A - 8) / 2, YB2, 2 * A + 8, HB, "a2a", ACC_F, ACC, color=ACC)
    x += w + 12
endB = x - 12

# saved-time bracket
s.line(endA, YA - 4, endA, YB2 + HB + 6, MUTED, 1.5, dash="6,5")
s.line(endB + 3, YA + HA + 8, endB + 3, YB2 + HB + 6, MUTED, 1.5, dash="6,5")
ym = (YB1 + YB2 + HB) / 2
s.line(endB + 6, ym, endA - 2, ym, ACC, 2.2, "arrAcc", start_marker="arrAcc")
s.text((endB + endA) / 2, ym - 12, "time saved", 18, ACC, weight="bold")

# time axis
s.line(X0 - 4, 262, 862, 262, INK, 2, "arr")
s.text(858, 288, "time", 18, INK, anchor="end")
s.text(X0, 288, "one pipeline stage, steady state", 17, MUTED, anchor="start", style="italic")
s.save(os.path.join(HERE, "fig_merge.svg"))
