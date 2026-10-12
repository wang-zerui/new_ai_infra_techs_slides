"""fig_timeline: Megatron-FSDP unit schedule for 3 FSDP units (one training step on one rank, schematic):
all-gather of unit k+1 overlaps compute of unit k (forward and backward), reduce-scatter of unit k's gradient
overlaps backward of unit k-1; weights are re-gathered for backward (primes); only the first all-gather and the last reduce-scatter are exposed."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 262, "fig_timeline: compute / all-gather / reduce-scatter streams")

X0 = 150
F, B, C = 62, 112, 48           # forward, backward, collective widths
YC, YA, YR, H = 44, 106, 158, 38
s.text(X0 - 10, YC + H / 2 + 6, "compute", 17, INK, anchor="end", weight="bold")
s.text(X0 - 10, YA + H / 2 + 6, "all-gather", 17, INK, anchor="end")
s.text(X0 - 10, YR + H / 2 + 6, "reduce-scatter", 17, INK, anchor="end")


def comp(x, w, lab, fwd):
    s.box(x, YC, w, H, lab, NEU_F if fwd else "#D9D9D9", NEU, size=17)


def coll(x, y, lab, exposed=False):
    s.box(x, y, C, H, lab, PRB_F if exposed else ACC_F, PRB if exposed else ACC, PRB if exposed else ACC, 16)


t = X0
coll(t, YA, "AG1", True)
t += C
xf = []
for k in range(3):                      # forward F1..F3
    comp(t, F, f"F{k + 1}", True)
    xf.append(t)
    t += F
# all-gathers: forward prefetch (AG2 under F1, AG3 under F2); backward re-gather (AG3' under F3)
coll(xf[0] + 4, YA, "AG2")
coll(xf[1] + 4, YA, "AG3")
coll(xf[2] + 4, YA, "AG3′")
xb = []
for k in (3, 2, 1):                     # backward B3..B1
    comp(t, B, f"B{k}", False)
    xb.append(t)
    t += B
coll(xb[0] + 4, YA, "AG2′")             # prefetch unit 2 under B3
coll(xb[1] + 4, YA, "AG1′")             # prefetch unit 1 under B2
coll(xb[1] + 4, YR, "RS3")             # grads of unit 3 reduced under B2
coll(xb[2] + 4, YR, "RS2")             # grads of unit 2 reduced under B1
coll(t, YR, "RS1", True)               # last one exposed
end = t + C
s.text(xf[0], YC - 10, "forward", 15, MUTED, anchor="start", style="italic")
s.text(xb[0], YC - 10, "backward", 15, MUTED, anchor="start", style="italic")
s.line(X0, 222, 860, 222, INK, 2, "arr")
s.text(860, 248, "time", 18, INK, anchor="end")
s.text(X0, 248, "one rank · three FSDP units · ′ marks the re-gather for backward", 15, MUTED, anchor="start",
       style="italic")
s.save(os.path.join(HERE, "fig_timeline.svg"))
