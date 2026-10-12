"""fig_statusquo: (a) batched GEMM with a fixed capacity pads or drops; (b) one GEMM per expert = many launches
with wave tails and a host read of the token counts."""
import os
from tilekit import SVG, seg, waste, INK, MUTED, PRB, PRB_F, NEU, NEU_F, EXP

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 292, "fig_statusquo: capacity batching vs per-expert launches")

# ---------------- panel (a): batched GEMM with capacity C
s.text(170, 24, "(a) Batched GEMM, fixed capacity", 19, INK, weight="bold")
BASE, CAP = 250, 150                         # slot bottom, capacity height
xs = [34, 116, 198]
toks = [190, 50, 100]                        # rows routed to E0, E1, E2 (E0 overflows)
for e, (x, t) in enumerate(zip(xs, toks)):
    inside = min(t, CAP)
    seg(s, x, BASE - inside, 66, inside, e, f"E{e}", size=18)
    if t < CAP:
        waste(s, x, BASE - CAP, 66, CAP - t, "padding" if CAP - t > 60 else "pad", size=15)
    else:
        waste(s, x, BASE - t, 66, t - CAP, None)
        s.text(x + 33, BASE - t + (t - CAP) / 2 + 6, "dropped", 15, PRB, weight="bold", halo=True)
s.line(24, BASE - CAP, 356, BASE - CAP, INK, 1.8, dash="6,4")
s.text(356, BASE - CAP - 8, "capacity C", 16, INK, anchor="end", style="italic")
s.line(24, BASE, 356, BASE, INK, 1.8)
s.text(180, 278, "same M for every expert slot", 16, MUTED, style="italic")

# divider
s.line(372, 14, 372, 286, "#999999", 1)

# ---------------- panel (b): one GEMM per expert
s.text(620, 24, "(b) One GEMM launch per expert", 19, INK, weight="bold")
X0 = 470
s.text(X0 - 10, 72, "host", 17, INK, anchor="end", weight="bold")
s.text(X0 - 10, 168, "SMs", 17, INK, anchor="end", weight="bold")
# host: wait for counts, then three launches
s.box(X0, 54, 118, 32, "read counts", PRB_F, PRB, PRB, 16, "bold")
launch_x = [X0 + 122, X0 + 264, X0 + 330]
for lx in launch_x:
    s.box(lx, 58, 60, 24, "launch", NEU_F, NEU, INK, 15)
# GPU lane: idle while host waits, then kernels with tails
ROWS = [118, 140, 162, 184]                 # four SM rows
for y in ROWS:
    s.rect(X0, y, 118, 18, "#FFFFFF", MUTED, 1, rx=0, dash="3,3")
s.text(X0 + 59, 110, "idle", 15, MUTED, style="italic")
tails = []
kern = [(0, X0 + 126, 140, 2), (1, X0 + 274, 62, 3), (2, X0 + 344, 46, 1)]   # (expert, x, width, rows busy in tail)
for e, kx, kw, busy in kern:
    f, st = EXP[e]
    tail = 0.3 * kw
    for i, y in enumerate(ROWS):
        full = kw if i < busy else kw - tail
        s.rect(kx, y, full, 18, f, st, 1.2, rx=0)
        if full < kw:
            s.rect(kx + full, y, tail, 18, "url(#hatchPrb)", PRB, 1, rx=0)
    s.text(kx + kw / 2, 110, f"E{e}", 16, INK, weight="bold")
    tails.append(kx + kw - tail / 2)
s.text(X0 + 250, 248, "wave tails", 16, PRB, weight="bold")
for tx in tails:
    s.line(X0 + 250 + (tx - X0 - 250) * 0.25, 232, tx, 205, PRB, 1.6, "arrPrb")
s.line(X0, 262, 860, 262, INK, 1.8, "arr")
s.text(856, 284, "time", 15, MUTED, anchor="end")
s.save(os.path.join(HERE, "fig_statusquo.svg"))
