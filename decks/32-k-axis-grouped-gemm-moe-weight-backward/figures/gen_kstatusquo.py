"""fig_kstatusquo: (a) one wgrad GEMM per expert: the host holds the K list and launches G kernels with gaps;
(b) padding every expert's tokens to the longest expert: zero tokens still cost full FLOPs."""
import os
from tilekit import SVG, seg, waste, outline, EXP, INK, MUTED, PRB, PRB_F, NEU, NEU_F

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 292, "fig_kstatusquo: per-expert wgrad launches vs pad-to-max")

# ---------- (a) one GEMM per expert
s.text(10, 24, "(a) One Wgrad GEMM per expert", 19, INK, anchor="start", weight="bold")
X0 = 70
s.text(470, 50, "L = kernel launch", 15, MUTED, anchor="end")
s.text(X0 - 8, 76, "host", 17, INK, anchor="end", weight="bold")
s.text(X0 - 8, 156, "GPU", 17, INK, anchor="end", weight="bold")
s.box(X0, 58, 96, 30, "K list", PRB_F, PRB, PRB, 16, "bold")
kern = [(0, 104), (1, 40), (2, 70)]
cur = X0 + 136
for i, (e, w) in enumerate(kern):
    s.box(cur - 30, 60, 26, 26, "L", NEU_F, NEU, INK, 15)
    f, st = EXP[e]
    s.rect(cur, 138, w, 30, f, st, 1.4, rx=0)
    s.text(cur + w / 2, 159, f"dW{e}", 15, INK, weight="bold")
    if i < len(kern) - 1:
        s.rect(cur + w, 138, 26, 30, "url(#hatchPrb)", PRB, 1, rx=0)
    cur += w + 26
s.line(X0, 196, 478, 196, INK, 1.6, "arr")
s.text(474, 214, "time", 15, MUTED, anchor="end")
s.text(250, 246, "G launches per weight per layer", 17, PRB, weight="bold")
s.text(250, 268, "token counts must reach the host", 16, PRB)

s.line(488, 14, 488, 286, "#999999", 1)

# ---------- (b) pad to the longest expert
s.text(508, 24, "(b) Pad every expert to the longest", 19, INK, anchor="start", weight="bold")
KMAX, BASE = 180, 214
xs = [540, 640, 740]
toks = [180, 36, 120]
for e, (x, t) in enumerate(zip(xs, toks)):
    seg(s, x, BASE - KMAX, 70, t, e, f"E{e}", size=17)
    if t < KMAX:
        waste(s, x, BASE - KMAX + t, 70, KMAX - t, None)
    outline(s, x, BASE - KMAX, 70, KMAX, INK, 1.8)
s.text(524, BASE - KMAX / 2 + 6, "K", 17, INK, weight="bold")
s.text(675, 246, "zero tokens cost full FLOPs", 17, PRB, weight="bold")
s.text(675, 268, "K = longest expert for all", 16, PRB)
s.save(os.path.join(HERE, "fig_kstatusquo.svg"))
