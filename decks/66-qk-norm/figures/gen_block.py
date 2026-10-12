"""fig_block: where QK-Norm sits in a GLM-4.5 attention layer.

Order from transformers modeling_glm4_moe.py (Glm4MoeAttention.forward): q/k/v proj (with bias) -> q_norm/k_norm
(per-head RMSNorm, 128-dim weight) -> partial RoPE -> KV cache update -> softmax(q.k * 128^-0.5) V.
Heads: 96 query / 8 KV (GLM-4.5 Table 1).
"""
import os
from svgkit import SVG, rich, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 300
s = SVG(W, H, "fig_block: GLM-4.5 attention with QK-Norm")
RQ, RK, RV = (20, 72), (100, 152), (226, 278)
C0, C1, C2, C3, C4, C5 = (2, 92), (116, 240), (266, 410), (436, 540), (566, 666), (694, 866)


def mid(r):
    return (r[0] + r[1]) / 2


s.box(C0[0], RQ[0], C0[1] - C0[0], RV[1] - RQ[0], ["normed", "input x"], NEU_F, NEU, size=19)
for r, lab in ((RQ, ["q proj", "96 heads"]), (RK, ["k proj", "8 heads"]), (RV, ["v proj", "8 heads"])):
    s.box(C1[0], r[0], C1[1] - C1[0], r[1] - r[0], lab, NEU_F, NEU, size=18)
    s.line(C0[1] + 2, mid(r), C1[0] - 2, mid(r), INK, 2, "arr")
for r in (RQ, RK):
    s.box(C2[0], r[0], C2[1] - C2[0], r[1] - r[0], ["RMSNorm", "per head"], ACC_F, ACC, ACC, size=19, weight="bold", sw=2.6)
    s.line(C1[1] + 2, mid(r), C2[0] - 2, mid(r), INK, 2, "arr")
    s.box(C3[0], r[0], C3[1] - C3[0], r[1] - r[0], ["partial", "RoPE"], NEU_F, NEU, size=18)
    s.line(C2[1] + 2, mid(r), C3[0] - 2, mid(r), INK, 2, "arr")
s.text(mid(C2), RK[1] + 24, "learned gains (128)", 17, ACC, style="italic")
s.badge(C2[0] + 2, RQ[0] + 2, 1, r=12, size=16)
for r in (RK, RV):
    s.box(C4[0], r[0], C4[1] - C4[0], r[1] - r[0], ["KV", "cache"], "#FFFFFF", NEU, size=18)
s.badge(C4[0] + 2, RK[0] + 2, 3, r=12, size=16)
s.line(C3[1] + 2, mid(RK), C4[0] - 2, mid(RK), INK, 2, "arr")
s.line(C1[1] + 2, mid(RV), C4[0] - 2, mid(RV), INK, 2, "arr")
# logits, softmax, xV
LG = (RQ[0], RK[1])
s.rect(C5[0], LG[0], C5[1] - C5[0], LG[1] - LG[0], ACC_F, ACC, 2.6, rx=4)
rich(s, mid(C5), mid(LG) - 4, [("logits",)], 19, weight="bold", color=ACC)
rich(s, mid(C5), mid(LG) + 22, [("q·k / √128",)], 19, color=ACC)
s.badge(C5[0] + 2, LG[0] + 2, 2, r=12, size=16)
s.line(C3[1] + 2, mid(RQ), C5[0] - 2, mid(RQ), INK, 2, "arr")
s.line(C4[1] + 2, mid(RK), C5[0] - 2, mid(RK), INK, 2, "arr")
SM = (172, 206)
s.box(C5[0] + 30, SM[0] + 6, C5[1] - C5[0] - 60, SM[1] - SM[0] - 6, "softmax", NEU_F, NEU, size=18)
s.line(mid(C5), LG[1] + 1, mid(C5), SM[0] + 4, INK, 2, "arr")
s.box(C5[0], RV[0], C5[1] - C5[0], RV[1] - RV[0], "weighted sum of V", NEU_F, NEU, size=18)
s.line(mid(C5), SM[1] + 1, mid(C5), RV[0] - 2, INK, 2, "arr")
s.line(C4[1] + 2, mid(RV), C5[0] - 2, mid(RV), INK, 2, "arr")
s.save(os.path.join(HERE, "fig_block.svg"))
