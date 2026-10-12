"""fig_overview: GLM-4.5 at three zoom levels — model (92 x 5120), one attention layer (96 Q / 8 KV heads,
QK-Norm, partial RoPE), one head (64 rotated + 64 position-free dims).

Order inside the attention block follows transformers modeling_glm4_moe.py (Glm4MoeAttention.forward):
q/k/v proj -> q_norm/k_norm (per-head RMSNorm) -> partial RoPE on q, k -> attention (GQA) -> o_proj.
"""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 360
s = SVG(W, H, "fig_overview: model -> attention layer -> one head")

# ---------------- panel A: the model stack
s.badge(16, 20, 1)
s.text(36, 27, "Model", 20, INK, anchor="start", weight="bold")
SX, SW, SY0, SY1, L = 62, 60, 52, 322, 92
pitch = (SY1 - SY0) / L
s.rect(SX, SY0, SW, SY1 - SY0, ACC_F, ACC, 2.2, rx=2)
for i in range(1, L):
    yy = SY1 - i * pitch
    s.line(SX + 1.5, yy, SX + SW - 1.5, yy, "#A9C4E0", 0.8)
s.rect(SX, SY1 - 3 * pitch, SW, 3 * pitch, ACC, None, 0, rx=0)
HL_Y = SY0 + 44 * pitch
s.rect(SX - 5, HL_Y - 4, SW + 10, pitch + 8, "none", ACC, 2.6, rx=1)
s.text(40, (SY0 + SY1) / 2, "92 layers", 18, INK, rotate=-90)
s.text(SX + SW / 2, 346, "5120 wide", 18, INK)

# ---------------- panel B: one attention layer
BX0, BY0, BX1, BY1 = 214, 40, 616, 352
s.rect(BX0, BY0, BX1 - BX0, BY1 - BY0, "#FFFFFF", MUTED, 1.6, rx=6, dash="7,4")
s.text(BX0 + 2, 27, "One attention layer", 20, INK, anchor="start", weight="bold")
# zoom lines from the highlighted layer
s.line(SX + SW + 5, HL_Y - 4, BX0, BY0, MUTED, 1.4, dash="5,4")
s.line(SX + SW + 5, HL_Y + pitch + 4, BX0, BY1, MUTED, 1.4, dash="5,4")

s.box(316, 52, 200, 34, "hidden state (5120)", NEU_F, NEU, size=18)
VX, QX, KX = (224, 324), (336, 492), (505, 605)
R2 = (106, 146)
s.box(VX[0], R2[0], VX[1] - VX[0], R2[1] - R2[0], "V: 8 heads", ACC_F, ACC, size=18)
s.box(QX[0], R2[0], QX[1] - QX[0], R2[1] - R2[0], "Q: 96 heads", ACC_F, ACC, size=18, weight="bold")
s.box(KX[0], R2[0], KX[1] - KX[0], R2[1] - R2[0], "K: 8 heads", ACC_F, ACC, size=18)
s.badge(QX[0] + 2, R2[0] + 1, 2, r=12, size=16)
for cx in ((VX[0] + VX[1]) / 2, (QX[0] + QX[1]) / 2, (KX[0] + KX[1]) / 2):
    sx = min(max(cx, 330), 500)
    s.line(sx, 86, cx, R2[0] - 1, INK, 2, "arr")

R3 = (164, 196)
s.box(QX[0], R3[0], KX[1] - QX[0], R3[1] - R3[0], "QK-Norm: per-head RMSNorm", "#FFFFFF", ACC, size=18, dash="6,3")
R4 = (214, 246)
s.box(QX[0], R4[0], KX[1] - QX[0], R4[1] - R4[0], "partial RoPE", ACC_F, ACC, size=18, weight="bold", sw=2.4)
R5 = (266, 300)
s.box(VX[0], R5[0], KX[1] - VX[0], R5[1] - R5[0], "attention (12 Q heads per KV head)", NEU_F, NEU, size=18)
s.box(316, 316, 200, 30, "output proj (5120)", NEU_F, NEU, size=18)

qc, kc, vc = (QX[0] + QX[1]) / 2, (KX[0] + KX[1]) / 2, (VX[0] + VX[1]) / 2
for cx in (qc, kc):
    s.line(cx, R2[1], cx, R3[0] - 1, INK, 2, "arr")
    s.line(cx, R3[1], cx, R4[0] - 1, INK, 2, "arr")
    s.line(cx, R4[1], cx, R5[0] - 1, INK, 2, "arr")
s.line(vc, R2[1], vc, R5[0] - 1, INK, 2, "arr")
s.line(416, R5[1], 416, 315, INK, 2, "arr")

# ---------------- panel C: one head
CX0 = 652
s.badge(CX0 + 14, 20, 3)
s.text(CX0 + 34, 27, "One head", 20, INK, anchor="start", weight="bold")
HX, HW, HY0, HY1 = 668, 46, 58, 298
mid = (HY0 + HY1) / 2
s.rect(HX, HY0, HW, mid - HY0, "url(#hatchAcc)", ACC, 2.2, rx=0)
s.rect(HX, mid, HW, HY1 - mid, "#FFFFFF", NEU, 2.2, rx=0)
s.line(KX[1], R4[0], HX, HY0, MUTED, 1.4, dash="5,4")
s.line(KX[1], R4[1], HX, HY1, MUTED, 1.4, dash="5,4")
TX = HX + HW + 10
s.text(TX, HY0 + 44, "dims 0–63", 18, ACC, anchor="start", weight="bold")
s.text(TX, HY0 + 66, "rotated", 18, ACC, anchor="start")
s.text(TX, HY0 + 88, "by position", 18, ACC, anchor="start")
s.text(TX, mid + 44, "dims 64–127", 18, INK, anchor="start", weight="bold")
s.text(TX, mid + 66, "position-free", 18, INK, anchor="start")
s.text(HX + HW / 2, 322, "128 dims", 18, INK)

s.save(os.path.join(HERE, "fig_overview.svg"))
