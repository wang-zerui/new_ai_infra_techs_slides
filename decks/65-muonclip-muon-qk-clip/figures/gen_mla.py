"""fig_mla: per-head QK-Clip on MLA. logit_h = q_C.k_C + q_R.k_R; q_C, k_C scaled by sqrt(gamma); q_R by gamma; shared k_R
untouched; both terms scale by gamma, so the logit scales by gamma. Source: K2 §2.1, Algorithm 1; DeepSeek-V3 §2.1.1."""
import os
from svgkit import rich, SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 262, "fig_mla")

def vec(x, y, w, h, base, sub, fill, stroke, color=INK, dash=None):
    s.rect(x, y, w, h, fill, stroke, 2, 4, dash=dash)
    rich(s, x + w / 2, y + h / 2 + 8, [(base, ""), (sub, "sub")], 22, color, weight="bold")

# column headers
s.text(85, 26, "query of head h", 17, MUTED, style="italic")
s.text(330, 26, "key", 17, MUTED, style="italic")
s.text(560, 26, "logit term", 17, MUTED, style="italic")
RY = [48, 150]
H = 54
# content row
vec(20, RY[0], 130, H, "q", "C", ACC_F, ACC, ACC)
rich(s, 158, RY[0] + 35, [("× √γ", "")], 22, ACC, anchor="start", weight="bold")
s.text(240, RY[0] + 36, "·", 30, INK, weight="bold")
vec(266, RY[0], 130, H, "k", "C", ACC_F, ACC, ACC)
rich(s, 404, RY[0] + 35, [("× √γ", "")], 22, ACC, anchor="start", weight="bold")
# rotary row
vec(20, RY[1], 130, H, "q", "R", ACC_F, ACC, ACC)
rich(s, 158, RY[1] + 35, [("× γ", "")], 22, ACC, anchor="start", weight="bold")
s.text(240, RY[1] + 36, "·", 30, INK, weight="bold")
vec(266, RY[1], 130, H, "k", "R", NEU_F, NEU, INK, dash="6,4")
rich(s, 404, RY[1] + 35, [("× 1", "")], 22, MUTED, anchor="start", weight="bold")
s.text(331, RY[1] + H + 26, "shared by all heads", 18, MUTED)
s.text(331, RY[1] + H + 48, "left untouched", 18, MUTED)
# results
for y, a, b in ((RY[0], "C", "C"), (RY[1], "R", "R")):
    s.line(468, y + H / 2, 492, y + H / 2, INK, 2.2, "arr")
    s.rect(494, y, 140, H, "#FFFFFF", ACC, 2, 4)
    rich(s, 564, y + H / 2 + 8, [("γ · q", ""), (a, "sub"), ("·k", ""), (b, "sub")], 22, ACC, weight="bold")
# merge into the logit through a + node
yc = (RY[0] + RY[1] + H) / 2
s.path(f"M 636,{RY[0] + H / 2} L 672,{RY[0] + H / 2} L 672,{yc - 14}", INK, 2.2)
s.path(f"M 636,{RY[1] + H / 2} L 672,{RY[1] + H / 2} L 672,{yc + 14}", INK, 2.2)
s.add(f'<circle cx="672" cy="{yc}" r="14" fill="#FFFFFF" stroke="{INK}" stroke-width="2.2"/>')
s.text(672, yc + 8, "+", 24, INK, weight="bold")
s.line(686, yc, 708, yc, INK, 2.2, "arr")
s.rect(710, yc - 34, 154, 68, ACC_F, ACC, 2.4, 6)
rich(s, 787, yc - 4, [("logit", ""), ("h", "sub")], 22, ACC, weight="bold")
s.text(787, yc + 24, "× γ exactly", 20, ACC, weight="bold")
s.save(os.path.join(HERE, "fig_mla.svg"))
