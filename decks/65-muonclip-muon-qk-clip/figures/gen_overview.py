"""fig_overview: one MuonClip training step. Forward computes the per-head max logit S_max^h (1); backward; Muon update
of every matrix; QK-Clip rescales W_q/W_k of heads with S_max^h > tau (2, MLA-aware); (3) both run inside one optimizer
step, after the update. Source: K2 §2.1, Algorithm 1."""
import os
from svgkit import rich, SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 262, "fig_overview")

def badge(cx, cy, n):
    s.add(f'<circle cx="{cx}" cy="{cy}" r="13" fill="{ACC}"/>')
    s.text(cx, cy + 6, str(n), 17, "#FFFFFF", weight="bold")

def box2(x, y, w, h, l1, l2, fill, stroke, c1=INK):
    s.rect(x, y, w, h, fill, stroke, 2, 4)
    s.text(x + w / 2, y + 27, l1, 20, c1, weight="bold")
    s.text(x + w / 2, y + 51, l2, 17, INK)

Y, H = 96, 64
# MuonClip container
s.rect(392, 78, 404, 124, "#FFFFFF", ACC, 1.8, 6, dash="7,5")
s.text(594, 194, "MuonClip: one optimizer step", 18, ACC, weight="bold")
box2(20, Y, 156, H, "Forward", "logits per head", NEU_F, NEU)
box2(208, Y, 156, H, "Backward", "gradients", NEU_F, NEU)
box2(406, Y, 176, H, "Muon update", "every matrix", NEU_F, NEU)
box2(606, Y, 176, H, "QK-Clip", "heads above τ", ACC_F, ACC, ACC)
s.line(178, Y + H / 2, 205, Y + H / 2, INK, 2.2, "arr")
s.line(366, Y + H / 2, 403, Y + H / 2, INK, 2.2, "arr")
s.line(584, Y + H / 2, 603, Y + H / 2, INK, 2.2, "arr")
# S_max chip and its dashed path to QK-Clip
s.rect(40, 12, 176, 44, ACC_F, ACC, 2, 14)
rich(s, 128, 42, [("S", ""), ("max", "sub"), (" per head", "")], 22, ACC, weight="bold")
s.line(98, Y - 2, 98, 56, ACC, 2, "arrAcc")
s.path("M 218,34 L 694,34 L 694,93", ACC, 2.2, "arrAcc", dash="7,5")
# loop back
s.path("M 784,128 L 846,128 L 846,236 L 98,236 L 98,163", INK, 2.2, "arr")
s.text(472, 230, "next step", 18, MUTED, style="italic", halo=True)
badge(22, 34, 1)
badge(782, 98, 2)
badge(392, 78, 3)
s.save(os.path.join(HERE, "fig_overview.svg"))
