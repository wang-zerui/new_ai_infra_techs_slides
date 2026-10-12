"""fig_villain: (a) the logit bound: |q.k| <= |x_i||x_j| ||W_q|| ||W_k||, x bounded by RMSNorm, so growth = spectral norms;
(b) update spectra: Adam's update has a few large singular values, Muon's update has all singular values equal.
Source: K2 App. E (Eq. 2, 'structural difference in updates', 'attention-specific amplification'). Bars illustrative."""
import os
from svgkit import rich, SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 268, "fig_villain")
s.line(400, 14, 400, 262, "#BFBFBF", 1.2)

# (a) bound
s.text(12, 30, "(a) What bounds a logit", 20, INK, anchor="start", weight="bold")
rich(s, 12, 112, [("|q", ""), ("i", "sub"), (" · k", ""), ("j", "sub"), ("| ≤", "")], 22, INK, anchor="start", weight="bold")
s.rect(148, 82, 104, 46, NEU_F, NEU, 2, 4)
rich(s, 200, 112, [("‖x", ""), ("i", "sub"), ("‖ ‖x", ""), ("j", "sub"), ("‖", "")], 22, INK)
s.rect(264, 82, 124, 46, PRB_F, PRB, 2, 4)
rich(s, 326, 112, [("‖W", ""), ("q", "sub"), ("‖ ‖W", ""), ("k", "sub"), ("‖", "")], 22, PRB, weight="bold")
s.text(200, 156, "bounded by", 17, MUTED)
s.text(200, 176, "RMSNorm", 17, MUTED)
s.text(326, 156, "spectral norms", 17, PRB, weight="bold")
s.text(326, 176, "free to grow", 17, PRB, weight="bold")
s.rect(110, 210, 278, 46, "#FFFFFF", PRB, 2, 4, dash="6,4")
rich(s, 249, 239, [("W", ""), ("q", "sub"), ("W", ""), ("k", "sub"), ("ᵀ", ""), (" compounds growth", "")], 21, PRB)
s.line(326, 184, 326, 206, PRB, 2.2, "arrPrb")

# (b) spectra
s.text(412, 30, "(b) Singular values of one update", 20, INK, anchor="start", weight="bold")
def spectrum(x0, vals, color, fill):
    base = 196
    for i, v in enumerate(vals):
        s.rect(x0 + i * 19, base - v, 14, v, fill, color, 1.6, 1)
    s.line(x0 - 6, base, x0 + len(vals) * 19 + 1, base, INK, 1.6)
adam = [118, 70, 38, 22, 14, 10, 8, 6, 5, 4]
muon = [62] * 10
spectrum(424, adam, NEU, NEU_F)
spectrum(658, muon, PRB, PRB_F)
s.text(519, 226, "Adam: few large", 18, INK, weight="bold")
s.text(519, 248, "low effective rank", 17, MUTED)
s.text(752, 226, "Muon: all equal", 18, PRB, weight="bold")
s.text(752, 248, "full rank", 17, MUTED)
s.text(752, 98, "more weight directions", 17, PRB)
s.text(752, 118, "can grow at once", 17, PRB)
s.save(os.path.join(HERE, "fig_villain.svg"))
