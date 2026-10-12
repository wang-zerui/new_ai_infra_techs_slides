"""fig_algo: MuonClip as pseudocode (paraphrase of K2 Algorithm 1), with three callouts.
Muon: M = mu*M + G; O = NewtonSchulz(M) * 0.2*sqrt(max(n,m)); W = W - eta*(O + lambda*W).
QK-Clip: if S_max[h] > tau: gamma = tau/S_max[h]; W_qc, W_kc *= sqrt(gamma); W_qr *= gamma (shared k_R untouched)."""
import os
from svgkit import rich, SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 268, "fig_algo")
MONO = "Menlo, 'Courier New', monospace"
LH, X0, Y0, FS = 24, 18, 12, 18
lines = [
    ("for each training step t:", INK, 0),
    ("# Muon: every weight matrix W (n×m)", MUTED, 1),
    ("M ← μ·M + G", INK, 1),
    ("O ← NewtonSchulz(M) · 0.2·√max(n,m)", INK, 1),
    ("W ← W − η·(O + λ·W)", INK, 1),
    ("# QK-Clip: every attention head h", ACC, 1),
    ("if S_max[h] > τ:", INK, 1),
    ("γ ← τ / S_max[h]", INK, 2),
    ("W_qc ← √γ·W_qc;  W_kc ← √γ·W_kc", INK, 2),
    ("W_qr ← γ·W_qr    # shared k_R untouched", INK, 2),
]
# panels
s.rect(6, 4, 540, 260, "#FFFFFF", NEU, 1.8, 6)
s.rect(10, 132, 532, 128, ACC_F, "none", 0, 4)
for i, (t, c, ind) in enumerate(lines):
    y = Y0 + i * LH + LH * 0.55 + (6 if i >= 5 else 0)
    s.text(X0 + ind * 2 * 0.6 * FS, y, t.replace("  ", "\u00a0\u00a0"), FS, c, anchor="start", family=MONO,
           weight="bold" if t.startswith("#") else "normal")

def callout(y_line, ytext, l1, l2, color):
    yl = Y0 + y_line * LH + LH * 0.35 + (6 if y_line >= 5 else 0)
    s.line(600, ytext - 6, 552, yl, color, 2, "arrAcc" if color == ACC else "arr")
    s.text(606, ytext - 6, l1, 19, color, anchor="start", weight="bold")
    s.text(606, ytext + 17, l2, 18, INK, anchor="start")

callout(3, 96, "update RMS matched", "to AdamW: reuse its LR", INK)
callout(6, 172, "reuses the forward's", "per-head max logit", ACC)
callout(8.5, 236, "runs after the update", "this step's pass unchanged", ACC)
s.save(os.path.join(HERE, "fig_algo.svg"))
