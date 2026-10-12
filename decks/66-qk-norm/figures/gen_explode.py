"""fig_explode: why attention logits explode — the chain from unbounded W_q, W_k to divergence.

Bound from Kimi K2 App. E (arXiv 2507.20534, p.30): |q_i.k_j| <= ||x_i|| ||x_j|| ||W_q|| ||W_k||; pre-norm bounds x.
One-hot attention / entropy collapse: ViT-22B Sec. 2 (2302.05442), Wortsman et al. Sec. 3.1.1 (2309.14322).
"""
import os
from svgkit import SVG, rich, ACC, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 240
s = SVG(W, H, "fig_explode: logit growth chain")
Y0, BH = 30, 86
boxes = [  # x, w, fill, stroke
    (2, 170, NEU_F, NEU),
    (200, 150, PRB_F, PRB),
    (378, 196, PRB_F, PRB),
    (602, 120, PRB_F, PRB),
    (750, 117, PRB_F, PRB),
]
for x, w, f, st in boxes:
    s.rect(x, Y0, w, BH, f, st, 2.2, rx=5)
for (x, w, _, _), (nx, _, _, _) in zip(boxes, boxes[1:]):
    s.line(x + w + 2, Y0 + BH / 2, nx - 2, Y0 + BH / 2, PRB if x > 0 else INK, 2.4, "arrPrb" if x > 0 else "arr")
L1, L2 = Y0 + 36, Y0 + 64
rich(s, 86, L1, [("inputs x",), ("i", "sub"), (", x",), ("j", "sub")], 19, weight="bold")
s.text(86, L2, "pre-norm bounded", 17, INK)
rich(s, 275, L1, [("W",), ("q", "sub"), (", W",), ("k", "sub")], 19, weight="bold", color=PRB)
s.text(275, L2, "norms unbounded", 17, PRB)
rich(s, 476, L1, [("logit q",), ("i", "sub"), (" · k",), ("j", "sub"), (" / √d",)], 19, weight="bold", color=PRB)
s.text(476, L2, "grows with the weights", 17, PRB)
s.text(662, L1, "softmax", 19, PRB, weight="bold")
s.text(662, L2, "near one-hot", 17, PRB)
s.text(808, L1, "loss spikes", 19, PRB, weight="bold")
s.text(808, L2, "divergence", 17, PRB)
# annotations under boxes
s.text(275, Y0 + BH + 26, "Muon: full-rank updates", 17, PRB, style="italic")
s.text(662, Y0 + BH + 26, "entropy collapse", 17, PRB, style="italic")
# the bound
s.rect(150, 166, 570, 64, "#FFFFFF", NEU, 1.6, rx=6)
rich(s, 435, 208, [("|q",), ("i", "sub"), (" · k",), ("j", "sub"), ("|  ≤  ",),
                   ("‖x",  "", MUTED), ("i", "sub", MUTED), ("‖ ‖x", "", MUTED), ("j", "sub", MUTED), ("‖",  "", MUTED),
                   ("  ‖W", "", PRB, "bold"), ("q", "sub", PRB, "bold"), ("‖ ‖W", "", PRB, "bold"), ("k", "sub", PRB, "bold"),
                   ("‖", "", PRB, "bold")], 26)
s.save(os.path.join(HERE, "fig_explode.svg"))
